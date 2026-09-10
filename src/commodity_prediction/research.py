"""Fingerprint every stage; compare feature families with a fixed diagnostic model."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .data import load_data, make_folds, reconstruct_targets
from .features import build_candidates, screen_features
from .metrics import correlation_sharpe, daily_rank_correlations, paired_block_interval
from .runtime import RunLog, atomic_json, digest, fingerprint, seal_checkpoint, verify_checkpoint


def lineage_for(root: Path, config: dict) -> tuple[str, dict]:
    paths = [
        *sorted((root / "src/commodity_prediction").glob("*.py")),
        *sorted((root / "data/raw").rglob("*.csv")),
    ]
    environment = {
        p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]
    }
    evidence = {
        "config": config,
        "environment": environment,
        "files": {str(p.relative_to(root)): digest(p) for p in paths},
    }
    return fingerprint(evidence), evidence


def fit_diagnostic(
    x_train: np.ndarray, x_valid: np.ndarray, y: pd.DataFrame, config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    predictions = np.empty((len(x_valid), len(y.columns)))
    coefficients = np.zeros((len(y.columns), x_train.shape[1]))
    intercepts = np.zeros(len(y.columns))
    fallback_count = 0
    for j, target in enumerate(y):
        values = y[target].to_numpy(dtype=float)
        observed = np.isfinite(values)
        if observed.sum() < config["min_target_observations"]:
            fallback_count += 1
            intercepts[j] = float(values[observed].mean()) if observed.any() else 0.0
            predictions[:, j] = intercepts[j]
        else:
            model = Ridge(alpha=config["ridge_alpha"], solver="cholesky")
            model.fit(x_train[observed], values[observed])
            coefficients[j] = model.coef_
            intercepts[j] = model.intercept_
            predictions[:, j] = model.predict(x_valid)
    if not np.isfinite(predictions).all():
        raise ValueError("Diagnostic model produced invalid predictions")
    return predictions, coefficients, intercepts, fallback_count


def run_research(root: Path, sync: bool = False, variants: list[str] | None = None) -> dict:
    config = json.loads((root / "configs/research.json").read_text())
    if config["feature_gate"] != "open":
        raise ValueError(
            "This command runs feature research only; final training needs a reviewed gate"
        )
    lineage, evidence = lineage_for(root, config)
    directory = root / "artifacts" / lineage
    log = RunLog(root / "logs/research.jsonl", config["heartbeat_seconds"])
    if sync:
        from .cloud import restore_run

        restore_run(root, lineage, log)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "lineage.json", evidence)

    def checkpoint(stage: Path) -> None:
        if sync:
            from .cloud import upload_stage

            upload_stage(root, stage, log)

    with log.stage("data_contracts"):
        x, y, pairs = load_data(root)
        folds, development_stop = make_folds(len(x), config)
        # Holdout values never enter EDA, feature construction, screening, or fits.
        development_x = x.iloc[:development_stop]
        development_y = y.iloc[:development_stop]
        reconstructed = reconstruct_targets(development_x, pairs)
        comparable = reconstructed.notna() & development_y.notna()
        difference = (reconstructed - development_y).abs().where(comparable)
        max_error = float(difference.max().max())
        if max_error > 1e-5:
            raise ValueError("Target reconstruction disagrees with the official data")
        origins = pd.Series([c.split("_")[0] for c in x.columns]).value_counts().to_dict()
        data_report = {
            "lineage": lineage,
            "dates": len(x),
            "raw_features": len(x.columns),
            "targets": len(y.columns),
            "development_dates": development_stop,
            "holdout_dates": len(x) - development_stop,
            "holdout_start_date_id": int(x.index[development_stop]),
            "holdout_values_used_for_selection": False,
            "release_delays": {str(lag): lag + 1 for lag in range(1, 5)},
            "target_reconstruction_max_absolute_error": max_error,
            "target_reconstruction_compared_values": int(comparable.sum().sum()),
            "origin_feature_counts": origins,
            "folds": [asdict(fold) for fold in folds],
            "missing_fraction_by_column": development_x.isna().mean().to_dict(),
            "target_availability_by_horizon": {
                str(lag): float(
                    development_y[pairs.loc[pairs.lag == lag, "target"]].notna().mean().mean()
                )
                for lag in range(1, 5)
            },
        }
        atomic_json(directory / "data_audit.json", data_report)
    feature_directory = directory / "features"
    with log.stage("candidate_features"):
        if verify_checkpoint(feature_directory, lineage):
            features = pd.read_parquet(feature_directory / "candidates.parquet")
            families = json.loads((feature_directory / "families.json").read_text())
            log.event(
                "checkpoint_reused", stage="candidate_features", candidates=len(features.columns)
            )
        else:
            features, families = build_candidates(development_x, pairs)
            feature_directory.mkdir(parents=True, exist_ok=True)
            features.to_parquet(feature_directory / "candidates.parquet")
            atomic_json(feature_directory / "families.json", families)
            seal_checkpoint(feature_directory, lineage, ["candidates.parquet", "families.json"])
            checkpoint(feature_directory)
        log.event(
            "candidate_count",
            generated=len(features.columns),
            families=pd.Series(families).value_counts().to_dict(),
        )
    family_names = sorted(set(families.values()) - {"reference"})
    experiments = {
        "reference": ["reference"],
        **{f"add_{f}": ["reference", f] for f in family_names},
        "all_families": ["reference", *family_names],
    }
    if variants is not None:
        experiments = {name: experiments[name] for name in variants}
    results = []
    with threadpool_limits(limits=config["threads"]):
        for fold in folds:
            for name, included in experiments.items():
                stage = directory / f"fold_{fold.number}" / name
                with log.stage(f"fold_{fold.number}/{name}"):
                    if verify_checkpoint(stage, lineage):
                        result = json.loads((stage / "result.json").read_text())
                        log.event("checkpoint_reused", stage=f"fold_{fold.number}/{name}")
                    else:
                        cols = [c for c, family in families.items() if family in included]
                        fit_x, fit_y = (
                            features.iloc[: fold.train_stop][cols],
                            development_y.iloc[: fold.train_stop],
                        )
                        screen = screen_features(fit_x, fit_y, config)
                        validation_x = features.iloc[fold.validation_start : fold.validation_stop]
                        validation_y = development_y.iloc[
                            fold.validation_start : fold.validation_stop
                        ]
                        predicted, coefficients, intercepts, fallback = fit_diagnostic(
                            screen.transform(fit_x), screen.transform(validation_x), fit_y, config
                        )
                        prediction = pd.DataFrame(
                            predicted, index=validation_y.index, columns=validation_y.columns
                        )
                        daily = daily_rank_correlations(validation_y, prediction)
                        stage.mkdir(parents=True, exist_ok=True)
                        prediction.to_parquet(stage / "predictions.parquet")
                        np.savez_compressed(
                            stage / "models.npz",
                            coefficients=coefficients,
                            intercepts=intercepts,
                            columns=np.asarray(screen.columns),
                            medians=screen.medians.to_numpy(),
                            means=screen.means.to_numpy(),
                            scales=screen.scales.to_numpy(),
                        )
                        screen.audit.to_csv(stage / "screening.csv", index=False)
                        horizon_metrics = {}
                        for lag in range(1, 5):
                            columns = pairs.loc[pairs.lag == lag, "target"].tolist()
                            eligible = validation_y[columns].notna().sum(axis=1) >= 2
                            by_horizon = daily_rank_correlations(
                                validation_y.loc[eligible, columns],
                                prediction.loc[eligible, columns],
                            )
                            horizon_metrics[str(lag)] = {
                                "correlation_sharpe": correlation_sharpe(by_horizon),
                                "eligible_dates": int(eligible.sum()),
                                "excluded_dates": int((~eligible).sum()),
                            }
                        result = {
                            "lineage": lineage,
                            "variant": name,
                            "fold": fold.number,
                            "included_families": included,
                            "official_metric": correlation_sharpe(daily),
                            "mean_daily_rank_correlation": float(daily.mean()),
                            "std_daily_rank_correlation": float(daily.std(ddof=0)),
                            "positive_correlation_fraction": float((daily > 0).mean()),
                            "horizon_metrics": horizon_metrics,
                            "daily_rank_correlations": daily.tolist(),
                            "candidate_count": len(cols),
                            "retained_count": len(screen.columns),
                            "rejected_count": len(cols) - len(screen.columns),
                            "retained_family_counts": pd.Series(
                                [families[c] for c in screen.columns]
                            )
                            .value_counts()
                            .to_dict(),
                            "rejection_reasons": screen.audit.loc[
                                screen.audit.status == "rejected", "reason"
                            ]
                            .str.split(":")
                            .str[0]
                            .value_counts()
                            .to_dict(),
                            "fallback_target_count": fallback,
                            "model": "Ridge(alpha=100), fixed diagnostic",
                            "fold_interval": asdict(fold),
                        }
                        atomic_json(stage / "result.json", result)
                        seal_checkpoint(
                            stage,
                            lineage,
                            ["predictions.parquet", "models.npz", "screening.csv", "result.json"],
                        )
                        checkpoint(stage)
                        log.event(
                            "fold_result",
                            variant=name,
                            fold=fold.number,
                            metric=result["official_metric"],
                            retained=len(screen.columns),
                        )
                    results.append(result)
    reference = np.concatenate(
        [np.asarray(r["daily_rank_correlations"]) for r in results if r["variant"] == "reference"]
    )
    comparison: list[dict[str, Any]] = []
    for name in experiments:
        subset = [r for r in results if r["variant"] == name]
        daily = np.concatenate([np.asarray(r["daily_rank_correlations"]) for r in subset])
        metric = correlation_sharpe(daily)
        interval = (
            [0.0, 0.0]
            if name == "reference"
            else paired_block_interval(
                reference,
                daily,
                config["bootstrap_repetitions"],
                config["bootstrap_block_dates"],
                config["seed"],
            )
        )
        comparison.append(
            {
                "variant": name,
                "official_metric": metric,
                "delta_from_reference": metric - correlation_sharpe(reference),
                "conditional_delta_95_interval": interval,
                "fold_metrics": [r["official_metric"] for r in subset],
                "mean_daily_rank_correlation": float(daily.mean()),
                "retained_per_fold": [r["retained_count"] for r in subset],
            }
        )
    summary = {
        "lineage": lineage,
        "phase": "initial_feature_research",
        "feature_gate": "open",
        "candidate_count": len(features.columns),
        "family_counts": pd.Series(families).value_counts().to_dict(),
        "folds_completed": len(folds),
        "experiments_completed": len(results),
        "target_fits_attempted": len(results) * len(y.columns),
        "holdout_evaluated": False,
        "comparison": sorted(comparison, key=lambda r: r["official_metric"], reverse=True),
        "results": results,
        "limitations": [
            "Initial additive family ablations use one fixed linear diagnostic model.",
            "Feature selection is fold-local; no final retained feature set is declared.",
            "Bootstrap intervals are conditional on fitted predictions and unadjusted for multiple comparisons.",
            "Conditional group importance, nonlinear controls, and regime robustness remain open.",
            "Historical validation is not a Kaggle leaderboard score or a trading-profit estimate.",
        ],
    }
    atomic_json(directory / "summary.json", summary)
    # Publish only aggregate results; market rows, labels, and predictions stay private.
    public = root / "reports"
    atomic_json(public / "data_audit.json", data_report)
    atomic_json(public / "research.json", summary)
    atomic_json(public / "lineage.json", evidence)
    if sync:
        from .cloud import upload_stage

        upload_stage(root, directory, log)
    log.event(
        "research_completed",
        lineage=lineage,
        candidate_count=len(features.columns),
        experiments=len(results),
        gate="open",
    )
    return summary
