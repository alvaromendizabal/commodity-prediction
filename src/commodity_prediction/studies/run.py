"""Execute the controlled target-aware study: python -m commodity_prediction.studies.run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import load_data, make_folds
from commodity_prediction.features import price_columns
from commodity_prediction.research import lineage_for
from commodity_prediction.runtime import (
    RunLog,
    atomic_json,
    digest,
    fingerprint,
    seal_checkpoint,
    verify_checkpoint,
)

from .evaluation import (
    compare_predictions,
    evaluate,
    group_permutations,
    selection_stability,
    selection_summary,
)
from .models import ControlBundle, ModelBundle, fit_output, projection_for
from .representation import (
    asset_labels,
    existing_metadata,
    extend_candidates,
    metadata_from_dict,
    metadata_to_dict,
    routed_columns,
)
from .screening import prepare_features, relevance_scores, select_features


@dataclass(frozen=True)
class Experiment:
    name: str
    scope: str
    representation: str
    stable: bool = True
    algorithm: str = "ridge"
    drop_family: str = ""


def experiment_plan() -> list[Experiment]:
    original_families = [
        "momentum",
        "volatility",
        "reversion",
        "missingness",
        "liquidity",
        "ohlc",
        "cross_market",
        "pairs",
    ]
    return [
        Experiment("historical_mean", "control", "historical_mean"),
        Experiment("released_mean", "control", "released_mean"),
        Experiment("target_reference", "global", "reference", stable=False),
        Experiment("target_all", "global", "original", stable=False),
        Experiment("target_stable", "global", "original"),
        Experiment("aligned_reference", "aligned", "reference"),
        Experiment("aligned_original", "aligned", "original"),
        Experiment("aligned_extended", "aligned", "extended"),
        *[
            Experiment(f"drop_{family}", "aligned", "extended", drop_family=family)
            for family in [*original_families, "regime", "interactions", "release_history"]
        ],
        Experiment("structural_reference", "structural", "reference"),
        Experiment("structural_extended", "structural", "extended"),
        Experiment("hist_reference", "aligned", "reference", algorithm="histogram"),
        Experiment("hist_extended", "aligned", "extended", algorithm="histogram"),
    ]


def study_lineage(root: Path, config: dict) -> tuple[str, dict]:
    parent_config = json.loads((root / "configs/research.json").read_text())
    parent, _ = lineage_for(root, parent_config)
    if parent != config["parent_lineage"]:
        raise ValueError(
            "Initial source/data/configuration lineage changed; review the parent before reuse"
        )
    files = sorted((root / "src/commodity_prediction/studies").glob("*.py"))
    evidence = {
        "parent_lineage": parent,
        "config": config,
        "files": {str(p.relative_to(root)): digest(p) for p in files},
        "experiments": [asdict(e) for e in experiment_plan()],
    }
    return fingerprint(evidence), evidence


def declared_comparisons(names: list[str]) -> list[tuple[str, str]]:
    comparisons = [(name, "historical_mean") for name in names if name != "historical_mean"]
    comparisons += [
        ("target_all", "target_reference"),
        ("target_stable", "target_all"),
        ("aligned_original", "aligned_reference"),
        ("aligned_extended", "aligned_original"),
        ("structural_extended", "structural_reference"),
        ("hist_extended", "hist_reference"),
        ("aligned_extended", "released_mean"),
    ]
    comparisons += [("aligned_extended", name) for name in names if name.startswith("drop_")]
    return comparisons


def study_folds(n_dates: int, parent_config: dict, config: dict) -> tuple[list, int]:
    folds, stop = make_folds(n_dates, parent_config)
    embargo = config["terminal_embargo_dates"]
    if embargo < 5 or embargo >= parent_config["validation_dates"]:
        raise ValueError("Terminal embargo must cover every target's release delay")
    folds[-1] = replace(folds[-1], validation_stop=stop - embargo)
    return folds, stop


def run_study(root: Path, sync: bool = False) -> dict:
    config = json.loads((root / "configs/feature_study.json").read_text())
    if config["feature_gate"] != "open":
        raise ValueError("This command is a feature study, not final model training")
    lineage, evidence = study_lineage(root, config)
    parent = config["parent_lineage"]
    directory = root / "artifacts" / lineage
    log = RunLog(root / "logs/feature_study.jsonl", config["heartbeat_seconds"])
    storage = None
    if sync:
        from commodity_prediction.cloud import restore_run

        from .storage import StudyStorage

        with log.stage("restore_study_checkpoints"):
            restore_run(root, lineage, log)
        storage = StudyStorage(root, log)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "lineage.json", evidence)

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    parent_features = root / "artifacts" / parent / "features"
    if not verify_checkpoint(parent_features, parent):
        raise ValueError("Missing verified initial features; restore the parent checkpoint")
    plan = experiment_plan()
    parent_config = json.loads((root / "configs/research.json").read_text())
    with log.stage("study_data_contracts"):
        x, y, pairs = load_data(root)
        folds, stop = study_folds(len(x), parent_config, config)
        x, y = x.iloc[:stop], y.iloc[:stop]
        prices = price_columns(x)
        structural_y = asset_labels(x, [1, 2, 3, 4])
        original = pd.read_parquet(parent_features / "candidates.parquet")
        if not original.index.equals(x.index):
            raise ValueError("Parent candidates do not cover exactly the development interval")
        metadata = existing_metadata(list(original.columns), prices, pairs)
        original_names = set(original.columns)
    feature_stage = directory / "features"
    with log.stage("extended_candidates"):
        if verify_checkpoint(feature_stage, lineage):
            extended = pd.read_parquet(feature_stage / "candidates.parquet")
            extra_metadata = metadata_from_dict(
                json.loads((feature_stage / "metadata.json").read_text())
            )
            log.event("checkpoint_reused", stage="extended_candidates")
            checkpoint(feature_stage)
        else:
            extended, extra_metadata = extend_candidates(x, y, pairs)
            feature_stage.mkdir(parents=True, exist_ok=True)
            extended.to_parquet(feature_stage / "candidates.parquet")
            atomic_json(feature_stage / "metadata.json", metadata_to_dict(extra_metadata))
            seal_checkpoint(feature_stage, lineage, ["candidates.parquet", "metadata.json"])
            checkpoint(feature_stage)
        metadata.update(extra_metadata)
        features = pd.concat([original, extended], axis=1)
        if not features.columns.is_unique or not features.index.equals(x.index):
            raise ValueError("Invalid extended feature schema")
        log.event(
            "candidate_count",
            reused=len(original.columns),
            added=len(extended.columns),
            total=len(features.columns),
        )
    target_routes = {
        row.target: routed_columns(metadata, tuple(row.pair.split(" - ")), row.pair, row.target)
        for row in pairs.itertuples(index=False)
    }
    asset_routes = {asset: routed_columns(metadata, (asset,)) for asset in prices}
    results = []
    audits_by_variant: dict[str, list[list[dict]]] = {e.name: [] for e in plan}
    with threadpool_limits(limits=config["threads"]):
        for fold in folds:
            fit_y = y.iloc[: fold.train_stop]
            fit_structural = structural_y.iloc[: fold.train_stop]
            valid = features.iloc[fold.validation_start : fold.validation_stop]
            truth = y.iloc[fold.validation_start : fold.validation_stop]
            if fold.train_stop - 1 + 5 >= fold.validation_start:
                raise ValueError(
                    "A structural training label is unavailable at the prediction boundary"
                )
            pending = [
                e
                for e in plan
                if not verify_checkpoint(directory / f"fold_{fold.number}" / e.name, lineage)
            ]
            prepared = None
            scores = None
            if any(e.scope != "control" for e in pending):
                with log.stage(f"fold_{fold.number}/train_only_screening_statistics"):
                    prepared = prepare_features(features.iloc[: fold.train_stop], valid, config)
                    scores = relevance_scores(prepared.train, fit_y.to_numpy(dtype=float), config)
                    positions = {name: i for i, name in enumerate(prepared.columns)}
            for experiment in plan:
                stage = directory / f"fold_{fold.number}" / experiment.name
                with log.stage(f"fold_{fold.number}/{experiment.name}"):
                    if verify_checkpoint(stage, lineage):
                        result = json.loads((stage / "result.json").read_text())
                        audits = json.loads((stage / "selection.json").read_text())
                        log.event(
                            "checkpoint_reused", stage=f"fold_{fold.number}/{experiment.name}"
                        )
                        checkpoint(stage)
                    else:
                        audits = []
                        bundle: ModelBundle | ControlBundle
                        if experiment.scope == "control":
                            bundle = ControlBundle(
                                list(y.columns),
                                fit_y.mean().fillna(0).to_numpy(),
                                experiment.name == "released_mean",
                            )
                        else:
                            if prepared is None or scores is None:
                                raise ValueError("Missing fold-local preprocessing")
                            structural = experiment.scope == "structural"
                            output_y = fit_structural if structural else fit_y
                            models = []
                            for j, output in enumerate(output_y.columns):
                                if structural:
                                    asset = output.split(":", 1)[1]
                                    pool = asset_routes[asset]
                                else:
                                    pool = (
                                        list(features.columns)
                                        if experiment.scope == "global"
                                        else target_routes[output]
                                    )
                                candidates = [
                                    name
                                    for name in pool
                                    if (
                                        experiment.representation != "reference"
                                        or metadata[name].family == "reference"
                                    )
                                    and (
                                        experiment.representation != "original"
                                        or name in original_names
                                    )
                                    and metadata[name].family != experiment.drop_family
                                ]
                                if structural:
                                    usable = [
                                        positions[name]
                                        for name in candidates
                                        if name not in prepared.rejected
                                    ]
                                    local_scores = relevance_scores(
                                        prepared.train[:, usable],
                                        output_y[[output]].to_numpy(dtype=float),
                                        config,
                                    )
                                    relevance = np.zeros(len(prepared.columns))
                                    relevance[usable] = local_scores[int(experiment.stable)][:, 0]
                                else:
                                    relevance = scores[int(experiment.stable)][:, j]
                                selected, audit = select_features(
                                    prepared, candidates, relevance, config, experiment.stable
                                )
                                audit["output"] = output
                                model = fit_output(
                                    prepared,
                                    output_y[output].to_numpy(dtype=float),
                                    selected,
                                    experiment.algorithm,
                                    config,
                                )
                                audit["constant_fallback"] = model.estimator is None
                                models.append(model)
                                audits.append(audit)
                                if (j + 1) % 100 == 0:
                                    log.event(
                                        "model_progress",
                                        variant=experiment.name,
                                        fold=fold.number,
                                        completed=j + 1,
                                        total=len(output_y.columns),
                                    )
                            bundle = ModelBundle(
                                prepared.columns,
                                prepared.medians,
                                prepared.means,
                                prepared.scales,
                                list(output_y.columns),
                                list(y.columns),
                                models,
                                projection_for(pairs, list(output_y.columns), structural),
                            )
                        prediction = bundle.predict(valid)
                        result = {
                            "lineage": lineage,
                            "parent_lineage": parent,
                            "variant": experiment.name,
                            "fold": fold.number,
                            "fold_interval": asdict(fold),
                            "specification": asdict(experiment),
                            **evaluate(truth, prediction, pairs),
                            "screening": selection_summary(audits, metadata) if audits else None,
                            "constant_output_count": sum(a["constant_fallback"] for a in audits)
                            if audits
                            else len(y.columns),
                        }
                        stage.mkdir(parents=True, exist_ok=True)
                        joblib.dump(bundle, stage / "model.joblib", compress=3)
                        # Reload exactly what downstream consumers will use, and verify every prediction.
                        loaded = joblib.load(stage / "model.joblib")
                        replayed = loaded.predict(valid)
                        np.testing.assert_allclose(
                            replayed.to_numpy(), prediction.to_numpy(), rtol=0, atol=1e-12
                        )
                        prediction.to_parquet(stage / "predictions.parquet")
                        atomic_json(stage / "selection.json", audits)
                        atomic_json(stage / "result.json", result)
                        seal_checkpoint(
                            stage,
                            lineage,
                            [
                                "model.joblib",
                                "predictions.parquet",
                                "selection.json",
                                "result.json",
                            ],
                        )
                        checkpoint(stage)
                        log.event(
                            "fold_result",
                            variant=experiment.name,
                            fold=fold.number,
                            metric=result["official_metric"],
                            constant_outputs=result["constant_output_count"],
                        )
                    results.append(result)
                    audits_by_variant[experiment.name].append(audits)
            permutation_stage = directory / f"fold_{fold.number}" / "group_permutation"
            with log.stage(f"fold_{fold.number}/group_permutation"):
                if verify_checkpoint(permutation_stage, lineage):
                    log.event("checkpoint_reused", stage=f"fold_{fold.number}/group_permutation")
                    checkpoint(permutation_stage)
                else:
                    model_path = (
                        directory / f"fold_{fold.number}" / "aligned_extended" / "model.joblib"
                    )
                    permutation_bundle = joblib.load(model_path)
                    if not isinstance(permutation_bundle, ModelBundle):
                        raise ValueError("Group permutation requires a fitted feature model")
                    permutation = group_permutations(
                        permutation_bundle, valid, truth, metadata, config
                    )
                    atomic_json(permutation_stage / "importance.json", permutation)
                    seal_checkpoint(permutation_stage, lineage, ["importance.json"])
                    checkpoint(permutation_stage)
    with log.stage("study_evaluation"):
        daily = {
            e.name: np.concatenate(
                [
                    np.asarray(r["daily_rank_correlations"])
                    for r in results
                    if r["variant"] == e.name
                ]
            )
            for e in plan
        }
        comparison: list[dict[str, Any]] = []
        for e in plan:
            subset = [r for r in results if r["variant"] == e.name]
            values = daily[e.name]
            comparison.append(
                {
                    "variant": e.name,
                    "official_metric": float(values.mean() / values.std(ddof=0)),
                    "mean_daily_rank_correlation": float(values.mean()),
                    "positive_correlation_fraction": float((values > 0).mean()),
                    "fold_metrics": [r["official_metric"] for r in subset],
                    "period_metrics": [
                        {
                            "fold": r["fold"],
                            "block": start // 60,
                            "dates": len(r["daily_rank_correlations"][start : start + 60]),
                            "official_metric": float(
                                np.mean(r["daily_rank_correlations"][start : start + 60])
                                / np.std(r["daily_rank_correlations"][start : start + 60], ddof=0)
                            ),
                        }
                        for r in subset
                        for start in range(0, len(r["daily_rank_correlations"]), 60)
                    ],
                    "selection_stability": selection_stability(audits_by_variant[e.name]),
                }
            )
        intervals = compare_predictions(
            daily,
            [f.validation_stop - f.validation_start for f in folds],
            declared_comparisons([e.name for e in plan]),
            config,
        )
        permutation = {
            f"fold_{f.number}": json.loads(
                (directory / f"fold_{f.number}" / "group_permutation/importance.json").read_text()
            )
            for f in folds
        }
        rescored_parent = []
        initial_report = json.loads((root / "reports/research.json").read_text())
        for initial in initial_report["comparison"]:
            predictions = []
            truth_blocks = []
            for fold in folds:
                stage = root / "artifacts" / parent / f"fold_{fold.number}" / initial["variant"]
                if not verify_checkpoint(stage, parent):
                    raise ValueError("Initial comparison checkpoint missing")
                predictions.append(
                    pd.read_parquet(stage / "predictions.parquet").loc[
                        y.index[fold.validation_start : fold.validation_stop]
                    ]
                )
                truth_blocks.append(y.iloc[fold.validation_start : fold.validation_stop])
            rescored_parent.append(
                {
                    "variant": initial["variant"],
                    **evaluate(pd.concat(truth_blocks), pd.concat(predictions), pairs),
                }
            )
        report = {
            "lineage": lineage,
            "parent_lineage": parent,
            "phase": "target_aware_feature_research",
            "feature_gate": "open",
            "holdout_evaluated": False,
            "candidate_count": len(features.columns),
            "reused_candidate_count": len(original.columns),
            "new_candidate_count": len(extended.columns),
            "family_counts": pd.Series({name: info.family for name, info in metadata.items()})
            .value_counts()
            .to_dict(),
            "experiments_completed": len(results),
            "variants_completed": len(plan),
            "folds_completed": len(folds),
            "development_dates": stop,
            "validation_dates": sum(f.validation_stop - f.validation_start for f in folds),
            "terminal_embargo_dates": config["terminal_embargo_dates"],
            "fold_intervals": [asdict(f) for f in folds],
            "comparison": sorted(comparison, key=lambda r: r["official_metric"], reverse=True),
            "results": results,
            "paired_intervals": intervals,
            "group_permutation": permutation,
            "initial_variants_rescored_on_common_dates": rescored_parent,
            "limitations": [
                "All scores are historical development estimates; the reserved final 252 dates remain untouched.",
                "Bootstrap intervals condition on fitted models; simultaneous bounds cover this declared comparison set, not prior adaptive research.",
                "Group permutation measures reliance under block perturbations; correlated groups can substitute and the perturbation may leave the data manifold.",
                "No broad hyperparameter search, portfolio trading simulation, or state-of-the-art claim is justified by this study.",
            ],
        }
        atomic_json(directory / "summary.json", report)
        seal_checkpoint(directory, lineage, ["summary.json", "lineage.json"])
        # Upload these files only; completed child checkpoints have already been synchronized.
        checkpoint(directory)
        atomic_json(root / "reports/feature_study.json", report)
        atomic_json(root / "reports/feature_study_lineage.json", evidence)
        log.event("study_completed", lineage=lineage, experiments=len(results), feature_gate="open")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    run_study(args.root.resolve(), args.sync_s3)


if __name__ == "__main__":
    main()
