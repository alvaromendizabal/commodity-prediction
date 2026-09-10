"""Fit or reuse feature sensitivities and report coverage-aware subgroup evidence."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import load_data
from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.diagnostics import temporal_diagnostics
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.robustness.features import experiment_plan, transformed_panel
from commodity_prediction.domain.robustness.run import robustness_lineage as fitting_lineage
from commodity_prediction.domain.run import load_panel
from commodity_prediction.metrics import daily_rank_correlations
from commodity_prediction.runtime import (
    RunLog,
    atomic_json,
    digest,
    fingerprint,
    seal_checkpoint,
    verify_checkpoint,
)
from commodity_prediction.studies.evaluation import compare_predictions, evaluate
from commodity_prediction.studies.run import study_folds


def analysis_lineage(root: Path) -> tuple[str, dict]:
    fit_id, fit_evidence = fitting_lineage(root)
    evidence = {
        "fitting_lineage": fit_id,
        "fitting_evidence": fit_evidence,
        "parent_lineage": fit_evidence["parent_lineage"],
        "feature_lineage": fit_evidence["feature_lineage"],
        "config": fit_evidence["config"],
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted(
                (root / "src/commodity_prediction/domain/robustness/reporting").glob("*.py")
            )
        },
        "analysis_contract": "Global official metric unchanged; subgroup days need two labels; undefined subgroup and horizon scores remain null with coverage.",
    }
    return fingerprint(evidence), evidence


def subgroup_score(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    eligible = truth.notna().sum(axis=1) >= 2
    result: dict[str, float | str | None] = {
        "official_metric": None,
        "mean_daily_rank_correlation": None,
        "eligible_dates": int(eligible.sum()),
        "excluded_dates": int((~eligible).sum()),
        "undefined_reason": None,
    }
    if eligible.sum() < 2:
        result["undefined_reason"] = "fewer_than_two_eligible_dates"
        return result
    try:
        daily = daily_rank_correlations(truth.loc[eligible], prediction.loc[eligible])
    except ZeroDivisionError:
        result["undefined_reason"] = "constant_daily_target_or_prediction"
        return result
    result["mean_daily_rank_correlation"] = float(daily.mean())
    if daily.std(ddof=0) == 0:
        result["undefined_reason"] = "zero_daily_correlation_variance"
    else:
        result["official_metric"] = float(daily.mean() / daily.std(ddof=0))
    return result


def subgroup_metrics(truth: pd.DataFrame, prediction: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    groups = {"single": ~pairs.pair.str.contains(" - "), "paired": pairs.pair.str.contains(" - ")}
    for market in ["FX", "US", "LME", "JPX"]:
        groups[market + "_involved"] = pairs.pair.map(
            lambda value, market=market: any(a.startswith(market + "_") for a in value.split(" - "))
        )
    result = {}
    for name, mask in groups.items():
        subset = pairs.loc[mask]
        if len(subset) >= 2:
            result[name] = {
                "targets": len(subset),
                **subgroup_score(truth[subset.target], prediction[subset.target]),
                "horizon_metrics": {
                    str(h): subgroup_score(truth[g.target], prediction[g.target])
                    for h, g in subset.groupby("lag")
                },
            }
    return result


def run_study(root: Path, sync: bool = False) -> dict:
    lineage, evidence = analysis_lineage(root)
    fit_id = evidence["fitting_lineage"]
    parent, feature = evidence["parent_lineage"], evidence["feature_lineage"]
    config = json.loads((root / "configs/domain_study.json").read_text())
    log = RunLog(root / "logs/domain_robustness.jsonl", config["heartbeat_seconds"])
    directory = root / "artifacts" / fit_id
    analysis_directory = root / "artifacts" / lineage
    summary_stage = analysis_directory / "summary"
    storage = None
    if sync:
        from commodity_prediction.cloud import restore_run
        from commodity_prediction.studies.storage import StudyStorage

        with log.stage("restore_robustness"):
            restore_run(root, fit_id, log)
            restore_run(root, lineage, log)
        storage = StudyStorage(root, log)

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    if verify_checkpoint(summary_stage, lineage):
        with log.stage("verify_complete_robustness_resume"):
            manifests = sorted(directory.rglob("manifest.json")) + sorted(
                analysis_directory.rglob("manifest.json")
            )
            if len(manifests) != 37:
                raise ValueError("Incomplete robustness checkpoint inventory")
            for manifest in manifests:
                verify_checkpoint(
                    manifest.parent,
                    lineage if manifest.is_relative_to(analysis_directory) else fit_id,
                )
                checkpoint(manifest.parent)
            summary = json.loads((summary_stage / "summary.json").read_text())
            log.event("complete_run_reused", verified_checkpoints=len(manifests))
        atomic_json(root / "reports/domain_robustness.json", summary)
        atomic_json(root / "reports/domain_robustness_lineage.json", evidence)
        return summary
    feature_dir = root / "artifacts" / feature
    attr_dir = root / "artifacts" / parent
    for stage, expected in [
        (feature_dir / "features", feature),
        (feature_dir / "summary", feature),
        (attr_dir / "summary", parent),
    ]:
        if not verify_checkpoint(stage, expected):
            raise ValueError("Restore verified parent study artifacts first")
    domain_report = json.loads((feature_dir / "summary/summary.json").read_text())
    attr_report = json.loads((attr_dir / "summary/summary.json").read_text())
    atomic_json(directory / "lineage.json", evidence["fitting_evidence"])
    atomic_json(analysis_directory / "lineage.json", evidence)
    with threadpool_limits(limits=config["threads"]):
        with log.stage("reuse_verified_parent_panel"):
            x, y, pairs = load_data(root)
            folds, stop = study_folds(
                len(x),
                json.loads((root / "configs/research.json").read_text()),
                json.loads((root / "configs/feature_study.json").read_text()),
            )
            y = y.iloc[:stop]
            parent_panel = load_panel(feature_dir / "features")
            del x
        results = []
        for fold in folds:
            if (
                fold.train_stop - 1 + 5 >= fold.validation_start
                or fold.validation_stop - 1 + 5 >= stop
            ):
                raise ValueError("Invalid release boundary")
            shared_stats = None
            for variant in experiment_plan():
                stage = directory / f"fold_{fold.number}" / variant.name
                with log.stage(f"fold_{fold.number}/{variant.name}"):
                    current_config = {
                        **config,
                        "max_features": variant.max_features,
                        "max_abs_correlation": variant.max_correlation,
                    }
                    panel, stats = None, None
                    if not verify_checkpoint(stage, fit_id):
                        panel = transformed_panel(parent_panel, pairs, variant)
                        if variant.transform == "identity" and not variant.families:
                            if shared_stats is None:
                                shared_stats = prepare(panel, y, fold.train_stop, current_config)
                            stats = shared_stats
                        else:
                            stats = prepare(panel, y, fold.train_stop, current_config)
                    # A resumed fit_stage returns its sealed result before using inputs.
                    result = fit_stage(
                        parent_panel if panel is None else panel,
                        y,
                        pairs,
                        fold,
                        Experiment(variant.name, algorithm="histogram"),
                        current_config,
                        stats,
                        stage,
                        fit_id,
                    )
                    checkpoint(stage)
                    results.append(result)
                    del panel, stats
                    gc.collect()
            del shared_stats
            gc.collect()
        lengths = [f.validation_stop - f.validation_start for f in folds]
        summaries = {}
        max_error = 0.0
        with log.stage("replay_models_and_subgroup_diagnostics"):
            for variant in experiment_plan():
                panel = transformed_panel(parent_panel, pairs, variant)
                predictions = []
                for fold in folds:
                    stage = directory / f"fold_{fold.number}" / variant.name
                    verify_checkpoint(stage, fit_id)
                    saved = pd.read_parquet(stage / "predictions.parquet")
                    replay = joblib.load(stage / "model.joblib").predict(
                        panel, fold.validation_start, fold.validation_stop
                    )
                    error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError("Saved model replay differs")
                    max_error = max(max_error, error)
                    predictions.append(saved)
                prediction = pd.concat(predictions)
                truth = y.loc[prediction.index]
                metrics = evaluate(truth, prediction, pairs)
                records = [r for r in results if r["variant"] == variant.name]
                summaries[variant.name] = {
                    **metrics,
                    "candidate_templates": len(panel.names),
                    "fold_scores": [r["metrics"]["official_metric"] for r in records],
                    "subgroups": subgroup_metrics(truth, prediction, pairs),
                    **temporal_diagnostics(
                        np.asarray(metrics["daily_rank_correlations"]),
                        lengths,
                        [r["selection"]["selected_names"] for r in records],
                    ),
                }
                del panel
                gc.collect()
            for name, folder, experiment, file in [
                ("all_control", feature_dir, "outer/tree_all", "raw_predictions.parquet"),
                ("historical_mean", feature_dir, "outer/historical_mean", "predictions.parquet"),
                ("compact_control", attr_dir, "add_released_priors", "predictions.parquet"),
            ]:
                source = (
                    attr_report["summaries"]["add_released_priors"]
                    if name == "compact_control"
                    else domain_report["summaries"][
                        "raw__tree_all" if name == "all_control" else name
                    ]
                )
                # The sealed parent reports supply controls; subgroup diagnostics use saved predictions.
                saved_parts = []
                for fold in folds:
                    stage = folder / f"fold_{fold.number}" / experiment
                    if not verify_checkpoint(stage, parent if folder == attr_dir else feature):
                        raise ValueError("Missing matched control checkpoint")
                    saved_parts.append(pd.read_parquet(stage / file))
                saved = pd.concat(saved_parts)
                summaries[name] = {
                    **source,
                    "subgroups": subgroup_metrics(y.loc[saved.index], saved, pairs),
                }
        with log.stage("joint_robustness_uncertainty"):
            daily = {
                "domain::" + n: np.asarray(s["daily_rank_correlations"])
                for n, s in domain_report["summaries"].items()
            }
            daily.update(
                {
                    "attribution::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in attr_report["summaries"].items()
                }
            )
            daily.update(
                {
                    "robustness::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in summaries.items()
                }
            )
            comparisons = list(
                dict.fromkeys(
                    (r["variant"], r["reference"]) for r in attr_report["joint_comparisons"]
                )
            )
            for variant in experiment_plan():
                comparisons += [
                    ("robustness::" + variant.name, "robustness::" + c)
                    for c in ["historical_mean", "all_control"]
                ]
            comparisons += [
                ("robustness::" + n, "robustness::compact_control")
                for n in ["compact_conditional", "compact_factor", "compact_risk_freshness"]
            ]
            bounds = compare_predictions(daily, lengths, comparisons, config)
            summary = {
                "lineage": lineage,
                "fitting_lineage": fit_id,
                "parent_lineage": parent,
                "feature_lineage": feature,
                "feature_gate": "open",
                "holdout_evaluated": False,
                "validation_dates": sum(lengths),
                "new_fitted_models": len(results),
                "model_checkpoints_replayed": len(results),
                "maximum_prediction_replay_error": max_error,
                "checkpoint_count": len(results) + 1,
                "new_distinct_interaction_templates": 68,
                "new_template_target_assignments": 68 * len(pairs),
                "summaries": summaries,
                "results": results,
                "joint_comparisons": bounds,
                "joint_comparison_count": len(comparisons),
                "comparisons": [
                    {
                        **r,
                        "variant": r["variant"].removeprefix("robustness::"),
                        "reference": r["reference"].removeprefix("robustness::"),
                    }
                    for r in bounds
                    if r["variant"].startswith("robustness::")
                ],
                "limitations": [
                    "Exploratory sensitivities selected after earlier development results; no independent confirmation.",
                    "Fixed unshrunk tree settings isolate feature representation; the best point estimate is not a selected final model.",
                    "Subgroup metrics overlap and rank different target subsets; they are descriptive and not directly comparable to the global metric.",
                    "Joint bounds cover three domain phases conditional on fitted models, not all research adaptivity.",
                    "Extra delays are conservative stress tests; robustness cannot establish historical exchange-session timestamps.",
                    "Subgroup analysis uses truth-based eligibility; unavailable market days are excluded and single-target horizon scores remain undefined. Global metrics use all 535 dates.",
                ],
            }
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/domain_robustness.json", summary)
    atomic_json(root / "reports/domain_robustness_lineage.json", evidence)
    log.event(
        "domain_robustness_completed",
        lineage=lineage,
        new_fitted_models=len(results),
        feature_gate="open",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    run_study(Path.cwd(), args.sync_s3)


if __name__ == "__main__":
    main()
