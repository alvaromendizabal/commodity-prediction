"""Execute matched compact feature tests with exact checkpoint reuse and joint uncertainty."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import load_data
from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.compact.features import (
    build_panel,
    declared_comparisons,
    experiment_plan,
)
from commodity_prediction.domain.diagnostics import temporal_diagnostics
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.robustness.reporting.run import analysis_lineage, subgroup_metrics
from commodity_prediction.domain.run import load_panel
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


def compact_lineage(root: Path) -> tuple[str, dict]:
    parent, parent_evidence = analysis_lineage(root)
    config = json.loads((root / "configs/compact_study.json").read_text())
    if (
        config["parent_lineage"] != parent
        or config["feature_gate"] != "open"
        or config["residual_weight"] != 1
    ):
        raise ValueError("Unexpected compact feature contract")
    if len(experiment_plan()) * 3 != config["max_new_fits"]:
        raise ValueError("Experiment budget changed")
    evidence = {
        "parent_lineage": parent,
        "feature_lineage": parent_evidence["feature_lineage"],
        "parent_fitting_lineage": parent_evidence["fitting_lineage"],
        "config": config,
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "src/commodity_prediction/domain/compact").glob("*.py"))
        },
        "experiments": [asdict(v) for v in experiment_plan()],
        "comparisons": declared_comparisons(),
    }
    return fingerprint(evidence), evidence


def run_study(root: Path, sync: bool = False) -> dict:
    lineage, evidence = compact_lineage(root)
    config = json.loads((root / "configs/domain_study.json").read_text())
    log = RunLog(root / "logs/compact_study.jsonl", config["heartbeat_seconds"])
    directory = root / "artifacts" / lineage
    summary_stage = directory / "summary"
    storage = None
    if sync:
        from commodity_prediction.cloud import restore_run
        from commodity_prediction.studies.storage import StudyStorage

        with log.stage("restore_compact_study"):
            restore_run(root, lineage, log)
        storage = StudyStorage(root, log)

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    if verify_checkpoint(summary_stage, lineage):
        with log.stage("verify_complete_compact_resume"):
            manifests = sorted(directory.rglob("manifest.json"))
            if len(manifests) != evidence["config"]["max_new_fits"] + 1:
                raise ValueError("Incomplete compact checkpoint inventory")
            for manifest in manifests:
                verify_checkpoint(manifest.parent, lineage)
                checkpoint(manifest.parent)
            summary = json.loads((summary_stage / "summary.json").read_text())
            log.event("complete_run_reused", verified_checkpoints=len(manifests))
        atomic_json(root / "reports/compact_study.json", summary)
        atomic_json(root / "reports/compact_lineage.json", evidence)
        return summary
    parent_dir = root / "artifacts" / evidence["parent_lineage"]
    feature_dir = root / "artifacts" / evidence["feature_lineage"]
    for stage, expected in [
        (parent_dir / "summary", evidence["parent_lineage"]),
        (feature_dir / "features", evidence["feature_lineage"]),
    ]:
        if not verify_checkpoint(stage, expected):
            raise ValueError("Restore verified parent artifacts first")
    parent = json.loads((parent_dir / "summary/summary.json").read_text())
    atomic_json(directory / "lineage.json", evidence)
    with threadpool_limits(limits=config["threads"]):
        with log.stage("load_verified_development_panel"):
            x, y, pairs = load_data(root)
            folds, stop = study_folds(
                len(x),
                json.loads((root / "configs/research.json").read_text()),
                json.loads((root / "configs/feature_study.json").read_text()),
            )
            y = y.iloc[:stop]
            original = load_panel(feature_dir / "features")
            del x
        results = []
        for fold in folds:
            if (
                fold.train_stop - 1 + 5 >= fold.validation_start
                or fold.validation_stop - 1 + 5 >= stop
            ):
                raise ValueError("Invalid release boundary")
            for variant in experiment_plan():
                stage = directory / f"fold_{fold.number}" / variant.name
                with log.stage(f"fold_{fold.number}/{variant.name}"):
                    panel, stats = original, None
                    settings = config
                    if not verify_checkpoint(stage, lineage):
                        panel = build_panel(original, pairs, variant)
                        settings = variant.settings(config, panel)
                        stats = prepare(panel, y, fold.train_stop, settings)
                    result = fit_stage(
                        panel,
                        y,
                        pairs,
                        fold,
                        Experiment(variant.name, algorithm="histogram"),
                        settings,
                        stats,
                        stage,
                        lineage,
                    )
                    if variant.admit and any(
                        k in result["selection"]["rejection_reasons"]
                        for k in ["feature_budget", "correlated", "unstable_sign"]
                    ):
                        raise ValueError("Admitted feature block was screened out")
                    results.append(result)
                    checkpoint(stage)
                    del panel, stats
                    gc.collect()
        lengths = [f.validation_stop - f.validation_start for f in folds]
        summaries = {}
        max_error = 0.0
        with log.stage("replay_models_and_descriptive_subgroups"):
            for variant in experiment_plan():
                panel = build_panel(original, pairs, variant)
                parts = []
                for fold in folds:
                    stage = directory / f"fold_{fold.number}" / variant.name
                    verify_checkpoint(stage, lineage)
                    saved = pd.read_parquet(stage / "predictions.parquet")
                    replay = joblib.load(stage / "model.joblib").predict(
                        panel, fold.validation_start, fold.validation_stop
                    )
                    error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError("Saved model replay differs")
                    max_error = max(max_error, error)
                    parts.append(saved)
                prediction = pd.concat(parts)
                truth = y.loc[prediction.index]
                metrics = evaluate(truth, prediction, pairs)
                records = [r for r in results if r["variant"] == variant.name]
                summaries[variant.name] = {
                    **metrics,
                    "candidate_templates": len(panel.names),
                    "selection_policy": "admitted_usable" if variant.admit else "screened_64",
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
            summaries.update(
                {
                    n: parent["summaries"][n]
                    for n in ["historical_mean", "compact_control", "all_control"]
                }
            )
            summaries["joint_control"] = parent["summaries"]["compact_risk_freshness"]
        with log.stage("joint_compact_uncertainty"):
            daily = {}
            for prefix, report_path in [
                ("domain", feature_dir / "summary/summary.json"),
                (
                    "attribution",
                    root / "artifacts" / parent["parent_lineage"] / "summary/summary.json",
                ),
                ("robustness", parent_dir / "summary/summary.json"),
            ]:
                expected = report_path.parent.parent.name
                if not verify_checkpoint(report_path.parent, expected):
                    raise ValueError("Missing parent comparison evidence")
                report = json.loads(report_path.read_text())
                daily.update(
                    {
                        prefix + "::" + n: np.asarray(s["daily_rank_correlations"])
                        for n, s in report["summaries"].items()
                    }
                )
            daily.update(
                {
                    "compact::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in summaries.items()
                }
            )
            comparisons = list(
                dict.fromkeys((r["variant"], r["reference"]) for r in parent["joint_comparisons"])
            )
            comparisons += [("compact::" + a, "compact::" + b) for a, b in declared_comparisons()]
            bounds = compare_predictions(daily, lengths, comparisons, config)
            summary = {
                "lineage": lineage,
                "parent_lineage": evidence["parent_lineage"],
                "feature_lineage": evidence["feature_lineage"],
                "feature_gate": "open",
                "holdout_evaluated": False,
                "validation_dates": sum(lengths),
                "new_fitted_models": len(results),
                "model_checkpoints_replayed": len(results),
                "maximum_prediction_replay_error": max_error,
                "checkpoint_count": len(results) + 1,
                "new_distinct_templates": 42,
                "new_template_target_assignments": 42 * len(pairs),
                "reused_product_templates": 12,
                "summaries": summaries,
                "results": results,
                "joint_comparisons": bounds,
                "joint_comparison_count": len(comparisons),
                "comparisons": [
                    {
                        **r,
                        "variant": r["variant"].removeprefix("compact::"),
                        "reference": r["reference"].removeprefix("compact::"),
                    }
                    for r in bounds
                    if r["variant"].startswith("compact::")
                ],
                "limitations": [
                    "Exploratory study chosen after prior development results; no independent final confirmation.",
                    "Admitting usable columns removes relevance and near-correlation exclusions; fixed tree capacity can still limit interaction learning.",
                    "Products reuse the same twelve prior hypotheses; 42 FX-gated state templates condition target representations, not pure FX-leg returns.",
                    "Joint intervals cover declared comparisons in four domain phases conditional on fitted models, not the full adaptive search history.",
                    "Subgroup rankings overlap and have explicit truth-only coverage exclusions; they do not establish global predictive improvement.",
                    "Only 247 final-test origins remain untouched; dated external fundamentals still require verified calendar and vintage mapping.",
                ],
            }
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/compact_study.json", summary)
    atomic_json(root / "reports/compact_lineage.json", evidence)
    log.event(
        "compact_study_completed",
        lineage=lineage,
        new_fitted_models=len(results),
        feature_gate="open",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-s3", action="store_true")
    run_study(Path.cwd(), parser.parse_args().sync_s3)


if __name__ == "__main__":
    main()
