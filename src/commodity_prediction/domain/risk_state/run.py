"""Bounded matched risk-state experiments with preserved controls and exact replay."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import load_data
from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.compact.run import compact_lineage
from commodity_prediction.domain.diagnostics import temporal_diagnostics
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.risk_state.features import (
    build_panel,
    declared_comparisons,
    experiment_plan,
)
from commodity_prediction.domain.robustness.reporting.run import subgroup_metrics
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


def study_lineage(root: Path) -> tuple[str, dict]:
    parent, parent_evidence = compact_lineage(root)
    config = json.loads((root / "configs/risk_state_study.json").read_text())
    if config["parent_lineage"] != parent or config["feature_gate"] != "open":
        raise ValueError("Risk-state parent lineage or feature gate changed")
    if len(experiment_plan()) * 3 != config["max_new_fits"] or config["residual_weight"] != 1:
        raise ValueError("Matched study budget or residual strength changed")
    evidence = {
        "parent_lineage": parent,
        "feature_lineage": parent_evidence["feature_lineage"],
        "config": config,
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "src/commodity_prediction/domain/risk_state").glob("*.py"))
        },
        "experiments": [asdict(v) for v in experiment_plan()],
        "comparisons": declared_comparisons(),
    }
    return fingerprint(evidence), evidence


def joint_history(root: Path, parent: dict) -> dict[str, np.ndarray]:
    daily = {}
    for prefix, filename in [
        ("domain", "domain_study"),
        ("attribution", "tree_attribution"),
        ("robustness", "domain_robustness"),
        ("compact", "compact_study"),
    ]:
        report = json.loads((root / "reports" / (filename + ".json")).read_text())
        stage = root / "artifacts" / report["lineage"] / "summary"
        if not verify_checkpoint(stage, report["lineage"]):
            raise ValueError("Missing sealed comparison history")
        sealed = json.loads((stage / "summary.json").read_text())
        if sealed != report:
            raise ValueError("Public comparison report differs from sealed history")
        daily.update(
            {
                prefix + "::" + n: np.asarray(s["daily_rank_correlations"])
                for n, s in report["summaries"].items()
            }
        )
    if parent["joint_comparison_count"] != len(
        set((r["variant"], r["reference"]) for r in parent["joint_comparisons"])
    ):
        raise ValueError("Parent comparison inventory differs")
    return daily


def run_study(
    root: Path,
    checkpoint_hook: Callable[[Path], None] | None = None,
    sync: bool = False,
) -> dict:
    started = monotonic()
    lineage, evidence = study_lineage(root)
    config = json.loads((root / "configs/domain_study.json").read_text())
    log = RunLog(root / "logs/risk_state_study.jsonl", config["heartbeat_seconds"])
    directory = root / "artifacts" / lineage
    summary_stage = directory / "summary"
    if sync:
        from commodity_prediction.cloud import restore_run
        from commodity_prediction.studies.storage import StudyStorage

        restore_run(root, lineage, log)
        checkpoint_hook = StudyStorage(root, log).upload_checkpoint

    def checkpoint(stage: Path) -> None:
        if checkpoint_hook is not None:
            checkpoint_hook(stage)

    if verify_checkpoint(summary_stage, lineage):
        with log.stage("verify_complete_risk_state_resume"):
            manifests = sorted(directory.rglob("manifest.json"))
            if len(manifests) != evidence["config"]["max_new_fits"] + 1:
                raise ValueError("Incomplete risk-state checkpoint inventory")
            for manifest in manifests:
                verify_checkpoint(manifest.parent, lineage)
            summary = json.loads((summary_stage / "summary.json").read_text())
            log.event("complete_run_reused", verified_checkpoints=len(manifests))
        atomic_json(root / "reports/risk_state_study.json", summary)
        atomic_json(root / "reports/risk_state_lineage.json", evidence)
        return summary

    feature_dir = root / "artifacts" / evidence["feature_lineage"]
    parent_dir = root / "artifacts" / evidence["parent_lineage"]
    for stage, expected in [
        (feature_dir / "features", evidence["feature_lineage"]),
        (parent_dir / "summary", evidence["parent_lineage"]),
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
        results, summaries = [], {}
        maximum_error = 0.0
        lengths = [f.validation_stop - f.validation_start for f in folds]
        for variant in experiment_plan():
            with log.stage("features/" + variant.name):
                panel = build_panel(original, pairs, variant)
                settings = variant.settings(config, panel)
            predictions = []
            for fold in folds:
                if monotonic() - started >= evidence["config"]["max_run_seconds"]:
                    raise TimeoutError("Risk-state study exceeded its declared runtime budget")
                if (
                    fold.train_stop - 1 + 5 >= fold.validation_start
                    or fold.validation_stop - 1 + 5 >= stop
                ):
                    raise ValueError("Invalid release boundary")
                stage = directory / f"fold_{fold.number}" / variant.name
                with log.stage(f"fold_{fold.number}/{variant.name}"):
                    stats = None
                    if not verify_checkpoint(stage, lineage):
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
                    saved = pd.read_parquet(stage / "predictions.parquet")
                    replay = joblib.load(stage / "model.joblib").predict(
                        panel,
                        fold.validation_start,
                        fold.validation_stop,
                    )
                    error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError("Saved model replay differs")
                    maximum_error = max(maximum_error, error)
                    checkpoint(stage)
                    predictions.append(saved)
                    results.append(result)
                    del stats
            prediction = pd.concat(predictions)
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
                for n in [
                    "historical_mean",
                    "admitted_base",
                    "admitted_tail",
                    "admitted_joint",
                    "screened_tail",
                ]
            }
        )
        with log.stage("joint_risk_state_uncertainty"):
            daily = joint_history(root, parent)
            daily.update(
                {
                    "risk_state::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in summaries.items()
                }
            )
            comparisons = list(
                dict.fromkeys((r["variant"], r["reference"]) for r in parent["joint_comparisons"])
            )
            comparisons += [
                ("risk_state::" + a, "risk_state::" + b) for a, b in declared_comparisons()
            ]
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
                "maximum_prediction_replay_error": maximum_error,
                "checkpoint_count": len(results) + 1,
                "new_distinct_templates": 34,
                "new_template_target_assignments": 34 * len(pairs),
                "summaries": summaries,
                "results": results,
                "joint_comparisons": bounds,
                "joint_comparison_count": len(comparisons),
                "comparisons": [
                    {
                        **r,
                        "variant": r["variant"].removeprefix("risk_state::"),
                        "reference": r["reference"].removeprefix("risk_state::"),
                    }
                    for r in bounds
                    if r["variant"].startswith("risk_state::")
                ],
                "limitations": [
                    "Exploratory after 233 prior domain comparisons; simultaneous bounds condition on fitted models and do not undo adaptive search.",
                    "Risk features transform already released target history; they are proxies, not macro regimes, volatility forecasts, or trading recommendations.",
                    "No model or feature policy is chosen using the weak middle fold alone.",
                    "External calendar, contract and release-vintage mapping remain unavailable; the 247 final origins are untouched.",
                ],
            }
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/risk_state_study.json", summary)
    atomic_json(root / "reports/risk_state_lineage.json", evidence)
    log.event(
        "risk_state_study_completed",
        lineage=lineage,
        new_fitted_models=len(results),
        feature_gate="open",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-s3", action="store_true")
    run_study(Path.cwd(), sync=parser.parse_args().sync_s3)


if __name__ == "__main__":
    main()
