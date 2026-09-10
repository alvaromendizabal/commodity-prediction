"""Attribute the observed nonlinear signal without refitting parent experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold, load_data
from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.diagnostics import group_permutations, temporal_diagnostics
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.experiment import declared_comparisons as parent_comparisons
from commodity_prediction.domain.features import FAMILIES
from commodity_prediction.domain.model import Statistics, fit, prepare, select
from commodity_prediction.domain.run import domain_lineage, load_panel
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


def experiment_plan() -> list[Experiment]:
    return [
        Experiment(f"add_{family}", include=family, algorithm="histogram") for family in FAMILIES
    ] + [Experiment(f"drop_{family}", drop=family, algorithm="histogram") for family in FAMILIES]


def attribution_lineage(root: Path) -> tuple[str, dict]:
    config = json.loads((root / "configs/tree_attribution.json").read_text())
    parent, _ = domain_lineage(root, json.loads((root / "configs/domain_study.json").read_text()))
    if (
        config["parent_lineage"] != parent
        or config["feature_gate"] != "open"
        or config["residual_weight"] != 1
    ):
        raise ValueError("Unexpected attribution configuration or parent lineage")
    files = sorted((root / "src/commodity_prediction/domain/attribution").glob("*.py"))
    evidence = {
        "parent_lineage": parent,
        "config": config,
        "files": {str(p.relative_to(root)): digest(p) for p in files},
        "experiments": [asdict(e) for e in experiment_plan()],
    }
    return fingerprint(evidence), evidence


def fit_stage(
    panel: Panel,
    y: pd.DataFrame,
    pairs: pd.DataFrame,
    fold: Fold,
    experiment: Experiment,
    config: dict,
    stats: Statistics | None,
    stage: Path,
    lineage: str,
) -> dict:
    if verify_checkpoint(stage, lineage):
        return json.loads((stage / "result.json").read_text())
    if stats is None:
        raise ValueError("Missing fitting statistics")
    selected, audit = select(stats, experiment.candidates(panel.names), config, False)
    model = fit(stats, selected, panel, y, "histogram", config)
    truth = y.iloc[fold.validation_start : fold.validation_stop]
    prediction = model.predict(panel, fold.validation_start, fold.validation_stop)
    prediction.index.name = truth.index.name
    result = {
        "variant": experiment.name,
        "fold": fold.number,
        "train_stop": fold.train_stop,
        "validation_start": fold.validation_start,
        "validation_stop": fold.validation_stop,
        "selected_weight": 1.0,
        "selection": audit,
        "metrics": evaluate(truth, prediction, pairs),
    }
    stage.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, stage / "model.joblib", compress=3)
    replay = joblib.load(stage / "model.joblib").predict(
        panel, fold.validation_start, fold.validation_stop
    )
    np.testing.assert_array_equal(replay.to_numpy(), prediction.to_numpy())
    prediction.to_parquet(stage / "predictions.parquet")
    atomic_json(stage / "result.json", result)
    seal_checkpoint(stage, lineage, ["model.joblib", "predictions.parquet", "result.json"])
    return result


def run_study(root: Path, sync: bool = False) -> dict:
    lineage, evidence = attribution_lineage(root)
    parent = evidence["parent_lineage"]
    config = json.loads((root / "configs/domain_study.json").read_text())
    log = RunLog(root / "logs/tree_attribution.jsonl", config["heartbeat_seconds"])
    directory, parent_dir = root / "artifacts" / lineage, root / "artifacts" / parent
    storage = None
    if sync:
        from commodity_prediction.cloud import restore_run
        from commodity_prediction.studies.storage import StudyStorage

        with log.stage("restore_tree_attribution"):
            restore_run(root, lineage, log)
        storage = StudyStorage(root, log)

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    summary_stage = directory / "summary"
    if verify_checkpoint(summary_stage, lineage):
        with log.stage("verify_complete_attribution_resume"):
            manifests = sorted(directory.rglob("manifest.json"))
            if len(manifests) != 82:
                raise ValueError("Incomplete attribution checkpoint inventory")
            for manifest in manifests:
                verify_checkpoint(manifest.parent, lineage)
                checkpoint(manifest.parent)
            summary = json.loads((summary_stage / "summary.json").read_text())
            log.event("complete_run_reused", verified_checkpoints=len(manifests))
        atomic_json(root / "reports/tree_attribution.json", summary)
        atomic_json(root / "reports/tree_attribution_lineage.json", evidence)
        return summary
    if not verify_checkpoint(parent_dir / "summary", parent) or not verify_checkpoint(
        parent_dir / "features", parent
    ):
        raise ValueError("Restore verified parent study inputs before attribution")
    parent_report = json.loads((parent_dir / "summary/summary.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "lineage.json", evidence)
    with threadpool_limits(limits=config["threads"]):
        with log.stage("reuse_parent_features"):
            x, y, pairs = load_data(root)
            folds, stop = study_folds(
                len(x),
                json.loads((root / "configs/research.json").read_text()),
                json.loads((root / "configs/feature_study.json").read_text()),
            )
            y = y.iloc[:stop]
            panel = load_panel(parent_dir / "features")
        results, permutations = [], []
        for fold in folds:
            if fold.train_stop - 1 + 5 >= fold.validation_start:
                raise ValueError("Unreleased fitting labels")
            pending = [
                e
                for e in experiment_plan()
                if not verify_checkpoint(directory / f"fold_{fold.number}" / e.name, lineage)
            ]
            stats = None
            if pending:
                with log.stage(f"fold_{fold.number}/training_statistics"):
                    stats = prepare(panel, y, fold.train_stop, config)
            for experiment in experiment_plan():
                stage = directory / f"fold_{fold.number}" / experiment.name
                with log.stage(f"fold_{fold.number}/{experiment.name}"):
                    result = fit_stage(
                        panel, y, pairs, fold, experiment, config, stats, stage, lineage
                    )
                    checkpoint(stage)
                    results.append(result)
            del stats
            stage = directory / f"fold_{fold.number}/permutation"
            with log.stage(f"fold_{fold.number}/tree_family_permutation"):
                if not verify_checkpoint(stage, lineage):
                    parent_model = parent_dir / f"fold_{fold.number}/outer/tree_all"
                    if not verify_checkpoint(parent_model, parent):
                        raise ValueError("Missing parent tree checkpoint")
                    model = joblib.load(parent_model / "model.joblib")
                    groups = group_permutations(
                        model,
                        panel,
                        y.iloc[fold.validation_start : fold.validation_stop],
                        fold.validation_start,
                        fold.validation_stop,
                        config,
                    )
                    atomic_json(stage / "result.json", groups)
                    seal_checkpoint(stage, lineage, ["result.json"])
                checkpoint(stage)
                permutations.append(
                    {
                        "fold": fold.number,
                        "families": json.loads((stage / "result.json").read_text()),
                    }
                )
        with log.stage("joint_domain_attribution_uncertainty"):
            summaries = {
                "all_control": parent_report["summaries"]["raw__tree_all"],
                "reference_control": parent_report["summaries"]["raw__tree_reference"],
                "historical_mean": parent_report["summaries"]["historical_mean"],
            }
            lengths = [f.validation_stop - f.validation_start for f in folds]
            for experiment in experiment_plan():
                predictions = pd.concat(
                    [
                        pd.read_parquet(
                            directory / f"fold_{f.number}" / experiment.name / "predictions.parquet"
                        )
                        for f in folds
                    ]
                )
                metrics = evaluate(y.loc[predictions.index], predictions, pairs)
                records = [r for r in results if r["variant"] == experiment.name]
                summaries[experiment.name] = {
                    **metrics,
                    "fold_scores": [r["metrics"]["official_metric"] for r in records],
                    **temporal_diagnostics(
                        np.asarray(metrics["daily_rank_correlations"]),
                        lengths,
                        [r["selection"]["selected_names"] for r in records],
                    ),
                }
            own = [(name, "historical_mean") for name in summaries if name != "historical_mean"]
            own += (
                [("add_" + f, "reference_control") for f in FAMILIES]
                + [("all_control", "drop_" + f) for f in FAMILIES]
                + [("all_control", "reference_control")]
            )
            daily = {
                "domain::" + n: np.asarray(m["daily_rank_correlations"])
                for n, m in parent_report["summaries"].items()
            }
            daily.update(
                {
                    "attribution::" + n: np.asarray(m["daily_rank_correlations"])
                    for n, m in summaries.items()
                }
            )
            comparisons = [
                ("domain::" + a, "domain::" + b)
                for a, b in parent_comparisons(list(parent_report["summaries"]))
            ]
            comparisons += [("attribution::" + a, "attribution::" + b) for a, b in own]
            bounds = compare_predictions(daily, lengths, comparisons, config)
            own_bounds = [
                {
                    **r,
                    "variant": r["variant"].removeprefix("attribution::"),
                    "reference": r["reference"].removeprefix("attribution::"),
                }
                for r in bounds
                if r["variant"].startswith("attribution::")
            ]
            summary = {
                "lineage": lineage,
                "parent_lineage": parent,
                "feature_lineage": parent,
                "feature_gate": "open",
                "holdout_evaluated": False,
                "validation_dates": sum(lengths),
                "new_fitted_models": len(results),
                "controls": 0,
                "parent_controls_reused": 3,
                "residual_weight": 1.0,
                "summaries": summaries,
                "results": results,
                "comparisons": own_bounds,
                "joint_comparisons": bounds,
                "joint_comparison_count": len(comparisons),
                "group_permutation": permutations,
                "limitations": [
                    "Adaptive exploratory follow-up motivated by the parent tree result; no independent confirmation set.",
                    "Joint simultaneous bounds cover both domain phases, not all earlier research or model-refitting uncertainty.",
                    "Family tests keep the fixed tree, training-only 64-template screening, and residual weight one; nested calibration is not retuned.",
                    "Parent inputs and control models are reused without refitting; final test remains untouched.",
                ],
            }
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/tree_attribution.json", summary)
    atomic_json(root / "reports/tree_attribution_lineage.json", evidence)
    log.event(
        "tree_attribution_completed",
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
