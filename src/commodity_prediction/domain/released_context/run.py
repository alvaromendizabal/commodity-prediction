"""Nine matched fits with a cumulative budget, atomic stages and no-refit resume."""

from __future__ import annotations

import argparse
import gc
import json
import signal
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
from commodity_prediction.domain.diagnostics import temporal_diagnostics
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.released_context.features import (
    build_panel,
    declared_comparisons,
    experiment_plan,
)
from commodity_prediction.domain.risk_state.run import joint_history
from commodity_prediction.domain.risk_state.run import study_lineage as parent_lineage
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
    parent, prior = parent_lineage(root)
    config = json.loads((root / "configs/released_context_study.json").read_text())
    if config["parent_lineage"] != parent or config["feature_gate"] != "open":
        raise ValueError("Released-context parent lineage or gate changed")
    if config["max_new_fits"] != len(experiment_plan()) * 3 or config["residual_weight"] != 1:
        raise ValueError("Matched fit budget or residual strength changed")
    evidence = {
        "parent_lineage": parent,
        "feature_lineage": prior["feature_lineage"],
        "config": config,
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted(
                (root / "src/commodity_prediction/domain/released_context").glob("*.py")
            )
        },
        "experiments": [asdict(v) for v in experiment_plan()],
        "comparisons": declared_comparisons(),
    }
    return fingerprint(evidence), evidence


def run_study(
    root: Path, fold_limit: int = 3, checkpoint_hook: Callable[[Path], None] | None = None
) -> dict:
    started = monotonic()
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    config = json.loads((root / "configs/domain_study.json").read_text())
    log = RunLog(root / "logs/released_context_study.jsonl", 30)
    if fold_limit not in {1, 3}:
        raise ValueError("Only a first-fold probe or the three-fold study is declared")

    def checkpoint(stage: Path) -> None:
        if checkpoint_hook is not None:
            checkpoint_hook(stage)

    summary_stage = directory / "summary"
    if verify_checkpoint(summary_stage, lineage):
        manifests = sorted(directory.rglob("manifest.json"))
        if len(manifests) != evidence["config"]["max_new_fits"] + 1:
            raise ValueError("Incomplete released-context checkpoint inventory")
        for manifest in manifests:
            verify_checkpoint(manifest.parent, lineage)
            checkpoint(manifest.parent)
        summary = json.loads((summary_stage / "summary.json").read_text())
        atomic_json(root / "reports/released_context_study.json", summary)
        atomic_json(root / "reports/released_context_lineage.json", evidence)
        log.event("complete_run_reused", verified_checkpoints=len(manifests))
        return summary

    budget_path = directory / "runtime_budget.json"
    previous = (
        json.loads(budget_path.read_text())["elapsed_seconds"] if budget_path.exists() else 0.0
    )

    def budget() -> None:
        used = previous + monotonic() - started
        atomic_json(
            budget_path,
            {"elapsed_seconds": used, "limit_seconds": evidence["config"]["max_run_seconds"]},
        )
        if used >= evidence["config"]["max_run_seconds"]:
            raise TimeoutError("Released-context cumulative experiment budget exhausted")

    feature_dir = root / "artifacts" / evidence["feature_lineage"] / "features"
    parent_dir = root / "artifacts" / evidence["parent_lineage"] / "summary"
    for stage, expected in [
        (feature_dir, evidence["feature_lineage"]),
        (parent_dir, evidence["parent_lineage"]),
    ]:
        if not verify_checkpoint(stage, expected):
            raise ValueError("Restore verified study inputs first")
    parent = json.loads((parent_dir / "summary.json").read_text())
    atomic_json(directory / "lineage.json", evidence)
    with threadpool_limits(limits=config["threads"]):
        with log.stage("load_development_inputs"):
            x, y, pairs = load_data(root)
            folds, stop = study_folds(
                len(x),
                json.loads((root / "configs/research.json").read_text()),
                json.loads((root / "configs/feature_study.json").read_text()),
            )
            y = y.iloc[:stop]
            original = load_panel(feature_dir)
            del x
        results, summaries = [], {}
        maximum_error = 0.0
        lengths = [f.validation_stop - f.validation_start for f in folds]
        for variant in experiment_plan():
            with log.stage("features/" + variant.name):
                panel = build_panel(original, y, pairs, variant)
                settings = variant.settings(config, panel)
            predictions = []
            for fold in folds[:fold_limit]:
                budget()
                if (
                    fold.train_stop - 1 + 5 >= fold.validation_start
                    or fold.validation_stop - 1 + 5 >= stop
                ):
                    raise ValueError("Invalid information-release boundary")
                stage = directory / f"fold_{fold.number}" / variant.name
                with log.stage(f"fold_{fold.number}/{variant.name}"):
                    stats = (
                        None
                        if verify_checkpoint(stage, lineage)
                        else prepare(panel, y, fold.train_stop, settings)
                    )
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
                    if any(
                        k in result["selection"]["rejection_reasons"]
                        for k in ["feature_budget", "correlated", "unstable_sign"]
                    ):
                        raise ValueError("An admitted feature block was screened out")
                    saved = pd.read_parquet(stage / "predictions.parquet")
                    replay = joblib.load(stage / "model.joblib").predict(
                        panel, fold.validation_start, fold.validation_stop
                    )
                    pd.testing.assert_index_equal(saved.index, replay.index, check_names=False)
                    pd.testing.assert_index_equal(saved.columns, replay.columns, check_names=False)
                    error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError("Saved prediction replay differs")
                    maximum_error = max(maximum_error, error)
                    budget()
                    checkpoint(stage)
                    predictions.append(saved)
                    results.append(result)
                    log.event(
                        "checkpoint_verified",
                        completed=len(results),
                        total=3 * fold_limit,
                        variant=variant.name,
                        fold=fold.number,
                        official_metric=result["metrics"]["official_metric"],
                    )
                    del stats
            prediction = pd.concat(predictions)
            metrics = evaluate(y.loc[prediction.index], prediction, pairs)
            records = [r for r in results if r["variant"] == variant.name]
            summaries[variant.name] = {
                **metrics,
                "candidate_templates": len(panel.names),
                "fold_scores": [r["metrics"]["official_metric"] for r in records],
                **temporal_diagnostics(
                    np.asarray(metrics["daily_rank_correlations"]),
                    lengths[:fold_limit],
                    [r["selection"]["selected_names"] for r in records],
                ),
            }
            del panel
            gc.collect()
        if fold_limit == 1:
            probe = {
                "status": "first_fold_completed",
                "lineage": lineage,
                "results": results,
                "summaries": summaries,
                "maximum_prediction_replay_error": maximum_error,
                "feature_gate": "open",
                "holdout_evaluated": False,
            }
            atomic_json(root / "reports/released_context_probe.json", probe)
            budget()
            return probe
        summaries.update(
            {
                n: parent["summaries"][n]
                for n in [
                    "historical_mean",
                    "admitted_tail",
                    "screened_tail",
                    "tail_states_and_priors",
                ]
            }
        )
        with log.stage("joint_context_uncertainty"):
            daily = joint_history(root, parent)
            daily.update(
                {
                    "risk_state::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in parent["summaries"].items()
                }
            )
            daily.update(
                {
                    "released_context::" + n: np.asarray(s["daily_rank_correlations"])
                    for n, s in summaries.items()
                }
            )
            comparisons = list(
                dict.fromkeys((r["variant"], r["reference"]) for r in parent["joint_comparisons"])
            )
            comparisons += [
                ("released_context::" + a, "released_context::" + b)
                for a, b in declared_comparisons()
            ]
            bounds = compare_predictions(daily, lengths, comparisons, config)
            budget()
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
                "candidate_templates_added": 24,
                "added_template_target_assignments": 24 * len(pairs),
                "summaries": summaries,
                "results": results,
                "joint_comparisons": bounds,
                "joint_comparison_count": len(comparisons),
                "comparisons": [
                    {
                        **r,
                        "variant": r["variant"].removeprefix("released_context::"),
                        "reference": r["reference"].removeprefix("released_context::"),
                    }
                    for r in bounds
                    if r["variant"].startswith("released_context::")
                ],
                "limitations": [
                    "Exploratory after 255 previous domain comparisons; intervals condition on fitted models and do not undo adaptive research.",
                    "Own latest-value history existed in the earlier routed study; this tests short context in the matched pooled representation, not an entirely new information source.",
                    "Signed peer averages are noisy relationship proxies, not recovered contemporaneous asset returns or causal economic effects.",
                    "No model or feature policy is selected from one favorable period; the final 247 origins remain untouched.",
                ],
            }
            atomic_json(summary_stage / "summary.json", summary)
            seal_checkpoint(summary_stage, lineage, ["summary.json"])
            checkpoint(summary_stage)
    atomic_json(root / "reports/released_context_study.json", summary)
    atomic_json(root / "reports/released_context_lineage.json", evidence)
    budget()
    log.event("released_context_completed", lineage=lineage, new_fitted_models=len(results))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-fold", action="store_true")
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    prior = directory / "runtime_budget.json"
    used = json.loads(prior.read_text())["elapsed_seconds"] if prior.exists() else 0.0
    complete = verify_checkpoint(directory / "summary", lineage)
    remaining = 120 if complete else evidence["config"]["max_run_seconds"] - used
    if remaining <= 0:
        raise TimeoutError("Cumulative experiment budget exhausted; inspect preserved stages")

    def timeout_handler(signum, frame):
        if not complete:
            atomic_json(
                prior,
                {
                    "elapsed_seconds": evidence["config"]["max_run_seconds"],
                    "limit_seconds": evidence["config"]["max_run_seconds"],
                    "deadline_reached": True,
                },
            )
        raise TimeoutError("Released-context hard wall-time deadline reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        hook = None
        if args.sync_s3:
            from commodity_prediction.studies.storage import StudyStorage

            hook = StudyStorage(
                root, RunLog(root / "logs/released_context_storage.jsonl")
            ).upload_checkpoint
        run_study(root, 1 if args.first_fold else 3, hook)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
