"""Six-fit matched ablation of fold-local diagonal-rank target metadata."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import signal
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.market_path.run import load_inputs
from commodity_prediction.domain.market_path.run import study_lineage as market_path_lineage
from commodity_prediction.domain.model import prepare, select
from commodity_prediction.domain.rank_prior.features import probe_methods
from commodity_prediction.domain.rank_prior_fit.features import (
    RANK_TEMPLATE,
    VARIANTS,
    build_variant_panel,
    settings,
)
from commodity_prediction.domain.risk_state.run import joint_history
from commodity_prediction.runtime import (
    RunLog,
    atomic_json,
    digest,
    fingerprint,
    seal_checkpoint,
    verify_checkpoint,
)
from commodity_prediction.studies.evaluation import compare_predictions, evaluate
from commodity_prediction.studies.storage import StudyStorage


def probe_lineage(root: Path) -> tuple[str, dict]:
    config = json.loads((root / "configs/rank_prior_probe.json").read_text())
    child = root / "src/commodity_prediction/domain/rank_prior"
    files = [
        *sorted(child.glob("*.py")),
        root / "scripts/probe_rank_priors.py",
        root / "configs/rank_prior_probe.json",
    ]
    evidence = {
        "config": config,
        "files": {str(path.relative_to(root)): digest(path) for path in files},
        "methods": list(probe_methods()),
    }
    return fingerprint(evidence), evidence


def declared_comparisons() -> list[tuple[str, str]]:
    return [
        ("rank_tail", "admitted_tail"),
        ("rank_current_market", "current_market"),
        ("rank_current_market", "rank_tail"),
    ]


def study_lineage(root: Path) -> tuple[str, dict]:
    market_id, market_evidence = market_path_lineage(root)
    probe_id, _ = probe_lineage(root)
    config = json.loads((root / "configs/rank_prior_fit_study.json").read_text())
    if (
        config["feature_gate"] != "open"
        or config["holdout_evaluated"]
        or config["parent_market_path_lineage"] != market_id
        or config["parent_rank_probe_lineage"] != probe_id
        or config["max_new_fits"] != len(VARIANTS) * 3
        or config["probe_fits"] != len(VARIANTS)
        or config["residual_weight"] != 1.0
    ):
        raise ValueError("Rank-prior fitted-study lineage or budget changed")
    evidence = {
        "parent_market_path_lineage": market_id,
        "parent_rank_probe_lineage": probe_id,
        "feature_lineage": market_evidence["feature_lineage"],
        "config": config,
        "files": {
            str(path.relative_to(root)): digest(path)
            for path in sorted((root / "src/commodity_prediction/domain/rank_prior_fit").glob("*.py"))
        },
        "variants": list(VARIANTS),
        "comparisons": declared_comparisons(),
        "final_evaluation_sha256": digest(root / "configs/final_evaluation.json"),
    }
    return fingerprint(evidence), evidence


def _comparison_daily(root: Path, parent_market: dict, new_summaries: dict) -> dict[str, np.ndarray]:
    released = json.loads((root / "reports/released_context_study.json").read_text())
    released_stage = root / "artifacts" / released["lineage"] / "summary"
    if not verify_checkpoint(released_stage, released["lineage"]):
        raise ValueError("Released-context comparison history is not sealed")
    if json.loads((released_stage / "summary.json").read_text()) != released:
        raise ValueError("Released-context public/private history differs")
    daily = joint_history(root, released)
    for prefix, filename in [
        ("risk_state", "risk_state_study"),
        ("released_context", "released_context_study"),
    ]:
        report = json.loads((root / "reports" / (filename + ".json")).read_text())
        stage = root / "artifacts" / report["lineage"] / "summary"
        if not verify_checkpoint(stage, report["lineage"]):
            raise ValueError("Missing sealed historical comparison stage")
        sealed = json.loads((stage / "summary.json").read_text())
        if sealed != report:
            raise ValueError("Historical comparison report differs from private seal")
        daily.update(
            {
                prefix + "::" + name: np.asarray(summary["daily_rank_correlations"])
                for name, summary in report["summaries"].items()
            }
        )
    daily.update(
        {
            "market_path::" + name: np.asarray(summary["daily_rank_correlations"])
            for name, summary in parent_market["summaries"].items()
        }
    )
    daily.update(
        {
            "rank_prior_fit::" + name: np.asarray(summary["daily_rank_correlations"])
            for name, summary in new_summaries.items()
        }
    )
    return daily


def run_study(root: Path, fold_limit: int = 1, sync: bool = False) -> dict:
    if fold_limit not in (1, 3):
        raise ValueError("Only a two-fit first-fold probe or all three folds are declared")
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    directory.mkdir(parents=True, exist_ok=True)
    summary_stage = directory / "summary"
    log = RunLog(root / "logs/rank_prior_fit_study.jsonl", evidence["config"]["heartbeat_seconds"])
    storage = StudyStorage(root, log) if sync else None

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    if verify_checkpoint(summary_stage, lineage):
        manifests = sorted(directory.rglob("manifest.json"))
        if len(manifests) != evidence["config"]["max_new_fits"] + 1:
            raise ValueError("Completed rank-prior fitted-study inventory differs")
        for manifest in manifests:
            verify_checkpoint(manifest.parent, lineage)
            checkpoint(manifest.parent)
        result = json.loads((summary_stage / "summary.json").read_text())
        atomic_json(root / "reports/rank_prior_fit_study.json", result)
        atomic_json(root / "reports/rank_prior_fit_lineage.json", evidence)
        log.event("complete_run_reused", verified_checkpoints=len(manifests))
        return result

    parent_stage = root / "artifacts" / evidence["parent_market_path_lineage"] / "summary"
    feature_stage = root / "artifacts" / evidence["feature_lineage"] / "features"
    for stage, expected in [
        (parent_stage, evidence["parent_market_path_lineage"]),
        (feature_stage, evidence["feature_lineage"]),
    ]:
        if not verify_checkpoint(stage, expected):
            raise ValueError("Restore verified rank-prior fitted-study inputs before fitting")
    parent_market = json.loads((parent_stage / "summary.json").read_text())
    if parent_market["holdout_evaluated"] or parent_market["validation_dates"] != 535:
        raise ValueError("Parent market-path evaluation boundary changed")
    probe_public = json.loads((root / "reports/rank_prior_probe_execution.json").read_text())
    if (
        probe_public["lineage"] != evidence["parent_rank_probe_lineage"]
        or not probe_public["decision"]["predeclared_gate_met"]
        or not probe_public["decision"]["advance_to_matched_fitted_ablation"]
    ):
        raise ValueError("Rank-prior information gate does not authorize fitted ablation")
    final = json.loads((root / "configs/final_evaluation.json").read_text())
    if final["evaluated"] or final["final_test_start_date_id"] != evidence["config"]["final_test_start_date_id"]:
        raise ValueError("Final-evaluation boundary changed")
    if fold_limit == 3:
        probe_path = directory / "probe.json"
        if not probe_path.exists():
            raise ValueError("Run and inspect the two-fit first-fold probe first")
        probe = json.loads(probe_path.read_text())
        if probe["lineage"] != lineage or not probe["continue_allowed"]:
            raise ValueError("First-fold fitted gate did not allow continuation")

    budget_path = directory / "runtime_budget.json"
    previous = json.loads(budget_path.read_text())["elapsed_seconds"] if budget_path.exists() else 0.0
    limit = evidence["config"]["max_run_seconds"]
    remaining = limit - previous
    if remaining <= 0:
        raise TimeoutError("Cumulative rank-prior fitted-study budget exhausted")
    started = monotonic()

    def budget() -> None:
        used = previous + monotonic() - started
        atomic_json(budget_path, {"elapsed_seconds": used, "limit_seconds": limit})
        if used >= limit:
            raise TimeoutError("Cumulative rank-prior fitted-study budget exhausted")

    def deadline(signum, frame):
        atomic_json(
            budget_path,
            {"elapsed_seconds": limit, "limit_seconds": limit, "deadline_reached": True},
        )
        raise TimeoutError("Rank-prior fitted-study hard deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    new_fits = 0
    results: list[dict] = []
    summaries: dict[str, dict] = {}
    inventories: dict[str, dict] = {}
    replay_error = 0.0
    try:
        atomic_json(directory / "lineage.json", evidence)
        config = json.loads((root / "configs/domain_study.json").read_text())
        with threadpool_limits(limits=config["threads"]):
            with log.stage("reuse_verified_development_inputs"):
                x, y, pairs, original, folds = load_inputs(root, evidence["feature_lineage"])
            lengths = [fold.validation_stop - fold.validation_start for fold in folds[:fold_limit]]
            for variant in VARIANTS:
                parts: list[pd.DataFrame] = []
                inventories[variant] = {}
                for fold in folds[:fold_limit]:
                    budget()
                    with log.stage(f"build/fold_{fold.number}/{variant}"):
                        panel, coverage = build_variant_panel(
                            original, x, y, pairs, fold.train_stop, variant
                        )
                        fit_config = settings(config, panel)
                        inventories[variant][f"fold_{fold.number}"] = coverage
                    stage = directory / f"fold_{fold.number}" / variant
                    with log.stage(f"fold_{fold.number}/{variant}"):
                        complete = verify_checkpoint(stage, lineage)
                        stats = None if complete else prepare(panel, y, fold.train_stop, fit_config)
                        if stats is not None:
                            selected, audit = select(stats, panel.names, fit_config, False)
                            names = [stats.names[i] for i in selected]
                            if RANK_TEMPLATE not in names:
                                raise ValueError(
                                    "Training-only diagonal-rank template was not admitted: "
                                    + json.dumps(audit["rejection_reasons"], sort_keys=True)
                                )
                            new_fits += 1
                        result = fit_stage(
                            panel,
                            y,
                            pairs,
                            fold,
                            Experiment(variant, algorithm="histogram"),
                            fit_config,
                            stats,
                            stage,
                            lineage,
                        )
                        if RANK_TEMPLATE not in result["selection"]["selected_names"]:
                            raise ValueError("Saved fitted stage does not contain the declared rank template")
                        saved = pd.read_parquet(stage / "predictions.parquet")
                        replay = joblib.load(stage / "model.joblib").predict(
                            panel, fold.validation_start, fold.validation_stop
                        )
                        pd.testing.assert_frame_equal(saved, replay, check_names=False)
                        error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                        if not np.isfinite(error) or error > 1e-12:
                            raise ValueError("Rank-prior fitted prediction replay differs")
                        replay_error = max(replay_error, error)
                        checkpoint(stage)
                        parts.append(saved)
                        results.append(result)
                        log.event(
                            "checkpoint_verified",
                            completed=len(results),
                            total=len(VARIANTS) * fold_limit,
                            variant=variant,
                            fold=fold.number,
                            reused=bool(complete),
                            official_metric=result["metrics"]["official_metric"],
                        )
                    del panel, stats
                    budget()
                prediction = pd.concat(parts)
                metrics = evaluate(y.loc[prediction.index], prediction, pairs)
                summaries[variant] = {
                    **metrics,
                    "fold_scores": [
                        result["metrics"]["official_metric"]
                        for result in results
                        if result["variant"] == variant
                    ],
                }
                gc.collect()

        if fold_limit == 1:
            control_map = evidence["config"]["matched_controls"]
            deltas = {
                variant: summaries[variant]["official_metric"]
                - parent_market["summaries"][control_map[variant]]["fold_scores"][0]
                for variant in VARIANTS
            }
            margin = evidence["config"]["first_fold_stop_margin"]
            proceed = any(delta >= -margin for delta in deltas.values())
            probe = {
                "status": "first_fold_completed",
                "lineage": lineage,
                "feature_gate": "open",
                "holdout_evaluated": False,
                "fits_this_invocation": new_fits,
                "summaries": summaries,
                "matched_control_deltas": deltas,
                "continue_allowed": proceed,
                "maximum_prediction_replay_error": replay_error,
            }
            atomic_json(directory / "probe.json", probe)
            atomic_json(root / "reports/rank_prior_fit_probe.json", probe)
            budget()
            return probe

        parent_controls = {
            name: parent_market["summaries"][name]
            for name in ["admitted_tail", "current_market", "historical_mean"]
        }
        combined = {**summaries, **parent_controls}
        with log.stage("cumulative_joint_uncertainty"):
            daily = _comparison_daily(root, parent_market, summaries)
            comparisons = list(
                dict.fromkeys(
                    (row["variant"], row["reference"])
                    for row in parent_market["joint_comparisons"]
                )
            )
            comparisons += [
                ("rank_prior_fit::rank_tail", "market_path::admitted_tail"),
                ("rank_prior_fit::rank_current_market", "market_path::current_market"),
                ("rank_prior_fit::rank_current_market", "rank_prior_fit::rank_tail"),
            ]
            bounds = compare_predictions(daily, lengths, comparisons, config)
            new_bounds = [row for row in bounds if row["variant"].startswith("rank_prior_fit::")]
        report = {
            "status": "completed",
            "lineage": lineage,
            "parent_market_path_lineage": evidence["parent_market_path_lineage"],
            "parent_rank_probe_lineage": evidence["parent_rank_probe_lineage"],
            "feature_gate": "open",
            "holdout_evaluated": False,
            "validation_dates": sum(lengths),
            "new_fitted_models": len(results),
            "fits_this_invocation": new_fits,
            "model_checkpoints_replayed": len(results),
            "maximum_prediction_replay_error": replay_error,
            "checkpoint_count": len(results) + 1,
            "rank_templates_added": 1,
            "summaries": combined,
            "results": results,
            "inventories": inventories,
            "joint_comparisons": bounds,
            "joint_comparison_count": len(comparisons),
            "comparisons": new_bounds,
            "decision_rule": evidence["config"]["decision_rule"],
            "limitations": [
                "Development folds are repeatedly inspected; uncertainty conditions on fitted models and does not undo adaptation.",
                "The diagonal rank template is target metadata estimated from each fold training prefix, not contemporaneously released validation labels.",
                "This study tests incremental value under the frozen histogram model, not a ranking-loss or shared-neural architecture.",
                "The final 247 origins remain untouched.",
            ],
        }
        atomic_json(summary_stage / "summary.json", report)
        seal_checkpoint(summary_stage, lineage, ["summary.json"])
        checkpoint(summary_stage)
        atomic_json(root / "reports/rank_prior_fit_study.json", report)
        atomic_json(root / "reports/rank_prior_fit_lineage.json", evidence)
        budget()
        return report
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        budget()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-fold", action="store_true")
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/rank_prior_fit.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_study(root, 1 if args.first_fold else 3, args.sync_s3)
    print(
        json.dumps(
            {
                "status": result["status"],
                "lineage": result["lineage"],
                "new_fitted_models": result.get("new_fitted_models", len(VARIANTS)),
                "fits_this_invocation": result["fits_this_invocation"],
                "feature_gate": "open",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
