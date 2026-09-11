"""Six matched fitted rank-prior ablations with a two-fit first-fold gate."""

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
from commodity_prediction.domain.compact.features import BASE
from commodity_prediction.domain.compact.features import Variant as CompactVariant
from commodity_prediction.domain.compact.features import build_panel as compact_panel
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.market_path.features import build_panel as market_panel
from commodity_prediction.domain.market_path.run import load_inputs
from commodity_prediction.domain.market_path.run import study_lineage as market_path_lineage
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.rank_prior_fit.features import RANK_TEMPLATE, append_diagonal_rank
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

VARIANTS = {
    "admitted_tail_rank": "admitted_tail",
    "current_market_rank": "current_market",
}


def declared_comparisons() -> list[tuple[str, str]]:
    return [
        ("admitted_tail_rank", "admitted_tail"),
        ("current_market_rank", "current_market"),
        ("current_market_rank", "admitted_tail_rank"),
        ("admitted_tail_rank", "historical_mean"),
        ("current_market_rank", "historical_mean"),
    ]


def study_lineage(root: Path) -> tuple[str, dict]:
    market_lineage, market_evidence = market_path_lineage(root)
    config = json.loads((root / "configs/rank_prior_fit_study.json").read_text())
    probe = json.loads((root / "reports/rank_prior_probe_execution.json").read_text())
    if (
        config["parent_market_path_lineage"] != market_lineage
        or config["probe_lineage"] != probe["lineage"]
        or not probe["decision"]["predeclared_gate_met"]
        or config["feature_gate"] != "open"
        or config["max_new_fits"] != 6
        or config["probe_fits"] != 2
    ):
        raise ValueError("Rank-prior fitted-study declaration or parent evidence changed")
    files = {
        str(path.relative_to(root)): digest(path)
        for path in sorted((root / "src/commodity_prediction/domain/rank_prior_fit").glob("*.py"))
    }
    evidence = {
        "parent_market_path_lineage": market_lineage,
        "parent_market_path_evidence": market_evidence,
        "probe_lineage": probe["lineage"],
        "probe_execution_sha256": digest(root / "reports/rank_prior_probe_execution.json"),
        "config": config,
        "files": files,
        "variants": VARIANTS,
        "comparisons": declared_comparisons(),
        "final_evaluation_sha256": digest(root / "configs/final_evaluation.json"),
    }
    return fingerprint(evidence), evidence


def build_base_panel(original, x: pd.DataFrame, pairs: pd.DataFrame, variant: str):
    if variant == "admitted_tail_rank":
        return compact_panel(
            original,
            pairs,
            CompactVariant(variant, (*BASE, "tail_risk"), admit=True),
        )
    if variant == "current_market_rank":
        panel, _ = market_panel(original, x, pairs, "current_market")
        return panel
    raise ValueError("Undeclared rank-prior fitted variant")


def run_study(root: Path, fold_limit: int = 1, sync: bool = False) -> dict:
    if fold_limit not in {1, 3}:
        raise ValueError("Only a first-fold probe or the full three-fold study is declared")
    lineage, evidence = study_lineage(root)
    config = evidence["config"]
    directory = root / "artifacts" / lineage
    directory.mkdir(parents=True, exist_ok=True)
    log = RunLog(root / "logs/rank_prior_fit_study.jsonl", 15)
    storage = StudyStorage(root, log) if sync else None

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    summary_stage = directory / "summary"
    if verify_checkpoint(summary_stage, lineage):
        manifests = sorted(directory.rglob("manifest.json"))
        if len(manifests) != config["max_new_fits"] + 1:
            raise ValueError("Completed rank-prior fitted checkpoint inventory differs")
        for manifest in manifests:
            verify_checkpoint(manifest.parent, lineage)
            checkpoint(manifest.parent)
        summary = json.loads((summary_stage / "summary.json").read_text())
        atomic_json(root / "reports/rank_prior_fit_study.json", summary)
        atomic_json(root / "reports/rank_prior_fit_lineage.json", evidence)
        log.event("complete_run_reused", verified_checkpoints=len(manifests), new_fits=0)
        return summary

    budget_path = directory / "runtime_budget.json"
    previous = (
        json.loads(budget_path.read_text())["elapsed_seconds"] if budget_path.exists() else 0.0
    )
    remaining = config["max_run_seconds"] - previous
    if remaining <= 0:
        raise TimeoutError("Cumulative rank-prior fitted-study budget exhausted")
    started = monotonic()

    def budget() -> None:
        used = previous + monotonic() - started
        atomic_json(
            budget_path,
            {"elapsed_seconds": used, "limit_seconds": config["max_run_seconds"]},
        )
        if used >= config["max_run_seconds"]:
            raise TimeoutError("Cumulative rank-prior fitted-study budget exhausted")

    def deadline(signum, frame):
        atomic_json(
            budget_path,
            {
                "elapsed_seconds": config["max_run_seconds"],
                "limit_seconds": config["max_run_seconds"],
                "deadline_reached": True,
            },
        )
        raise TimeoutError("Rank-prior fitted-study hard deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    new_fits = 0
    replay_error = 0.0
    try:
        market_lineage = evidence["parent_market_path_lineage"]
        market_stage = root / "artifacts" / market_lineage / "summary"
        if not verify_checkpoint(market_stage, market_lineage):
            raise ValueError("Restore the sealed market-path parent before fitting rank priors")
        market_report = json.loads((market_stage / "summary.json").read_text())
        public_market = json.loads((root / "reports/market_path_study.json").read_text())
        if public_market != market_report:
            raise ValueError("Public market-path evidence differs from the sealed parent")
        market_evidence = evidence["parent_market_path_evidence"]
        feature_lineage = market_evidence["feature_lineage"]
        feature_stage = root / "artifacts" / feature_lineage / "features"
        if not verify_checkpoint(feature_stage, feature_lineage):
            raise ValueError("Missing sealed domain feature panel")
        final = json.loads((root / "configs/final_evaluation.json").read_text())
        if final["evaluated"] or final["final_test_start_date_id"] != config["final_test_start"]:
            raise ValueError("Final-evaluation boundary changed")
        if fold_limit == 3:
            probe_path = directory / "probe.json"
            if not probe_path.is_file():
                raise ValueError("Run and inspect the first-fold gate before the full study")
            probe = json.loads(probe_path.read_text())
            if probe["lineage"] != lineage or not probe["continue_allowed"]:
                raise ValueError("First-fold gate does not allow continuation")

        domain_config = json.loads((root / "configs/domain_study.json").read_text())
        atomic_json(directory / "lineage.json", evidence)
        results: list[dict] = []
        summaries: dict[str, dict] = {}
        with threadpool_limits(limits=domain_config["threads"]):
            with log.stage("reuse_verified_development_inputs"):
                x, y, pairs, original, folds = load_inputs(root, feature_lineage)
            lengths = [fold.validation_stop - fold.validation_start for fold in folds[:fold_limit]]
            for variant, control in VARIANTS.items():
                base = build_base_panel(original, x, pairs, variant)
                predictions = []
                for fold in folds[:fold_limit]:
                    budget()
                    with log.stage(f"fold_{fold.number}/{variant}"):
                        panel = append_diagonal_rank(base, y.iloc[: fold.train_stop])
                        settings = {
                            **domain_config,
                            "max_features": len(panel.names),
                            "max_abs_correlation": 1.01,
                        }
                        stage = directory / f"fold_{fold.number}" / variant
                        complete = verify_checkpoint(stage, lineage)
                        stats = None if complete else prepare(panel, y, fold.train_stop, settings)
                        if not complete:
                            new_fits += 1
                        result = fit_stage(
                            panel,
                            y,
                            pairs,
                            fold,
                            Experiment(variant, algorithm="histogram"),
                            settings,
                            stats,
                            stage,
                            lineage,
                        )
                        if RANK_TEMPLATE not in result["selection"]["selected_names"]:
                            raise ValueError("The declared diagonal-rank feature was not admitted")
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
                        predictions.append(saved)
                        results.append(result)
                        log.event(
                            "checkpoint_verified",
                            variant=variant,
                            matched_control=control,
                            fold=fold.number,
                            reused=bool(complete),
                            completed=len(results),
                            total=len(VARIANTS) * fold_limit,
                            official_metric=result["metrics"]["official_metric"],
                        )
                        del stats, panel
                        budget()
                prediction = pd.concat(predictions)
                metrics = evaluate(y.loc[prediction.index], prediction, pairs)
                summaries[variant] = {
                    **metrics,
                    "candidate_templates": result["selection"]["candidate_templates"],
                    "fold_scores": [
                        row["metrics"]["official_metric"]
                        for row in results
                        if row["variant"] == variant
                    ],
                    "matched_control": control,
                }
                del base
                gc.collect()

            if fold_limit == 1:
                deltas = {
                    variant: summaries[variant]["official_metric"]
                    - market_report["summaries"][control]["fold_scores"][0]
                    for variant, control in VARIANTS.items()
                }
                threshold = config["first_fold_stop_if_both_below_control_by"]
                proceed = not all(delta < threshold for delta in deltas.values())
                probe = {
                    "status": "first_fold_completed",
                    "lineage": lineage,
                    "feature_gate": "open",
                    "holdout_evaluated": False,
                    "fits_this_invocation": new_fits,
                    "maximum_prediction_replay_error": replay_error,
                    "summaries": summaries,
                    "results": results,
                    "matched_control_deltas": deltas,
                    "stop_threshold": threshold,
                    "continue_allowed": proceed,
                }
                atomic_json(directory / "probe.json", probe)
                atomic_json(root / "reports/rank_prior_fit_probe.json", probe)
                budget()
                return probe

            for name in [
                "admitted_tail",
                "current_market",
                "historical_mean",
                "screened_tail",
                "tail_states_and_priors",
                "short_own",
            ]:
                summaries[name] = market_report["summaries"][name]

            with log.stage("matched_joint_uncertainty"):
                released_parent_lineage = market_evidence["parent_lineage"]
                released_parent_stage = root / "artifacts" / released_parent_lineage / "summary"
                if not verify_checkpoint(released_parent_stage, released_parent_lineage):
                    raise ValueError("Missing sealed released-context comparison parent")
                released_parent = json.loads((released_parent_stage / "summary.json").read_text())
                daily = joint_history(root, released_parent)
                for prefix, filename in [
                    ("risk_state", "risk_state_study"),
                    ("released_context", "released_context_study"),
                ]:
                    older = json.loads((root / "reports" / f"{filename}.json").read_text())
                    sealed = root / "artifacts" / older["lineage"] / "summary"
                    if (
                        not verify_checkpoint(sealed, older["lineage"])
                        or json.loads((sealed / "summary.json").read_text()) != older
                    ):
                        raise ValueError("Historical comparison evidence differs from its private seal")
                    daily.update(
                        {
                            f"{prefix}::{name}": np.asarray(summary["daily_rank_correlations"])
                            for name, summary in older["summaries"].items()
                        }
                    )
                daily.update(
                    {
                        f"market_path::{name}": np.asarray(summary["daily_rank_correlations"])
                        for name, summary in market_report["summaries"].items()
                    }
                )
                daily.update(
                    {
                        f"rank_prior_fit::{name}": np.asarray(summary["daily_rank_correlations"])
                        for name, summary in summaries.items()
                    }
                )
                old = list(
                    dict.fromkeys(
                        (row["variant"], row["reference"])
                        for row in market_report["joint_comparisons"]
                    )
                )
                new = [
                    (f"rank_prior_fit::{a}", f"rank_prior_fit::{b}")
                    for a, b in declared_comparisons()
                ]
                comparisons = old + new
                bounds = compare_predictions(daily, lengths, comparisons, domain_config)
                report = {
                    "status": "completed",
                    "lineage": lineage,
                    "parent_market_path_lineage": market_lineage,
                    "probe_lineage": evidence["probe_lineage"],
                    "feature_gate": "open",
                    "holdout_evaluated": False,
                    "validation_dates": sum(lengths),
                    "new_fitted_models": len(results),
                    "fits_this_invocation": new_fits,
                    "model_checkpoints_replayed": len(results),
                    "maximum_prediction_replay_error": replay_error,
                    "checkpoint_count": len(results) + 1,
                    "summaries": summaries,
                    "results": results,
                    "joint_comparisons": bounds,
                    "joint_comparison_count": len(comparisons),
                    "comparisons": [
                        row for row in bounds if row["variant"].startswith("rank_prior_fit::")
                    ],
                    "limitations": [
                        "The diagonal-rank family advanced from a predeclared no-fit gate, but these are repeatedly inspected development folds.",
                        "The rank score is estimated separately from each outer fold's training prefix and is static within that fold; no validation label enters the feature.",
                        "Conditional and simultaneous intervals condition on fitted models and do not undo adaptive feature research.",
                        "No covariance tuning, final-test evaluation, leaderboard submission, external data, or neural architecture is part of this study.",
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
                "fits_this_invocation": result.get("fits_this_invocation", 0),
                "feature_gate": "open",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
