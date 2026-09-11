"""A fixed twelve-fit short-market-path study with first-fold and no-refit gates."""

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

from commodity_prediction.data import Fold, validate_dates
from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.market_path.features import VARIANTS, build_panel
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.released_context.run import study_lineage as parent_lineage
from commodity_prediction.domain.risk_state.run import joint_history
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
from commodity_prediction.studies.storage import StudyStorage


def declared_comparisons() -> list[tuple[str, str]]:
    return [(v, c) for v in VARIANTS for c in ["admitted_tail", "historical_mean"]] + [
        ("joint_path", "current_market"),
        ("joint_path", "price_path"),
        ("joint_path", "activity_path"),
    ]


def study_lineage(root: Path) -> tuple[str, dict]:
    parent, prior = parent_lineage(root)
    config = json.loads((root / "configs/market_path_study.json").read_text())
    if (
        config["parent_lineage"] != parent
        or config["max_new_fits"] != 12
        or config["feature_gate"] != "open"
    ):
        raise ValueError("Unreviewed study configuration")
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted((root / "src/commodity_prediction/domain/market_path").glob("*.py"))
    }
    evidence = {
        "parent_lineage": parent,
        "feature_lineage": prior["feature_lineage"],
        "config": config,
        "files": files,
        "comparisons": declared_comparisons(),
        "final_evaluation_sha256": digest(root / "configs/final_evaluation.json"),
    }
    return fingerprint(evidence), evidence


def load_inputs(root: Path, feature_lineage: str):
    # Read only development rows. No final-test outcomes are loaded by this study.
    x = pd.read_csv(root / "data/raw/train.csv", nrows=1709)
    y = pd.read_csv(root / "data/raw/train_labels.csv", nrows=1709)
    pairs = pd.read_csv(root / "data/raw/target_pairs.csv")
    for frame in [x, y]:
        validate_dates(frame)
    targets = [f"target_{i}" for i in range(424)]
    if (
        len(x) != 1709
        or not x.date_id.equals(y.date_id)
        or list(y.columns) != ["date_id", *targets]
    ):
        raise ValueError("Development schema or alignment changed")
    if pairs.target.tolist() != targets or not pairs.lag.isin([1, 2, 3, 4]).all():
        raise ValueError("Target ordering or horizons changed")
    x, y = x.set_index("date_id"), y.set_index("date_id").replace(-999999, np.nan)
    original = load_panel(root / "artifacts" / feature_lineage / "features")
    folds = [Fold(0, 1164, 1169, 1349), Fold(1, 1344, 1349, 1529), Fold(2, 1524, 1529, 1704)]
    return x, y, pairs, original, folds


def run_study(root: Path, fold_limit: int = 1, sync: bool = False) -> dict:
    if fold_limit not in [1, 3]:
        raise ValueError("Only a first-fold probe or all three folds are declared")
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    directory.mkdir(parents=True, exist_ok=True)
    log = RunLog(root / "logs/market_path_study.jsonl", 15)
    storage = StudyStorage(root, log) if sync else None

    def checkpoint(stage: Path) -> None:
        if storage is not None:
            storage.upload_checkpoint(stage)

    summary_stage = directory / "summary"
    if verify_checkpoint(summary_stage, lineage):
        manifests = sorted(directory.rglob("manifest.json"))
        if len(manifests) != 13:
            raise ValueError("Completed study inventory differs")
        for path in manifests:
            verify_checkpoint(path.parent, lineage)
            checkpoint(path.parent)
        result = json.loads((summary_stage / "summary.json").read_text())
        log.event("complete_run_reused", new_fits_this_invocation=0, manifests=len(manifests))
        return result
    budget_path = directory / "runtime_budget.json"
    previous = (
        json.loads(budget_path.read_text())["elapsed_seconds"] if budget_path.exists() else 0.0
    )
    remaining = evidence["config"]["max_run_seconds"] - previous
    if remaining <= 0:
        raise TimeoutError("Cumulative study budget exhausted; do not extend automatically")
    started = monotonic()

    def budget() -> None:
        used = previous + monotonic() - started
        atomic_json(budget_path, {"elapsed_seconds": used, "limit_seconds": 300})
        if used >= 300:
            raise TimeoutError("Cumulative study budget exhausted")

    def timeout_handler(signum, frame):
        atomic_json(
            budget_path, {"elapsed_seconds": 300, "limit_seconds": 300, "deadline_reached": True}
        )
        raise TimeoutError("Market-path hard deadline reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    new_fits = 0
    try:
        feature_dir = root / "artifacts" / evidence["feature_lineage"] / "features"
        parent_stage = root / "artifacts" / evidence["parent_lineage"] / "summary"
        for stage, expected in [
            (feature_dir, evidence["feature_lineage"]),
            (parent_stage, evidence["parent_lineage"]),
        ]:
            if not verify_checkpoint(stage, expected):
                raise ValueError("Required sealed parent is missing")
        parent = json.loads((parent_stage / "summary.json").read_text())
        final = json.loads((root / "configs/final_evaluation.json").read_text())
        if final["evaluated"] or final["final_test_start_date_id"] != 1714:
            raise ValueError("Final-evaluation boundary changed")
        if fold_limit == 3:
            probe = json.loads((directory / "probe.json").read_text())
            if probe["lineage"] != lineage or not probe["continue_allowed"]:
                raise ValueError("First-fold review gate did not allow continuation")
        config = json.loads((root / "configs/domain_study.json").read_text())
        atomic_json(directory / "lineage.json", evidence)
        results, summaries, inventories = [], {}, {}
        replay_error = 0.0
        with threadpool_limits(limits=config["threads"]):
            with log.stage("reuse_verified_development_inputs"):
                x, y, pairs, original, folds = load_inputs(root, evidence["feature_lineage"])
            lengths = [f.validation_stop - f.validation_start for f in folds[:fold_limit]]
            for variant in VARIANTS:
                budget()
                with log.stage("build/" + variant):
                    panel, coverage = build_panel(original, x, pairs, variant)
                    settings = {
                        **config,
                        "max_features": len(panel.names),
                        "max_abs_correlation": 1.01,
                    }
                    inventories[variant] = coverage
                parts = []
                for fold in folds[:fold_limit]:
                    stage = directory / f"fold_{fold.number}" / variant
                    with log.stage(f"fold_{fold.number}/{variant}"):
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
                        if any(
                            k in result["selection"]["rejection_reasons"]
                            for k in ["feature_budget", "correlated", "unstable_sign"]
                        ):
                            raise ValueError("Declared admitted columns were excluded")
                        saved = pd.read_parquet(stage / "predictions.parquet")
                        replay = joblib.load(stage / "model.joblib").predict(
                            panel, fold.validation_start, fold.validation_stop
                        )
                        pd.testing.assert_frame_equal(saved, replay, check_names=False)
                        error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                        if error > 1e-12 or not np.isfinite(error):
                            raise ValueError("Prediction replay mismatch")
                        replay_error = max(replay_error, error)
                        checkpoint(stage)
                        parts.append(saved)
                        results.append(result)
                        log.event(
                            "checkpoint_verified",
                            completed=len(results),
                            total=4 * fold_limit,
                            variant=variant,
                            fold=fold.number,
                            reused=bool(complete),
                            official_metric=result["metrics"]["official_metric"],
                        )
                        del stats
                        budget()
                prediction = pd.concat(parts)
                metrics = evaluate(y.loc[prediction.index], prediction, pairs)
                summaries[variant] = {
                    **metrics,
                    "candidate_templates": len(panel.names),
                    "fold_scores": [
                        r["metrics"]["official_metric"] for r in results if r["variant"] == variant
                    ],
                }
                del panel
                gc.collect()
            if fold_limit == 1:
                reference = parent["summaries"]["admitted_tail"]["fold_scores"][0]
                proceed = not all(
                    v["official_metric"] < reference - 0.10 for v in summaries.values()
                )
                probe = {
                    "lineage": lineage,
                    "continue_allowed": proceed,
                    "results": results,
                    "summaries": summaries,
                    "frozen_control_score": reference,
                    "fits_this_invocation": new_fits,
                    "maximum_prediction_replay_error": replay_error,
                    "status": "first_fold_completed",
                    "feature_gate": "open",
                    "holdout_evaluated": False,
                }
                atomic_json(directory / "probe.json", probe)
                atomic_json(root / "reports/market_path_probe.json", probe)
                budget()
                return probe
            for name in [
                "admitted_tail",
                "historical_mean",
                "screened_tail",
                "tail_states_and_priors",
                "short_own",
            ]:
                summaries[name] = parent["summaries"][name]
            with log.stage("matched_joint_uncertainty"):
                daily = joint_history(root, parent)
                for prefix, filename in [
                    ("risk_state", "risk_state_study"),
                    ("released_context", "released_context_study"),
                ]:
                    older = json.loads((root / "reports" / (filename + ".json")).read_text())
                    sealed = root / "artifacts" / older["lineage"] / "summary"
                    if (
                        not verify_checkpoint(sealed, older["lineage"])
                        or json.loads((sealed / "summary.json").read_text()) != older
                    ):
                        raise ValueError("Comparison history differs from private seal")
                    daily.update(
                        {
                            prefix + "::" + n: np.asarray(s["daily_rank_correlations"])
                            for n, s in older["summaries"].items()
                        }
                    )
                daily.update(
                    {
                        "market_path::" + n: np.asarray(s["daily_rank_correlations"])
                        for n, s in summaries.items()
                    }
                )
                contrasts = list(
                    dict.fromkeys(
                        (c["variant"], c["reference"]) for c in parent["joint_comparisons"]
                    )
                )
                contrasts += [
                    ("market_path::" + a, "market_path::" + b) for a, b in declared_comparisons()
                ]
                bounds = compare_predictions(daily, lengths, contrasts, config)
                report = {
                    "lineage": lineage,
                    "parent_lineage": evidence["parent_lineage"],
                    "feature_lineage": evidence["feature_lineage"],
                    "feature_gate": "open",
                    "holdout_evaluated": False,
                    "validation_dates": 535,
                    "new_fitted_models": len(results),
                    "fits_this_invocation": new_fits,
                    "model_checkpoints_replayed": len(results),
                    "maximum_prediction_replay_error": replay_error,
                    "checkpoint_count": len(results) + 1,
                    "inventories": inventories,
                    "summaries": summaries,
                    "results": results,
                    "joint_comparisons": bounds,
                    "joint_comparison_count": len(contrasts),
                    "comparisons": [c for c in bounds if c["variant"].startswith("market_path::")],
                    "limitations": [
                        "Repeatedly inspected development folds; conditional intervals do not undo adaptation.",
                        "Tests short OHLC/activity representations, not a reproduction of a winning neural architecture.",
                        "Chronology and column count change jointly; a positive sequence contrast needs further capacity-matched study.",
                        "No true carry, external inventory, news, or real-calendar joins are tested.",
                    ],
                }
                atomic_json(summary_stage / "summary.json", report)
                seal_checkpoint(summary_stage, lineage, ["summary.json"])
                checkpoint(summary_stage)
                atomic_json(root / "reports/market_path_study.json", report)
                atomic_json(root / "reports/market_path_lineage.json", evidence)
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
    with (root / "logs/market_path.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_study(root, 1 if args.first_fold else 3, args.sync_s3)
    print(
        json.dumps(
            {
                "status": result.get("status", "completed"),
                "lineage": result["lineage"],
                "new_fitted_models": result.get("new_fitted_models", 4),
                "feature_gate": "open",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
