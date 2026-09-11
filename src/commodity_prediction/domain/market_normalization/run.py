"""Nine-fit, matched current-market normalization study with a first-fold gate."""

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
from commodity_prediction.domain.market_normalization.features import VARIANTS, augment
from commodity_prediction.domain.market_path.features import build_panel
from commodity_prediction.domain.market_path.run import load_inputs
from commodity_prediction.domain.market_path.run import study_lineage as parent_lineage
from commodity_prediction.domain.model import prepare
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


def declarations(root: Path) -> tuple[str, dict]:
    config = json.loads((root / "configs/market_normalization.json").read_text())
    if (
        config["max_new_fits"] != 9
        or config["max_run_seconds"] != 300
        or config["variants"] != list(VARIANTS)
        or config["first_fold_minimum_gain"] != 0.002
    ):
        raise ValueError("Unreviewed feature experiment declaration")
    parent, parent_evidence = parent_lineage(root)
    if config["parent_lineage"] != parent:
        raise ValueError("Frozen current-market parent changed")
    files = {
        str(path.relative_to(root)): digest(path)
        for path in sorted((root / "src/commodity_prediction/domain/market_normalization").glob("*.py"))
    }
    evidence = {
        "config": config,
        "parent_lineage": parent,
        "parent_evidence": parent_evidence,
        "files": files,
        "final_boundary_sha256": digest(root / "configs/final_evaluation.json"),
    }
    return fingerprint(evidence), evidence


def preflight(root: Path, evidence: dict) -> dict:
    """All declarations, private dependencies and sealed controls precede fitting."""
    required = ["data/raw/train.csv", "data/raw/train_labels.csv", "data/raw/target_pairs.csv"]
    for name in required:
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise ValueError(f"Missing private input: {name}")
    final = json.loads((root / "configs/final_evaluation.json").read_text())
    if final["evaluated"] or final["final_test_start_date_id"] != 1714:
        raise ValueError("Final evaluation gate changed")
    parent = evidence["parent_lineage"]
    feature = evidence["parent_evidence"]["feature_lineage"]
    for stage, lineage in [
        (root / "artifacts" / parent / "summary", parent),
        (root / "artifacts" / feature / "features", feature),
    ]:
        if not verify_checkpoint(stage, lineage):
            raise ValueError(f"Missing sealed parent: {stage}")
    report = json.loads((root / "artifacts" / parent / "summary/summary.json").read_text())
    if report["validation_dates"] != 535 or report["holdout_evaluated"]:
        raise ValueError("Parent evaluation dates changed")
    for fold in range(3):
        stage = root / "artifacts" / parent / f"fold_{fold}/current_market"
        if not verify_checkpoint(stage, parent):
            raise ValueError(f"Missing frozen control checkpoint: fold {fold}")
    return report


def run_study(root: Path, first_fold: bool, sync: bool = False) -> dict:
    lineage, evidence = declarations(root)
    config = evidence["config"]
    parent = preflight(root, evidence)
    directory = root / "artifacts" / lineage
    directory.mkdir(parents=True, exist_ok=True)
    summary_dir = directory / "summary"
    log = RunLog(root / "logs/market_normalization.jsonl", 15)
    storage = StudyStorage(root, log) if sync else None

    def persist(stage: Path) -> None:
        if storage:
            storage.upload_checkpoint(stage)

    if verify_checkpoint(summary_dir, lineage):
        report = json.loads((summary_dir / "summary.json").read_text())
        for path in sorted(directory.rglob("manifest.json")):
            if not verify_checkpoint(path.parent, lineage):
                raise ValueError("Incomplete resumed study")
            persist(path.parent)
        log.event("completed_run_reused", new_fits=0)
        return report
    budget_path = directory / "runtime_budget.json"
    previous = json.loads(budget_path.read_text())["seconds"] if budget_path.exists() else 0.0
    if previous >= config["max_run_seconds"]:
        raise TimeoutError("Cumulative budget exhausted; no automatic extension")
    started = monotonic()

    def budget() -> None:
        elapsed = previous + monotonic() - started
        atomic_json(budget_path, {"seconds": elapsed, "limit": config["max_run_seconds"]})
        if elapsed >= config["max_run_seconds"]:
            raise TimeoutError("Cumulative feature study deadline")

    def deadline(signum, frame):
        raise TimeoutError("Feature study hard deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, config["max_run_seconds"] - previous)
    atomic_json(directory / "lineage.json", evidence)
    new_fits = 0
    try:
        if not first_fold:
            probe = json.loads((directory / "probe.json").read_text())
            if probe["lineage"] != lineage or not probe["continue_allowed"]:
                raise ValueError("First-fold review did not authorize further fits")
        domain_config = json.loads((root / "configs/domain_study.json").read_text())
        results, summaries, inventories = [], {}, {}
        replay_error = 0.0
        with threadpool_limits(limits=domain_config["threads"]):
            with log.stage("reuse_sealed_inputs"):
                x, y, pairs, original, folds = load_inputs(root, evidence["parent_evidence"]["feature_lineage"])
                base, _ = build_panel(original, x, pairs, "current_market")
            used_folds = folds[:1] if first_fold else folds
            for fold in used_folds:
                budget()
                # Replay each saved control under exactly the input panel reused here.
                control_stage = root / "artifacts" / evidence["parent_lineage"] / f"fold_{fold.number}/current_market"
                saved = pd.read_parquet(control_stage / "predictions.parquet")
                replay = joblib.load(control_stage / "model.joblib").predict(base, fold.validation_start, fold.validation_stop)
                pd.testing.assert_frame_equal(saved, replay, check_names=False)
            for variant in VARIANTS:
                with log.stage("build/" + variant):
                    panel, inventory = augment(base, x, pairs, variant)
                    inventories[variant] = inventory
                settings = {**domain_config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
                predictions = []
                for fold in used_folds:
                    budget()
                    stage = directory / f"fold_{fold.number}" / variant
                    with log.stage(f"fold_{fold.number}/{variant}"):
                        reused = verify_checkpoint(stage, lineage)
                        stats = None if reused else prepare(panel, y, fold.train_stop, settings)
                        if not reused:
                            new_fits += 1
                        result = fit_stage(panel, y, pairs, fold, Experiment(variant, algorithm="histogram"), settings, stats, stage, lineage)
                        selected = result["selection"]["selected_names"]
                        added = [name for name in selected if name.startswith("market_normalization__")]
                        if not added:
                            raise ValueError("No usable new feature admitted; stop before more fits")
                        if any(reason in result["selection"]["rejection_reasons"] for reason in ["feature_budget", "correlated", "unstable_sign"]):
                            raise ValueError("Unreviewed feature screening exclusion")
                        saved = pd.read_parquet(stage / "predictions.parquet")
                        replay = joblib.load(stage / "model.joblib").predict(panel, fold.validation_start, fold.validation_stop)
                        pd.testing.assert_frame_equal(saved, replay, check_names=False)
                        error = float(np.max(np.abs(saved.to_numpy() - replay.to_numpy())))
                        if not np.isfinite(error) or error > 1e-12:
                            raise ValueError("Model replay mismatch")
                        replay_error = max(replay_error, error)
                        persist(stage)
                        results.append({**result, "added_retained_templates": len(added)})
                        predictions.append(saved)
                        log.event("checkpoint_verified", completed=len(results), total=3 * len(used_folds), reused=reused, new_fits=new_fits, official_metric=result["metrics"]["official_metric"])
                        del stats
                prediction = pd.concat(predictions)
                summaries[variant] = {**evaluate(y.loc[prediction.index], prediction, pairs), "fold_scores": [row["metrics"]["official_metric"] for row in results if row["variant"] == variant]}
                del panel
                gc.collect()
            report = {
                "lineage": lineage,
                "parent_lineage": evidence["parent_lineage"],
                "feature_gate": "open",
                "holdout_evaluated": False,
                "fits_this_invocation": new_fits,
                "model_checkpoints": len(results),
                "maximum_prediction_replay_error": replay_error,
                "results": results,
                "summaries": summaries,
                "inventories": inventories,
                "validation_dates": sum(f.validation_stop - f.validation_start for f in used_folds),
            }
            if first_fold:
                control = parent["summaries"]["current_market"]["fold_scores"][0]
                deltas = {name: row["official_metric"] - control for name, row in summaries.items()}
                proceed = max(deltas.values()) >= config["first_fold_minimum_gain"]
                report.update({"status": "first_fold_completed", "matched_deltas": deltas, "continue_allowed": proceed, "minimum_gain": config["first_fold_minimum_gain"]})
                atomic_json(directory / "probe.json", report)
                atomic_json(root / "reports/market_normalization_probe.json", report)
            else:
                summaries["current_market"] = parent["summaries"]["current_market"]
                summaries["historical_mean"] = parent["summaries"]["historical_mean"]
                daily = {name: np.asarray(row["daily_rank_correlations"]) for name, row in summaries.items()}
                contrasts = [(name, "current_market") for name in VARIANTS] + [("normalized_joint", "normalized_price"), ("normalized_joint", "volume_confirmation")]
                with log.stage("matched_family_uncertainty"):
                    bounds = compare_predictions(daily, [f.validation_stop - f.validation_start for f in folds], contrasts, domain_config)
                report.update({"status": "completed", "comparisons": bounds, "uncertainty_scope": "Five predeclared contrasts in this feature family only; not multiplicity correction for prior adaptive research.", "promotion_allowed": False})
                atomic_json(summary_dir / "summary.json", report)
                seal_checkpoint(summary_dir, lineage, ["summary.json"])
                persist(summary_dir)
                atomic_json(root / "reports/market_normalization_study.json", report)
            budget()
            report["cumulative_seconds"] = json.loads(budget_path.read_text())["seconds"]
            return report
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        atomic_json(budget_path, {"seconds": previous + monotonic() - started, "limit": config["max_run_seconds"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-fold", action="store_true")
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/market_normalization.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_study(root, args.first_fold, args.sync_s3)
    print(json.dumps({key: result[key] for key in ["status", "lineage", "fits_this_invocation", "validation_dates"]}), flush=True)


if __name__ == "__main__":
    main()
