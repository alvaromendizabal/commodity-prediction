"""Bounded no-fit probe for metric-aligned target-rank priors."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_prediction.data import load_data
from commodity_prediction.domain.rank_prior.features import constant_prediction, probe_methods
from commodity_prediction.runtime import RunLog, atomic_json, digest, fingerprint
from commodity_prediction.studies.evaluation import compare_predictions, evaluate
from commodity_prediction.studies.run import study_folds


def probe_lineage(source_root: Path) -> tuple[str, dict]:
    config = json.loads((source_root / "configs/rank_prior_probe.json").read_text())
    child = source_root / "src/commodity_prediction/domain/rank_prior"
    files = [
        *sorted(child.glob("*.py")),
        source_root / "scripts/probe_rank_priors.py",
        source_root / "configs/rank_prior_probe.json",
    ]
    evidence = {
        "config": config,
        "files": {str(path.relative_to(source_root)): digest(path) for path in files},
        "methods": list(probe_methods()),
    }
    return fingerprint(evidence), evidence


def run_probe(source_root: Path, private_root: Path) -> dict:
    started = time.monotonic()
    config = json.loads((source_root / "configs/rank_prior_probe.json").read_text())
    if config["feature_gate"] != "open" or config["holdout_evaluated"]:
        raise ValueError("Rank-prior probe must stay inside development feature research")
    lineage, evidence = probe_lineage(source_root)
    output = source_root / "reports/rank_prior_probe.json"
    log = RunLog(source_root / "logs/rank_prior_probe.jsonl", config["heartbeat_seconds"])
    with log.stage("load_development_contract"):
        _, y, pairs = load_data(private_root)
        folds, stop = study_folds(
            len(y),
            json.loads((source_root / "configs/research.json").read_text()),
            json.loads((source_root / "configs/feature_study.json").read_text()),
        )
        y = y.iloc[:stop]
        if folds[-1].validation_stop != 1704 or stop != 1709:
            raise ValueError("Development/final boundary changed")
    methods = probe_methods()
    fold_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in methods}
    fold_metrics: dict[str, list[float]] = {name: [] for name in methods}
    for fold in folds:
        with log.stage(f"fold_{fold.number}_training_only_rank_priors"):
            train = y.iloc[: fold.train_stop]
            valid = y.iloc[fold.validation_start : fold.validation_stop]
            for name, method in methods.items():
                scores = method(train)
                prediction = constant_prediction(valid.index, valid.columns, scores)
                metrics = evaluate(valid, prediction, pairs)
                fold_predictions[name].append(prediction)
                fold_metrics[name].append(metrics["official_metric"])
                log.event(
                    "probe_score",
                    method=name,
                    fold=fold.number,
                    official_metric=metrics["official_metric"],
                )
    summaries = {}
    daily = {}
    for name in methods:
        prediction = pd.concat(fold_predictions[name])
        truth = y.loc[prediction.index]
        metrics = evaluate(truth, prediction, pairs)
        summaries[name] = {**metrics, "fold_scores": fold_metrics[name]}
        daily[name] = np.asarray(metrics["daily_rank_correlations"])
    comparisons = [(name, "raw_mean") for name in methods if name != "raw_mean"]
    uncertainty = compare_predictions(
        daily,
        [fold.validation_stop - fold.validation_start for fold in folds],
        comparisons,
        json.loads((source_root / "configs/domain_study.json").read_text()),
    )
    best = max(summaries, key=lambda name: summaries[name]["official_metric"])
    elapsed = time.monotonic() - started
    if elapsed > config["max_run_seconds"]:
        raise TimeoutError("Rank-prior probe exceeded its declared wall-time budget")
    result = {
        "status": "completed",
        "lineage": lineage,
        "evidence": evidence,
        "feature_gate": "open",
        "holdout_evaluated": False,
        "validation_dates": sum(f.validation_stop - f.validation_start for f in folds),
        "new_training_fits": 0,
        "model_loads": 0,
        "notebook_executions": 0,
        "elapsed_seconds": elapsed,
        "summaries": summaries,
        "comparisons": uncertainty,
        "best_probe": best,
        "decision_rule": config["decision_rule"],
        "limitations": [
            "This is a no-fit structural-prior probe, not a promoted forecasting model.",
            "All scores reuse the repeatedly inspected development folds; final origins remain untouched.",
            "Ledoit-Wolf covariance uses training dates only and is evaluated as a fixed cross-target ordering.",
        ],
    }
    atomic_json(output, result)
    atomic_json(source_root / "reports/rank_prior_probe_lineage.json", evidence)
    log.event(
        "rank_prior_probe_completed",
        best_probe=best,
        best_score=summaries[best]["official_metric"],
        elapsed_seconds=elapsed,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    config = json.loads((source_root / "configs/rank_prior_probe.json").read_text())

    def deadline(signum, frame):
        raise TimeoutError("Rank-prior probe hard deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config["max_run_seconds"])
    try:
        result = run_probe(source_root, args.private_root.resolve())
    finally:
        signal.alarm(0)
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_probe": result["best_probe"],
                "scores": {
                    key: value["official_metric"] for key, value in result["summaries"].items()
                },
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
