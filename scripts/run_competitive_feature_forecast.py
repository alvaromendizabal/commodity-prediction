#!/usr/bin/env python3
"""Run a bounded competitive-mechanism gate against the canonical current_market baseline."""

# Ruff import sorting differs across the standalone execution context; imports are grouped intentionally.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_prediction.competitive.feature_forecast import (
    blend,
    mean_rank_prior,
    recursive_predict,
    regularized_kelly_rank_prior,
    released_label_signal,
    score_prediction,
    sha256,
    train_feature_forecaster,
    verify_manifest,
    write_manifest,
)
from commodity_prediction.data import load_data, make_folds
from commodity_prediction.metrics import paired_block_interval


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/competitive_feature_forecast.json"
LOG_PATH = ROOT / "logs/competitive_feature_forecast.jsonl"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def event(kind: str, payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": now_utc(), "event": kind, **payload}
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def git(command: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def config_for_mode(config: dict, mode: str) -> dict:
    out = dict(config)
    if mode == "smoke":
        out.update({"hidden_dim": 64, "max_epochs": 3, "early_stopping_patience": 2, "forecast_spaces": ["log"]})
    return out


def run_id(config: dict, mode: str) -> str:
    source = (ROOT / "src/commodity_prediction/competitive/feature_forecast.py").read_bytes()
    payload = json.dumps({"config": config, "mode": mode}, sort_keys=True).encode() + source
    return hashlib.sha256(payload).hexdigest()[:16]


def baseline_local_path(lineage: str, fold: int) -> Path:
    return ROOT / "artifacts" / "competitive_feature_forecast" / "baseline_cache" / lineage / f"fold_{fold}" / "predictions.parquet"


def ensure_baseline(config: dict, fold: int) -> Path:
    bucket = config["baseline_s3_bucket"]
    lineage = config["baseline_s3_lineage"]
    path = baseline_local_path(lineage, fold)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"checkpoints/artifacts/{lineage}/fold_{fold}/current_market/predictions.parquet"
    try:
        import boto3

        boto3.client("s3", region_name="us-west-2").download_file(bucket, key, str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve canonical baseline s3://{bucket}/{key}: {exc}") from exc
    return path


def load_baseline(config: dict, fold: int, truth: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_parquet(ensure_baseline(config, fold))
    if "date_id" in prediction.columns:
        prediction = prediction.set_index("date_id")
    prediction.index = prediction.index.astype(int)
    prediction = prediction.loc[truth.index, truth.columns]
    if not prediction.index.equals(truth.index) or not prediction.columns.equals(truth.columns):
        raise ValueError("Canonical baseline schema does not match truth")
    return prediction


def prediction_stage(directory: Path, fold: int, name: str) -> Path:
    return directory / f"fold_{fold}" / name


def save_prediction(stage: Path, prediction: pd.DataFrame, result: dict, payload: dict) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(stage / "predictions.parquet")
    (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_manifest(stage, ["predictions.parquet", "result.json"], payload)


def load_or_make_prediction(stage: Path, payload: dict, maker) -> tuple[pd.DataFrame, dict, bool]:
    if verify_manifest(stage, payload):
        return pd.read_parquet(stage / "predictions.parquet"), json.loads((stage / "result.json").read_text()), True
    prediction, result = maker()
    save_prediction(stage, prediction, result, payload)
    return prediction, result, False


def evaluate_variant(name: str, truth: pd.DataFrame, prediction: pd.DataFrame, reference_daily: np.ndarray | None, config: dict) -> dict:
    result = {"variant": name, **score_prediction(truth, prediction)}
    if reference_daily is not None:
        candidate = np.asarray(result["daily_rank_correlations"], dtype=float)
        result["delta_vs_current_market"] = float(result["official_metric"] - (reference_daily.mean() / reference_daily.std(ddof=0)))
        result["paired_block_95_interval"] = paired_block_interval(
            reference_daily,
            candidate,
            repetitions=int(config["bootstrap_repetitions"]),
            block=int(config["bootstrap_block_dates"]),
            seed=int(config["seed"]),
        )
    return result


def preflight() -> int:
    failures = []
    for relative in ["data/raw/train.csv", "data/raw/train_labels.csv", "data/raw/target_pairs.csv", "configs/research.json"]:
        if not (ROOT / relative).exists():
            failures.append(f"missing {relative}")
    try:
        import torch

        torch_version = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch_version = None
        device = None
        failures.append("PyTorch is not installed in this Python environment")
    status = {
        "root": str(ROOT),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "python": sys.version.split()[0],
        "torch": torch_version,
        "device": device,
        "failures": failures,
    }
    print(json.dumps(status, indent=2))
    return 1 if failures else 0


def run(mode: str) -> int:
    if preflight() != 0:
        raise RuntimeError(
            "Preflight failed; refusing to start competitive feature-forecast run"
        )
    base_config = json.loads(CONFIG_PATH.read_text())
    config = config_for_mode(base_config, mode)
    rid = run_id(config, mode)
    directory = ROOT / "artifacts" / "competitive_feature_forecast" / rid
    directory.mkdir(parents=True, exist_ok=True)
    event("run_start", {"run_id": rid, "mode": mode, "git_head": git(["rev-parse", "HEAD"]), "config": config})

    x, y, pairs = load_data(ROOT)
    parent_config = json.loads((ROOT / "configs/research.json").read_text())
    folds, _ = make_folds(len(x), parent_config)
    embargo = int(config["terminal_embargo_dates"])
    folds[-1] = replace(folds[-1], validation_stop=folds[-1].validation_stop - embargo)
    if [f.validation_stop - f.validation_start for f in folds] != [180, 180, 175]:
        raise ValueError("Expected canonical 535-date development geometry")
    if mode in {"smoke", "first-fold"}:
        folds = folds[:1]

    assets = sorted({asset for pair in pairs["pair"] for asset in pair.split(" - ")})
    missing_assets = sorted(set(assets) - set(x.columns))
    if missing_assets:
        raise ValueError(f"Target-linked assets missing from train.csv: {missing_assets[:5]}")
    asset_values = x[assets].to_numpy(dtype=float)

    all_fold_results = []
    all_predictions: dict[str, list[pd.DataFrame]] = {}
    all_truth = []
    fold_lengths = []

    for fold in folds:
        prediction_positions = np.arange(fold.validation_start, fold.validation_stop)
        truth = y.iloc[prediction_positions].copy()
        fold_lengths.append(len(truth))
        all_truth.append(truth)
        baseline = load_baseline(config, fold.number, truth)
        baseline_result = evaluate_variant("current_market", truth, baseline, None, config)
        baseline_daily = np.asarray(baseline_result["daily_rank_correlations"], dtype=float)
        baseline_payload = {"fold": fold.number, "variant": "current_market", "baseline_lineage": config["baseline_s3_lineage"]}
        save_prediction(prediction_stage(directory, fold.number, "current_market"), baseline, baseline_result, baseline_payload)
        fold_results = [baseline_result]
        predictions = {"current_market": baseline}

        train_y = y.iloc[: fold.train_stop]
        cheap = {
            "mean_rank_prior": mean_rank_prior(train_y, truth.index),
            "regularized_kelly_rank_prior": regularized_kelly_rank_prior(train_y, truth.index),
            "released_mean_5": released_label_signal(y, pairs, prediction_positions, int(config["released_mean_window"])),
        }
        for name, prediction in cheap.items():
            result = evaluate_variant(name, truth, prediction, baseline_daily, config)
            save_prediction(prediction_stage(directory, fold.number, name), prediction, result, {"fold": fold.number, "variant": name, "run_id": rid})
            fold_results.append(result)
            predictions[name] = prediction

        for space in config["forecast_spaces"]:
            variant = f"lstm_feature_forecast_{space}"
            stage = prediction_stage(directory, fold.number, variant)
            payload = {
                "fold": fold.number,
                "variant": variant,
                "run_id": rid,
                "train_stop": fold.train_stop,
                "validation_start": fold.validation_start,
                "validation_stop": fold.validation_stop,
            }

            def maker(
                space=space,
                variant=variant,
                stage=stage,
                fold=fold,
                prediction_positions=prediction_positions,
                truth=truth,
                baseline_daily=baseline_daily,
            ):
                started = time.monotonic()
                model, scale, training = train_feature_forecaster(asset_values, fold.train_stop, config, space, event)
                prediction = recursive_predict(model, scale, asset_values, prediction_positions, config, pairs, assets, event)
                prediction.index = truth.index
                result = evaluate_variant(variant, truth, prediction, baseline_daily, config)
                result["training"] = training
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
                stage.mkdir(parents=True, exist_ok=True)
                try:
                    import torch

                    torch.save(model.state_dict(), stage / "model.pt")
                    np.savez_compressed(stage / "scale.npz", means=scale.means, scales=scale.scales, positive_floor=scale.positive_floor, space=np.asarray([scale.space]))
                    (stage / "training.json").write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
                except Exception as exc:
                    event("checkpoint_warning", {"variant": variant, "fold": fold.number, "error": repr(exc)})
                return prediction, result

            prediction, result, reused = load_or_make_prediction(stage, payload, maker)
            event("stage_complete", {"fold": fold.number, "variant": variant, "reused": reused, "metric": result["official_metric"]})
            fold_results.append(result)
            predictions[variant] = prediction

            correction = blend(prediction, cheap["released_mean_5"], float(config["label_correction_alpha"]))
            correction_name = f"{variant}_released_correction"
            correction_result = evaluate_variant(correction_name, truth, correction, baseline_daily, config)
            save_prediction(prediction_stage(directory, fold.number, correction_name), correction, correction_result, {"fold": fold.number, "variant": correction_name, "run_id": rid})
            fold_results.append(correction_result)
            predictions[correction_name] = correction

            ensemble = blend(correction, baseline, float(config["fixed_ensemble_weight"]))
            ensemble_name = f"current_market_plus_{correction_name}"
            ensemble_result = evaluate_variant(ensemble_name, truth, ensemble, baseline_daily, config)
            save_prediction(prediction_stage(directory, fold.number, ensemble_name), ensemble, ensemble_result, {"fold": fold.number, "variant": ensemble_name, "run_id": rid})
            fold_results.append(ensemble_result)
            predictions[ensemble_name] = ensemble

        kelly_blend = blend(cheap["regularized_kelly_rank_prior"], baseline, 0.25)
        kelly_blend_result = evaluate_variant("current_market_plus_kelly_prior", truth, kelly_blend, baseline_daily, config)
        save_prediction(prediction_stage(directory, fold.number, "current_market_plus_kelly_prior"), kelly_blend, kelly_blend_result, {"fold": fold.number, "variant": "current_market_plus_kelly_prior", "run_id": rid})
        fold_results.append(kelly_blend_result)
        predictions["current_market_plus_kelly_prior"] = kelly_blend

        for result in fold_results:
            all_fold_results.append({"fold": fold.number, **result})
        for name, prediction in predictions.items():
            all_predictions.setdefault(name, []).append(prediction)
        event("fold_complete", {"fold": fold.number, "scores": {r["variant"]: r["official_metric"] for r in fold_results}})

    pooled_truth = pd.concat(all_truth)
    pooled = []
    for name, frames in all_predictions.items():
        if len(frames) != len(folds):
            continue
        prediction = pd.concat(frames)
        result = {"variant": name, **score_prediction(pooled_truth, prediction)}
        pooled.append(result)
    pooled.sort(key=lambda row: row["official_metric"], reverse=True)
    summary = {
        "status": "completed",
        "run_id": rid,
        "mode": mode,
        "git_head": git(["rev-parse", "HEAD"]),
        "fold_lengths": fold_lengths,
        "development_dates": int(sum(fold_lengths)),
        "reserved_final_origins_evaluated": False,
        "training_fits_upper_bound": len(folds) * len(config["forecast_spaces"]) * 2,
        "fold_results": all_fold_results,
        "pooled_results": pooled,
        "best_pooled": pooled[0] if pooled else None,
        "canonical_reference": {"name": "current_market", "expected_535_date_metric": 0.3097087232124053},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (directory / "DONE.json").write_text(json.dumps({"status": "DONE", "utc": now_utc(), "run_id": rid}, indent=2) + "\n")
    latest = ROOT / "artifacts" / "competitive_feature_forecast" / "LATEST.json"
    latest.write_text(json.dumps({"run_id": rid, "mode": mode, "directory": str(directory.relative_to(ROOT)), "summary": str((directory / "summary.json").relative_to(ROOT))}, indent=2) + "\n")
    event("run_complete", {"run_id": rid, "mode": mode, "best": summary["best_pooled"]})
    print(f"RESULT_DIR={directory}")
    return 0


def package_return() -> int:
    latest_path = ROOT / "artifacts" / "competitive_feature_forecast" / "LATEST.json"
    if not latest_path.exists():
        raise SystemExit("No competitive feature-forecast result exists yet")
    latest = json.loads(latest_path.read_text())
    result_dir = ROOT / latest["directory"]
    output = ROOT / "competitive_feature_forecast_return.zip"
    include = [result_dir / "summary.json", result_dir / "DONE.json", latest_path, LOG_PATH]
    notebook = ROOT / "notebooks/20_competitive_feature_forecast.ipynb"
    if notebook.exists():
        include.append(notebook)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include:
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
        for path in sorted(result_dir.glob("fold_*/*/result.json")):
            archive.write(path, path.relative_to(ROOT))
        for path in sorted(result_dir.glob("fold_*/*/training.json")):
            archive.write(path, path.relative_to(ROOT))
    print(f"RETURN_ZIP={output}")
    print(f"SHA256={sha256(output)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "first-fold", "full"])
    parser.add_argument("--package-return", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.package_return:
        return package_return()
    if not args.mode:
        parser.error("Choose --preflight, --mode, or --package-return")
    return run(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
