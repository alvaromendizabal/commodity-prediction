#!/usr/bin/env python3
"""Bounded runner for the documented 3rd-place MITSUI strategy reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from commodity_prediction.competitive.third_place_reproduction import (
    build_causal_pair_features,
    choose_stratified_targets,
    fit_causal_oof_stack,
    fit_source_described_stack,
    prefix_invariant_feature_check,
    prepare_train_valid,
)
from commodity_prediction.data import load_data, make_folds, reconstruct_targets
from commodity_prediction.metrics import daily_rank_correlations, paired_block_interval, score

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/third_place_reproduction.json"
ARTIFACT_ROOT = ROOT / "artifacts/third_place_reproduction"
BASELINE_CACHE = ROOT / "artifacts/baselines/current_market"


def utc() -> str:
    return datetime.now(UTC).isoformat()


def emit(event: str, **payload) -> None:
    row = {"utc": utc(), "event": event, **payload}
    print(json.dumps(row, sort_keys=True), flush=True)


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def current_git() -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    return head, branch


def preflight() -> int:
    failures = []
    versions = {}
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except Exception:
        failures.append("LightGBM is not installed")
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        failures.append("XGBoost is not installed")

    try:
        x, y, pairs = load_data(ROOT)
        cfg = config()
        folds, _ = make_folds(
            len(x),
            {
                "holdout_dates": 252,
                "validation_dates": 180,
                "n_folds": 3,
                "purge_dates": 5,
                "min_train_dates": 600,
            },
        )
        fold = folds[cfg["fold_number"]]
        sample_pair = pairs.iloc[0]["pair"]
        if not prefix_invariant_feature_check(
            x,
            sample_pair,
            min(700, fold.train_stop),
            positive_lags=cfg["positive_lags"],
            rolling_windows=cfg["rolling_windows"],
        ):
            failures.append("Causal feature prefix invariance failed")
        reconstructed = reconstruct_targets(x, pairs)
        mask = reconstructed.notna() & y.notna()
        vals = (reconstructed - y).where(mask).stack().abs()
        max_err = float(vals.max()) if len(vals) else float("nan")
        reconstruction_quantiles = (
            {
                "p50": float(vals.quantile(0.50)),
                "p95": float(vals.quantile(0.95)),
                "p99": float(vals.quantile(0.99)),
                "p999": float(vals.quantile(0.999)),
                "max": max_err,
            }
            if len(vals)
            else None
        )
        tolerance = float(cfg["target_reconstruction_tolerance"])
        if not np.isfinite(max_err) or max_err > tolerance:
            failures.append(
                f"Target reconstruction mismatch: max_abs_error={max_err}, tolerance={tolerance}"
            )
    except Exception as exc:
        failures.append(f"Data/feature preflight failed: {type(exc).__name__}: {exc}")
        max_err = None

    head, branch = current_git()
    result = {
        "root": str(ROOT),
        "git_head": head,
        "git_branch": branch,
        "python": sys.version.split()[0],
        "target_reconstruction_max_abs_error": max_err,
        "target_reconstruction_tolerance": float(config()["target_reconstruction_tolerance"]),
        "target_reconstruction_error_quantiles": (
            reconstruction_quantiles if "reconstruction_quantiles" in locals() else None
        ),
        "model_library_versions": versions,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 2


def baseline_key(cfg: dict, fold_number: int) -> str:
    return (
        f"checkpoints/artifacts/{cfg['baseline_s3_lineage']}/"
        f"fold_{fold_number}/current_market/predictions.parquet"
    )


def load_baseline(cfg: dict, fold_number: int) -> pd.DataFrame:
    BASELINE_CACHE.mkdir(parents=True, exist_ok=True)
    path = BASELINE_CACHE / f"fold_{fold_number}.parquet"
    if not path.exists():
        boto3.client("s3").download_file(
            cfg["baseline_s3_bucket"], baseline_key(cfg, fold_number), str(path)
        )
    frame = pd.read_parquet(path)
    if "date_id" in frame.columns:
        frame = frame.set_index("date_id")
    if frame.index.name is None:
        frame.index.name = "date_id"
    expected = [f"target_{i}" for i in range(424)]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"Baseline is missing targets: {missing[:5]}")
    return frame[expected]


def target_payload(target: str, pair: str, cfg: dict, fold_number: int) -> str:
    body = json.dumps(
        {
            "target": target,
            "pair": pair,
            "positive_lags": cfg["positive_lags"],
            "rolling_windows": cfg["rolling_windows"],
            "oof_splits": cfg["oof_splits"],
            "oof_gap": cfg["oof_gap"],
            "seed": cfg["seed"],
            "fold": fold_number,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def run_target(
    target: str,
    pair: str,
    x: pd.DataFrame,
    y: pd.DataFrame,
    fold,
    cfg: dict,
    run_dir: Path,
) -> dict:
    stage = run_dir / "targets" / target
    stage.mkdir(parents=True, exist_ok=True)
    receipt = stage / "result.json"
    pred_path = stage / "predictions.parquet"

    if receipt.exists() and pred_path.exists():
        result = json.loads(receipt.read_text())
        emit("target_reused", target=target)
        return result

    features = build_causal_pair_features(
        x,
        pair,
        positive_lags=cfg["positive_lags"],
        rolling_windows=cfg["rolling_windows"],
    )
    x_train = features.iloc[: fold.train_stop]
    x_valid = features.iloc[fold.validation_start : fold.validation_stop]
    y_train = y[target].iloc[: fold.train_stop]
    y_valid = y[target].iloc[fold.validation_start : fold.validation_stop]

    mask = y_train.notna()
    prepared = prepare_train_valid(x_train.loc[mask], x_valid)
    source_pred, source_parts = fit_source_described_stack(
        prepared.train,
        y_train.loc[mask].to_numpy(dtype=float),
        prepared.valid,
        seed=cfg["seed"],
        threads=cfg["threads"],
    )
    oof_pred, oof_parts = fit_causal_oof_stack(
        x_train,
        y_train,
        x_valid,
        seed=cfg["seed"],
        threads=cfg["threads"],
        n_splits=cfg["oof_splits"],
        gap=cfg["oof_gap"],
    )

    pred = pd.DataFrame(
        {
            "date_id": x_valid.index,
            "truth": y_valid.to_numpy(dtype=float),
            "source_stack": source_pred,
            "oof_stack": oof_pred,
            **{f"source_{k}": v for k, v in source_parts.items() if k != "stack"},
            **{f"oof_{k}": v for k, v in oof_parts.items() if k != "stack_oof"},
        }
    )
    pred.to_parquet(pred_path, index=False)

    observed = np.isfinite(pred["truth"])
    result = {
        "target": target,
        "pair": pair,
        "fold": fold.number,
        "feature_count": int(features.shape[1]),
        "train_rows": int(mask.sum()),
        "valid_rows": int(len(x_valid)),
        "source_stack_pearson": float(
            np.corrcoef(pred.loc[observed, "truth"], pred.loc[observed, "source_stack"])[0, 1]
        )
        if observed.sum() > 2
        else None,
        "oof_stack_pearson": float(
            np.corrcoef(pred.loc[observed, "truth"], pred.loc[observed, "oof_stack"])[0, 1]
        )
        if observed.sum() > 2
        else None,
        "payload": target_payload(target, pair, cfg, fold.number),
        "completed_utc": utc(),
    }
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit("target_complete", **result)
    return result


def assemble_variant(
    baseline: pd.DataFrame,
    selected: list[str],
    run_dir: Path,
    column: str,
    weight: float,
) -> pd.DataFrame:
    out = baseline.copy()
    for target in selected:
        p = pd.read_parquet(run_dir / "targets" / target / "predictions.parquet")
        p = p.set_index("date_id")
        idx = out.index.intersection(p.index)
        out.loc[idx, target] = (1.0 - weight) * out.loc[idx, target].to_numpy(
            dtype=float
        ) + weight * p.loc[idx, column].to_numpy(dtype=float)
    return out


def run(mode: str) -> int:
    if preflight() != 0:
        raise RuntimeError("Preflight failed; refusing to train")

    cfg = config()
    x, y, pairs = load_data(ROOT)
    folds, development_stop = make_folds(
        len(x),
        {
            "holdout_dates": 252,
            "validation_dates": 180,
            "n_folds": 3,
            "purge_dates": 5,
            "min_train_dates": 600,
        },
    )
    fold = folds[cfg["fold_number"]]

    n = {
        "smoke": cfg["smoke_targets"],
        "panel": cfg["panel_targets"],
        "full": 424,
    }[mode]
    selected = choose_stratified_targets(pairs, n)

    head, branch = current_git()
    run_payload = json.dumps(
        {
            "head": head,
            "mode": mode,
            "fold": fold.number,
            "selected": selected,
            "config": cfg,
        },
        sort_keys=True,
    ).encode()
    run_id = hashlib.sha256(run_payload).hexdigest()[:16]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "selected_targets.json").write_text(json.dumps(selected, indent=2) + "\n")

    emit(
        "run_start",
        run_id=run_id,
        mode=mode,
        fold=fold.number,
        selected_targets=len(selected),
        git_head=head,
        git_branch=branch,
        development_stop=development_stop,
        reserved_final_origins_evaluated=False,
    )

    pair_map = pairs.set_index("target")["pair"].to_dict()
    started = time.time()
    for i, target in enumerate(selected, 1):
        run_target(target, pair_map[target], x, y, fold, cfg, run_dir)
        if i == 1 or i % 4 == 0 or i == len(selected):
            emit(
                "progress",
                completed=i,
                total=len(selected),
                elapsed_seconds=round(time.time() - started, 2),
            )

    expected = [f"target_{i}" for i in range(424)]
    y_valid = y.iloc[fold.validation_start : fold.validation_stop][expected]
    baseline = load_baseline(cfg, fold.number)
    if not baseline.index.equals(y_valid.index):
        if len(baseline) == len(y_valid) and isinstance(baseline.index, pd.RangeIndex):
            baseline = baseline.copy()
            baseline.index = y_valid.index
        else:
            baseline = baseline.reindex(y_valid.index)
    if baseline.isna().all(axis=None):
        raise ValueError("Baseline alignment failed")

    scores = {"current_market": score(y_valid, baseline)}
    correlations = {"current_market": daily_rank_correlations(y_valid, baseline).tolist()}

    for stack_name, pred_col in [
        ("source_stack", "source_stack"),
        ("causal_oof_stack", "oof_stack"),
    ]:
        for weight in cfg["fixed_replacement_weights"]:
            variant = assemble_variant(baseline, selected, run_dir, pred_col, weight)
            name = f"{stack_name}__replace_{weight:.2f}"
            scores[name] = score(y_valid, variant)
            correlations[name] = daily_rank_correlations(y_valid, variant).tolist()

    best_name = max(scores, key=scores.get)
    intervals = {}
    ref = np.asarray(correlations["current_market"], dtype=float)
    for name, corr in correlations.items():
        if name == "current_market":
            continue
        intervals[name] = paired_block_interval(
            ref,
            np.asarray(corr, dtype=float),
            repetitions=cfg["bootstrap_repetitions"],
            block=cfg["bootstrap_block_dates"],
            seed=cfg["seed"],
        )

    summary = {
        "run_id": run_id,
        "mode": mode,
        "fold": fold.number,
        "selected_targets": selected,
        "selected_target_count": len(selected),
        "scores": scores,
        "paired_block_delta_intervals": intervals,
        "best_variant": best_name,
        "best_score": scores[best_name],
        "baseline_score": scores["current_market"],
        "reserved_final_origins_evaluated": False,
        "source_fidelity": {
            "documented": [
                "one model per target",
                "target_pairs routing",
                "positive/negative lag family",
                "rolling mean/max family",
                "difference features",
                "safe log transform",
                "median imputation",
                "standardization",
                "LightGBM + RandomForest + XGBoost base models",
                "XGBoost meta-model",
                "in-sample meta-feature source-described stack",
            ],
            "source_unspecified": cfg["source_unspecified_reconstruction_choices"],
            "forensic_negative_lags_trained": False,
        },
        "completed_utc": utc(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "DONE.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "complete",
                "reserved_final_origins_evaluated": False,
                "completed_utc": utc(),
            },
            indent=2,
        )
        + "\n"
    )

    emit("run_complete", run_id=run_id, best=best_name, scores=scores)
    print(f"RESULT_DIR={run_dir}")
    return 0


def package_return() -> int:
    out = ROOT / "third_place_reproduction_return.zip"
    import zipfile

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ARTIFACT_ROOT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
        for p in [
            ROOT / "configs/third_place_reproduction.json",
            ROOT / "docs/ROUND_21A_PROTOCOL.md",
            ROOT / "notebooks/21_third_place_reproduction.ipynb",
            ROOT / "scripts/run_third_place_reproduction.py",
            ROOT / "src/commodity_prediction/competitive/third_place_reproduction.py",
            ROOT / "tests/test_third_place_reproduction.py",
        ]:
            if p.exists():
                z.write(p, p.relative_to(ROOT))
    print(f"RETURN_ZIP={out}")
    print(f"SHA256={hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "panel", "full"])
    parser.add_argument("--package-return", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.package_return:
        return package_return()
    if args.mode:
        return run(args.mode)
    parser.error("choose --preflight, --mode, or --package-return")


if __name__ == "__main__":
    raise SystemExit(main())
