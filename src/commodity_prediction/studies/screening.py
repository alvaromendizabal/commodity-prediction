"""Training-only preprocessing, per-output screening, and temporal stability controls."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedFeatures:
    original_columns: list[str]
    columns: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    rejected: dict[str, str]
    signatures: list[str]


def prepare_features(train: pd.DataFrame, valid: pd.DataFrame, config: dict) -> PreparedFeatures:
    if list(train.columns) != list(valid.columns) or not train.columns.is_unique:
        raise ValueError("Feature schema differs or contains duplicates")
    if np.isinf(train.to_numpy()).any() or np.isinf(valid.to_numpy()).any():
        raise ValueError("Infinite candidate values")
    missing = train.isna().mean()
    cardinality = train.nunique()
    rejected = {
        c: "missingness" if missing[c] > config["max_missing_fraction"] else "constant"
        for c in train
        if missing[c] > config["max_missing_fraction"] or cardinality[c] < 2
    }
    columns = [c for c in train if c not in rejected]
    if not columns:
        raise ValueError("No usable candidates in the training interval")
    fit = train[columns].to_numpy(dtype=float)
    medians = np.nanmedian(fit, axis=0)
    fit = np.where(np.isnan(fit), medians, fit)
    means, scales = fit.mean(axis=0), fit.std(axis=0, ddof=0)
    if np.any(scales <= 0):
        raise ValueError("Unexpected degenerate training scale")
    val = valid[columns].to_numpy(dtype=float)
    val = np.where(np.isnan(val), medians, val)
    signatures = [hashlib.sha256(fit[:, i].tobytes()).hexdigest() for i in range(fit.shape[1])]
    return PreparedFeatures(
        list(train.columns),
        columns,
        medians,
        means,
        scales,
        np.ascontiguousarray((fit - means) / scales),
        np.ascontiguousarray((val - means) / scales),
        rejected,
        signatures,
    )


def correlations(x: np.ndarray, y: np.ndarray, minimum: int) -> np.ndarray:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Correlation matrices must share a training interval")
    observed = np.isfinite(y)
    mask = observed.astype(float)
    count = mask.sum(axis=0)
    sums = np.where(observed, y, 0).sum(axis=0)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    centered = np.where(observed, y - mean, 0)
    sx, sx2 = x.T @ mask, (x * x).T @ mask
    variance = sx2 - np.divide(sx * sx, count, out=np.zeros_like(sx), where=count > 0)
    denominator = np.sqrt(np.maximum(variance, 0) * (centered * centered).sum(axis=0))
    result = np.divide(
        x.T @ centered, denominator, out=np.zeros_like(denominator), where=denominator > 1e-12
    )
    result[:, count < minimum] = 0
    return np.clip(result, -1, 1)


def relevance_scores(x: np.ndarray, y: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    full = correlations(x, y, config["minimum_output_observations"])
    middle = len(x) // 2
    first = correlations(x[:middle], y[:middle], config["minimum_half_observations"])
    second = correlations(x[middle:], y[middle:], config["minimum_half_observations"])
    # Rank stable features conservatively; do not force a feature with a reversed sign.
    stable = np.where(
        first * second > 0, np.minimum(np.abs(first), np.abs(second)) + 0.25 * np.abs(full), 0
    )
    return np.abs(full), stable


def select_features(
    prepared: PreparedFeatures,
    candidate_names: list[str],
    relevance: np.ndarray,
    config: dict,
    require_stability: bool,
) -> tuple[list[int], dict]:
    if len(relevance) != len(prepared.columns):
        raise ValueError("Relevance and usable feature schema differ")
    positions = {name: i for i, name in enumerate(prepared.columns)}
    reasons: Counter[str] = Counter()
    usable = []
    seen = set()
    for name in candidate_names:
        if name in prepared.rejected:
            reasons[prepared.rejected[name]] += 1
            continue
        idx = positions[name]
        signature = prepared.signatures[idx]
        if signature in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(signature)
        usable.append(idx)
    order = sorted(usable, key=lambda i: (-float(relevance[i]), i))
    selected: list[int] = []
    for idx in order:
        if require_stability and relevance[idx] <= 0:
            reasons["unstable_sign_or_insufficient_history"] += 1
        elif len(selected) >= config["max_features_per_output"]:
            reasons["feature_budget"] += 1
        elif (
            selected
            and np.max(
                np.abs(prepared.train[:, selected].T @ prepared.train[:, idx] / len(prepared.train))
            )
            > config["max_abs_correlation"]
        ):
            reasons["correlated"] += 1
        else:
            selected.append(idx)
    if len(selected) + sum(reasons.values()) != len(candidate_names):
        raise ValueError("Screening accounting does not balance")
    audit = {
        "candidate_count": len(candidate_names),
        "retained_count": len(selected),
        "rejected_count": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "selected": [prepared.columns[i] for i in selected],
        "training_relevance": [float(relevance[i]) for i in selected],
    }
    return selected, audit
