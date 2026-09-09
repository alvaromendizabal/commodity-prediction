"""Independent implementation of Kaggle's daily rank-correlation Sharpe metric.

Reference: https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric
Uses average ranks and population standard deviation; no annualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def daily_rank_correlations(y: pd.DataFrame, prediction: pd.DataFrame) -> np.ndarray:
    if not y.index.equals(prediction.index) or not y.columns.equals(prediction.columns):
        raise ValueError("Prediction row IDs and ordered target columns must match")
    truth = y.replace(-999999, np.nan).to_numpy(dtype=float)
    pred = prediction.to_numpy(dtype=float)
    if not np.isfinite(pred).all() or np.isinf(truth).any():
        raise ValueError("Predictions must be finite; targets may only use NaN as missing")
    correlations = []
    for row, estimated in zip(truth, pred, strict=True):
        mask = np.isfinite(row)
        if mask.sum() < 2:
            raise ValueError("At least two observed targets are required per date")
        a, b = rankdata(row[mask]), rankdata(estimated[mask])
        a -= a.mean()
        b -= b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            raise ZeroDivisionError("Constant daily predictions or targets cannot be ranked")
        correlations.append(float(a @ b / denominator))
    return np.asarray(correlations)


def correlation_sharpe(correlations: np.ndarray) -> float:
    if len(correlations) < 2 or not np.isfinite(correlations).all():
        raise ValueError("Need at least two finite daily correlations")
    std = correlations.std(ddof=0)
    if std == 0:
        raise ZeroDivisionError("Daily correlation has zero variance")
    return float(correlations.mean() / std)


def score(y: pd.DataFrame, prediction: pd.DataFrame) -> float:
    return correlation_sharpe(daily_rank_correlations(y, prediction))


def paired_block_interval(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int = 500,
    block: int = 20,
    seed: int = 42,
) -> list[float]:
    """Conditional paired circular block bootstrap; no model refitting."""
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Paired daily observations must have the same one-dimensional shape")
    rng = np.random.default_rng(seed)
    n = len(reference)
    deltas = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        deltas.append(
            correlation_sharpe(candidate[indices]) - correlation_sharpe(reference[indices])
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()
