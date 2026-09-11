"""Training-only, metric-aligned target-rank priors for bounded diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from commodity_prediction.domain.catalog import Panel

SENTINEL = -999999
RANK_TEMPLATE = "rank_prior__diagonal"


def _clean(y: pd.DataFrame) -> pd.DataFrame:
    return y.replace([SENTINEL, np.inf, -np.inf], np.nan).astype(float)


def standardized_daily_ranks(y: pd.DataFrame) -> pd.DataFrame:
    """Center and unit-normalize each date's observed cross-sectional ranks.

    Missing targets become neutral zeros only after the observed ranks are
    centered and normalized. This mirrors daily Spearman geometry while giving
    each training date comparable scale in the covariance diagnostic.
    """
    clean = _clean(y)
    ranks = clean.rank(axis=1, method="average", na_option="keep")
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    norms = np.sqrt(centered.pow(2).sum(axis=1, skipna=True))
    if (norms <= 0).any() or (~np.isfinite(norms)).any():
        raise ValueError("Every rank-prior training date needs at least two observed targets")
    normalized = centered.div(norms, axis=0).fillna(0.0)
    values = normalized.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite standardized rank matrix")
    return normalized


def raw_mean_scores(y: pd.DataFrame) -> pd.Series:
    clean = _clean(y)
    scores = clean.mean(axis=0)
    if scores.isna().any():
        raise ValueError("Every target needs at least one training label")
    return scores


def mean_rank_scores(y: pd.DataFrame) -> pd.Series:
    ranks = standardized_daily_ranks(y)
    scores = ranks.mean(axis=0)
    if np.ptp(scores.to_numpy(dtype=float)) <= 1e-12:
        raise ValueError("Mean-rank prior is degenerate")
    return scores


def diagonal_rank_scores(y: pd.DataFrame) -> pd.Series:
    ranks = standardized_daily_ranks(y)
    values = ranks.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    variance = values.var(axis=0, ddof=0)
    positive = variance[variance > 1e-12]
    floor = float(np.median(positive)) if len(positive) else 1.0
    score = mean / np.maximum(variance, floor * 0.05)
    if not np.isfinite(score).all() or np.ptp(score) <= 1e-12:
        raise ValueError("Diagonal rank prior is invalid")
    return pd.Series(score, index=y.columns, dtype=float)


def ledoit_rank_scores(y: pd.DataFrame) -> pd.Series:
    ranks = standardized_daily_ranks(y)
    values = ranks.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    covariance = LedoitWolf(assume_centered=False).fit(values).covariance_
    scale = float(np.trace(covariance) / len(covariance))
    ridge = max(scale, 1.0) * 1e-10
    score = np.linalg.solve(covariance + np.eye(len(covariance)) * ridge, mean)
    if not np.isfinite(score).all() or np.ptp(score) <= 1e-12:
        raise ValueError("Ledoit-Wolf rank prior is invalid")
    return pd.Series(score, index=y.columns, dtype=float)


def constant_prediction(index: pd.Index, columns: pd.Index, scores: pd.Series) -> pd.DataFrame:
    aligned = scores.reindex(columns)
    if aligned.isna().any() or not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Rank-prior score schema differs from prediction targets")
    if np.ptp(aligned.to_numpy(dtype=float)) <= 1e-12:
        raise ValueError("Constant cross-target ordering cannot be evaluated")
    values = np.broadcast_to(aligned.to_numpy(dtype=float), (len(index), len(columns))).copy()
    return pd.DataFrame(values, index=index, columns=columns)


def append_diagonal_rank(panel: Panel, training_labels: pd.DataFrame) -> Panel:
    """Append one fold-local target-ordering template estimated from training labels only."""
    if list(training_labels.columns) != panel.targets:
        raise ValueError("Rank-prior training labels must match the panel target order")
    if RANK_TEMPLATE in panel.names:
        raise ValueError("Diagonal-rank template is already present")
    scores = diagonal_rank_scores(training_labels).reindex(panel.targets)
    if scores.isna().any():
        raise ValueError("Diagonal-rank target schema differs from the panel")
    values = np.broadcast_to(
        scores.to_numpy(dtype=np.float32)[None, :, None],
        (len(panel.dates), len(panel.targets), 1),
    ).copy()
    result = Panel(
        np.concatenate([panel.values, values], axis=2),
        list(panel.dates),
        list(panel.targets),
        [*panel.names, RANK_TEMPLATE],
        dict(panel.source_series),
    )
    result.validate()
    return result


def probe_methods() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "raw_mean": raw_mean_scores,
        "mean_rank": mean_rank_scores,
        "diagonal_rank": diagonal_rank_scores,
        "ledoit_rank": ledoit_rank_scores,
    }
