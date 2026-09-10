"""Feature reliance and temporal stability; neither is causal importance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations
from commodity_prediction.studies.evaluation import block_permutation

from .catalog import Panel
from .model import Model


def group_permutations(
    model: Model, panel: Panel, truth: pd.DataFrame, start: int, stop: int, config: dict
) -> dict:
    """Shuffle selected family inputs in shared date blocks, preserving target alignment."""
    z = model.transform(panel, start, stop).reshape(stop - start, len(panel.targets), -1)
    baseline = correlation_sharpe(
        daily_rank_correlations(truth, model.predict(panel, start, stop, weight=1))
    )
    rng = np.random.default_rng(config["seed"])
    result = {}
    for family in sorted({name.split("__", 1)[0] for name in model.feature_names}):
        positions = [
            i for i, name in enumerate(model.feature_names) if name.startswith(family + "__")
        ]
        drops = []
        for _ in range(config["permutation_repetitions"]):
            order = block_permutation(len(truth), config["permutation_block_dates"], rng)
            changed = z.copy()
            changed[:, :, positions] = z[order][:, :, positions]
            flat = changed.reshape(-1, len(model.feature_names))
            residual = (
                flat @ model.coefficient + model.intercept
                if model.estimator is None
                else model.estimator.predict(flat)
            )
            values = model.target_mean + model.target_scale * residual.reshape(
                len(truth), len(panel.targets)
            )
            prediction = pd.DataFrame(values, index=truth.index, columns=truth.columns)
            drops.append(baseline - correlation_sharpe(daily_rank_correlations(truth, prediction)))
        result[family] = {
            "mean_metric_drop": float(np.mean(drops)),
            "metric_drop_repetitions": drops,
        }
    return result


def temporal_diagnostics(
    daily: np.ndarray, lengths: list[int], selections: list[list[str]]
) -> dict:
    offset = 0
    blocks = []
    for fold, length in enumerate(lengths):
        for start in range(0, length, 60):
            values = daily[offset + start : offset + min(start + 60, length)]
            blocks.append(
                {
                    "fold": fold,
                    "block_start_offset": start,
                    "dates": len(values),
                    "official_metric": correlation_sharpe(values),
                }
            )
        offset += length
    overlaps = []
    for a, b in zip(selections[:-1], selections[1:], strict=True):
        left, right = set(a), set(b)
        if left | right:
            overlaps.append(len(left & right) / len(left | right))
    return {"temporal_blocks": blocks, "adjacent_fold_selection_jaccard": overlaps}
