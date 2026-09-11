"""Fold-local diagonal-rank template for matched pooled-model ablations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.rank_prior.features import diagonal_rank_scores

RANK_TEMPLATE = "rank_prior_fit__diagonal"


def append_diagonal_rank(panel: Panel, training_labels: pd.DataFrame) -> Panel:
    """Append one target-aligned static template learned only from a fold's training labels."""
    if list(training_labels.columns) != panel.targets:
        raise ValueError("Rank-prior training labels must match the panel target order")
    if RANK_TEMPLATE in panel.names:
        raise ValueError("Diagonal-rank template is already present")
    scores = diagonal_rank_scores(training_labels).reindex(panel.targets)
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
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
