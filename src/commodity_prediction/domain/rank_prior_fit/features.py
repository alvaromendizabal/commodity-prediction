"""Fold-local diagonal-rank metadata appended to frozen matched feature panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.compact.features import BASE, Variant as CompactVariant
from commodity_prediction.domain.compact.features import build_panel as compact_panel
from commodity_prediction.domain.market_path.features import build_panel as market_path_panel
from commodity_prediction.domain.rank_prior.features import diagonal_rank_scores

RANK_TEMPLATE = "rank_prior_fit__diagonal_rank_training_prefix"
VARIANTS = ("rank_tail", "rank_current_market")


def training_rank_scores(y: pd.DataFrame, train_stop: int) -> pd.Series:
    if train_stop <= 0 or train_stop > len(y):
        raise ValueError("Invalid rank-prior training boundary")
    scores = diagonal_rank_scores(y.iloc[:train_stop])
    if not scores.index.equals(y.columns):
        raise ValueError("Rank-prior target ordering differs")
    return scores


def append_training_rank(panel: Panel, y: pd.DataFrame, train_stop: int) -> Panel:
    if panel.dates != y.index.tolist() or panel.targets != list(y.columns):
        raise ValueError("Rank-prior feature axes must match the development panel")
    scores = training_rank_scores(y, train_stop).to_numpy(dtype=np.float32)
    values = np.broadcast_to(scores[None, :], (len(y), len(scores))).copy()
    result = Panel(
        np.concatenate([panel.values, values[:, :, None]], axis=2),
        panel.dates,
        panel.targets,
        [*panel.names, RANK_TEMPLATE],
        {**panel.source_series, "rank_prior_fit": len(scores)},
    )
    result.validate()
    return result


def build_variant_panel(
    original: Panel,
    x: pd.DataFrame,
    y: pd.DataFrame,
    pairs: pd.DataFrame,
    train_stop: int,
    variant: str,
) -> tuple[Panel, dict]:
    if variant not in VARIANTS:
        raise ValueError("Undeclared rank-prior fitted variant")
    if original.dates != x.index.tolist() or not x.index.equals(y.index):
        raise ValueError("Raw inputs, labels, and frozen feature panel must align")
    if variant == "rank_tail":
        baseline = compact_panel(
            original,
            pairs,
            CompactVariant("rank_tail", (*BASE, "tail_risk"), admit=True),
        )
        coverage = {"base": "admitted_tail", "base_templates": len(baseline.names)}
    else:
        baseline, market_coverage = market_path_panel(original, x, pairs, "current_market")
        coverage = {
            "base": "current_market",
            "base_templates": len(baseline.names),
            "market_coverage": market_coverage,
        }
    result = append_training_rank(baseline, y, train_stop)
    coverage.update(
        {
            "rank_templates_added": 1,
            "candidate_templates": len(result.names),
            "training_stop": train_stop,
            "rank_template": RANK_TEMPLATE,
        }
    )
    return result, coverage


def settings(config: dict, panel: Panel) -> dict:
    return {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
