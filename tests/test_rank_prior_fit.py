"""Contracts for the matched fold-local diagonal-rank fitted study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.rank_prior_fit.features import (
    RANK_TEMPLATE,
    append_training_rank,
    settings,
    training_rank_scores,
)
from commodity_prediction.domain.rank_prior_fit.run import declared_comparisons, study_lineage


def labels(rows: int = 180, columns: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(rows, columns)) + np.linspace(-0.5, 0.5, columns)[None, :]
    return pd.DataFrame(
        values,
        index=pd.Index(range(rows), name="date_id"),
        columns=[f"target_{i}" for i in range(columns)],
    )


def panel_for(y: pd.DataFrame) -> Panel:
    values = np.arange(len(y) * len(y.columns), dtype=np.float32).reshape(
        len(y), len(y.columns), 1
    )
    return Panel(values, y.index.tolist(), list(y.columns), ["reference__synthetic"], {})


def test_training_rank_uses_only_declared_prefix():
    y = labels()
    changed = y.copy()
    changed.iloc[120:] = changed.iloc[120:] * -1000 + 500
    pd.testing.assert_series_equal(
        training_rank_scores(y, 120), training_rank_scores(changed, 120)
    )


def test_rank_template_is_target_specific_but_constant_over_dates():
    y = labels()
    result = append_training_rank(panel_for(y), y, 120)
    assert result.names[-1] == RANK_TEMPLATE
    rank = result.values[:, :, -1]
    np.testing.assert_array_equal(rank[0], rank[-1])
    assert np.ptp(rank[0]) > 0
    result.validate()


def test_later_training_prefix_can_update_rank_metadata():
    y = labels()
    early = training_rank_scores(y, 90).to_numpy()
    late = training_rank_scores(y, 150).to_numpy()
    assert not np.array_equal(early, late)


def test_rank_template_rejects_axis_mismatch():
    y = labels()
    panel = panel_for(y)
    altered = y.rename(columns={"target_0": "wrong"})
    try:
        append_training_rank(panel, altered, 120)
    except ValueError as exc:
        assert "axes" in str(exc)
    else:
        raise AssertionError("Mismatched target axes must fail closed")


def test_admission_settings_remove_budget_and_correlation_screen():
    y = labels()
    panel = append_training_rank(panel_for(y), y, 120)
    configured = settings({"max_features": 1, "max_abs_correlation": 0.5}, panel)
    assert configured["max_features"] == len(panel.names)
    assert configured["max_abs_correlation"] == 1.01


def test_declared_comparisons_are_small_and_matched():
    assert declared_comparisons() == [
        ("rank_tail", "admitted_tail"),
        ("rank_current_market", "current_market"),
        ("rank_current_market", "rank_tail"),
    ]


def test_child_lineage_preserves_parent_probe_and_market_path():
    root = Path(__file__).resolve().parents[1]
    lineage, evidence = study_lineage(root)
    config = json.loads((root / "configs/rank_prior_fit_study.json").read_text())
    assert len(lineage) == 64
    assert evidence["parent_market_path_lineage"] == config["parent_market_path_lineage"]
    assert evidence["parent_rank_probe_lineage"] == config["parent_rank_probe_lineage"]
    assert config["max_new_fits"] == 6
    assert config["probe_fits"] == 2
    assert config["final_test_start_date_id"] == 1714
    assert config["holdout_evaluated"] is False
