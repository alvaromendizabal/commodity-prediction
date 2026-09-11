"""Point-in-time, alignment, and declaration tests for the fitted rank-prior child study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.rank_prior_fit.features import RANK_TEMPLATE, append_diagonal_rank
from commodity_prediction.domain.rank_prior_fit.run import VARIANTS, declared_comparisons, study_lineage


def panel(rows: int = 180, targets: int = 12) -> Panel:
    names = [f"target_{i}" for i in range(targets)]
    values = np.zeros((rows, targets, 2), dtype=np.float32)
    return Panel(values, list(range(rows)), names, ["reference__a", "tail_risk__b"], {})


def labels(rows: int = 180, targets: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    trend = np.linspace(-0.8, 0.8, targets)
    values = rng.normal(scale=0.5, size=(rows, targets)) + trend[None, :]
    return pd.DataFrame(values, columns=[f"target_{i}" for i in range(targets)])


def test_append_diagonal_rank_is_target_aligned_and_static_over_dates():
    base = panel()
    y = labels()
    result = append_diagonal_rank(base, y.iloc[:120])
    result.validate()
    assert result.names == [*base.names, RANK_TEMPLATE]
    assert result.values.shape == (180, 12, 3)
    np.testing.assert_array_equal(result.values[0, :, -1], result.values[-1, :, -1])
    assert np.ptp(result.values[0, :, -1]) > 0
    assert np.isfinite(result.values[:, :, -1]).all()


def test_future_label_perturbation_cannot_change_fold_feature():
    base = panel()
    y = labels()
    changed = y.copy()
    changed.iloc[120:] = changed.iloc[120:] * -1000 + 12345
    first = append_diagonal_rank(base, y.iloc[:120])
    second = append_diagonal_rank(base, changed.iloc[:120])
    np.testing.assert_array_equal(first.values, second.values)


def test_later_training_prefix_can_update_rank_state_without_touching_prior_fold():
    base = panel()
    y = labels()
    early = append_diagonal_rank(base, y.iloc[:100])
    later = append_diagonal_rank(base, y.iloc[:150])
    assert not np.array_equal(early.values[0, :, -1], later.values[0, :, -1])
    np.testing.assert_array_equal(early.values[0, :, -1], early.values[99, :, -1])


def test_rank_feature_rejects_target_order_mismatch():
    base = panel()
    y = labels()[list(reversed(labels().columns))]
    try:
        append_diagonal_rank(base, y)
    except ValueError as exc:
        assert "target order" in str(exc)
    else:
        raise AssertionError("Target-order mismatch must fail closed")


def test_fitted_study_is_six_fits_with_only_two_declared_variants():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/rank_prior_fit_study.json").read_text())
    assert config["max_new_fits"] == 6
    assert config["probe_fits"] == 2
    assert config["max_run_seconds"] == 180
    assert set(config["variants"]) == set(VARIANTS)
    assert config["final_test_start"] == 1714
    assert declared_comparisons() == [
        ("admitted_tail_rank", "admitted_tail"),
        ("current_market_rank", "current_market"),
        ("current_market_rank", "admitted_tail_rank"),
        ("admitted_tail_rank", "historical_mean"),
        ("current_market_rank", "historical_mean"),
    ]


def test_child_lineage_preserves_market_path_and_probe_parents():
    root = Path(__file__).resolve().parents[1]
    lineage, evidence = study_lineage(root)
    config = json.loads((root / "configs/rank_prior_fit_study.json").read_text())
    assert len(lineage) == 64
    assert evidence["parent_market_path_lineage"] == config["parent_market_path_lineage"]
    assert evidence["probe_lineage"] == config["probe_lineage"]
    assert all("rank_prior_fit" in path for path in evidence["files"])
