"""Actual retained metric, paired uncertainty and checkpoint contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations
from commodity_prediction.studies.evaluation import block_indices, compare_predictions


def test_daily_metric_matches_independent_rank_correlations():
    y = pd.DataFrame([[1, 2, 2, 4], [4, 1, np.nan, 3], [2, 4, 1, 3]])
    p = pd.DataFrame([[2, 1, 1, 4], [4, 3, 2, 1], [1, 3, 4, 2]])
    values = daily_rank_correlations(y, p)
    expected = [
        spearmanr(a[np.isfinite(a)], b[np.isfinite(a)]).statistic
        for a, b in zip(y.to_numpy(), p.to_numpy(), strict=True)
    ]
    np.testing.assert_allclose(values, expected, atol=1e-15)
    assert correlation_sharpe(values) == pytest.approx(np.mean(expected) / np.std(expected, ddof=0))


@pytest.mark.parametrize("fault", ["row_order", "column_order", "nonfinite", "too_few", "constant"])
def test_invalid_metric_inputs_fail(fault):
    y = pd.DataFrame([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    p = y.copy()
    if fault == "row_order":
        p = p.iloc[::-1]
    elif fault == "column_order":
        p = p[p.columns[::-1]]
    elif fault == "nonfinite":
        p.iloc[0, 0] = np.nan
    elif fault == "too_few":
        y.iloc[0, 1:] = np.nan
    else:
        p.iloc[0, :] = 1
    with pytest.raises((ValueError, ZeroDivisionError)):
        daily_rank_correlations(y, p)


def test_bootstrap_respects_every_fitted_fold_and_seed():
    indices = block_indices([7, 11], 3, 60, 42)
    np.testing.assert_array_equal(indices, block_indices([7, 11], 3, 60, 42))
    assert indices.shape == (60, 18)
    assert (indices[:, :7] < 7).all()
    assert ((indices[:, 7:] >= 7) & (indices[:, 7:] < 18)).all()


def test_paired_comparison_preserves_declared_populations():
    x = np.sin(np.arange(40) * 1.4) * 0.2 + 0.1
    scores = {"reference": x, "candidate": x + 0.02}
    rows = compare_predictions(
        scores,
        [17, 23],
        [("candidate", "reference")],
        {"bootstrap_block_dates": [2, 4], "bootstrap_repetitions": 60, "seed": 42},
    )
    assert len(rows) == 2
    for row in rows:
        assert row["delta"] == pytest.approx(correlation_sharpe(x + 0.02) - correlation_sharpe(x))
        assert (
            row["simultaneous_95_interval"][0] <= row["delta"] <= row["simultaneous_95_interval"][1]
        )


def test_checkpoint_reuse_requires_content_and_lineage(tmp_path):
    from commodity_prediction.runtime import atomic_json, seal_checkpoint, verify_checkpoint

    atomic_json(tmp_path / "result.json", {"value": 1})
    seal_checkpoint(tmp_path, "frozen-input", ["result.json"])
    assert verify_checkpoint(tmp_path, "frozen-input")
    with pytest.raises(ValueError, match="Stale"):
        verify_checkpoint(tmp_path, "other-input")
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify_checkpoint(tmp_path, "frozen-input")
