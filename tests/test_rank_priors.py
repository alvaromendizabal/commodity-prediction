"""Leakage, metric-geometry, and schema tests for rank priors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commodity_prediction.domain.rank_prior.features import (
    constant_prediction,
    diagonal_rank_scores,
    ledoit_rank_scores,
    mean_rank_scores,
    raw_mean_scores,
    standardized_daily_ranks,
)


def frame(rows: int = 180, columns: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    values = rng.normal(size=(rows, columns))
    values += np.linspace(-0.4, 0.4, columns)[None, :]
    return pd.DataFrame(values, columns=[f"target_{i}" for i in range(columns)])


def test_standardized_daily_ranks_are_row_unit_norm_and_missing_neutral():
    y = frame()
    y.iloc[5, 2] = -999999
    ranks = standardized_daily_ranks(y)
    assert ranks.iloc[5, 2] == 0
    np.testing.assert_allclose(np.linalg.norm(ranks.to_numpy(), axis=1), 1.0, atol=1e-12)


def test_daily_rank_geometry_is_invariant_to_monotone_row_transform():
    y = frame()
    transformed = np.exp(y / 4)
    pd.testing.assert_frame_equal(
        standardized_daily_ranks(y), standardized_daily_ranks(transformed)
    )


def test_training_prefix_is_unchanged_by_future_perturbation():
    y = frame()
    changed = y.copy()
    changed.iloc[120:] = changed.iloc[120:] * -500 + 1000
    for method in [raw_mean_scores, mean_rank_scores, diagonal_rank_scores, ledoit_rank_scores]:
        pd.testing.assert_series_equal(method(y.iloc[:120]), method(changed.iloc[:120]))


def test_rank_prior_methods_return_finite_nondegenerate_target_scores():
    y = frame()
    for method in [raw_mean_scores, mean_rank_scores, diagonal_rank_scores, ledoit_rank_scores]:
        scores = method(y)
        assert scores.index.equals(y.columns)
        assert np.isfinite(scores.to_numpy()).all()
        assert np.ptp(scores.to_numpy()) > 0


def test_mean_rank_prior_recovers_persistent_target_ordering():
    rng = np.random.default_rng(3)
    bias = np.linspace(-2, 2, 30)
    y = pd.DataFrame(
        rng.normal(scale=0.4, size=(250, 30)) + bias[None, :],
        columns=[f"target_{i}" for i in range(30)],
    )
    scores = mean_rank_scores(y)
    assert np.corrcoef(scores.to_numpy(), bias)[0, 1] > 0.95


def test_constant_prediction_reorders_scores_and_rejects_schema_mismatch():
    y = frame(columns=8)
    scores = mean_rank_scores(y)
    columns = pd.Index(list(reversed(y.columns)))
    prediction = constant_prediction(pd.Index([10, 11]), columns, scores)
    assert prediction.columns.equals(columns)
    np.testing.assert_array_equal(prediction.iloc[0].to_numpy(), scores.reindex(columns).to_numpy())
    with pytest.raises(ValueError, match="schema"):
        constant_prediction(pd.Index([1]), pd.Index(["unknown"]), scores)


def test_degenerate_training_targets_fail_closed():
    y = pd.DataFrame(np.ones((20, 4)), columns=[f"target_{i}" for i in range(4)])
    with pytest.raises(ValueError):
        standardized_daily_ranks(y)
