from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.competitive.feature_forecast import (
    fit_scale,
    mean_rank_prior,
    regularized_kelly_rank_prior,
    released_label_signal,
    targets_from_forecast_path,
)


def test_target_conversion_matches_known_formula() -> None:
    pairs = pd.DataFrame(
        {
            "target": ["target_0", "target_1"],
            "lag": [1, 4],
            "pair": ["A", "A - B"],
        }
    )
    path = np.array(
        [
            [100.0, 50.0],
            [110.0, 51.0],
            [120.0, 52.0],
            [130.0, 53.0],
            [140.0, 54.0],
        ]
    )
    got = targets_from_forecast_path(path, pairs, ["A", "B"])
    expected_0 = np.log(110.0) - np.log(100.0)
    expected_1 = (np.log(140.0) - np.log(100.0)) - (np.log(54.0) - np.log(50.0))
    np.testing.assert_allclose(got, [expected_0, expected_1])


def test_released_signal_respects_horizon_delay() -> None:
    y = pd.DataFrame({"target_0": np.arange(12, dtype=float), "target_1": np.arange(100, 112, dtype=float)})
    pairs = pd.DataFrame({"target": ["target_0", "target_1"], "lag": [1, 4], "pair": ["A", "B"]})
    pred_positions = np.array([10])
    got = released_label_signal(y, pairs, pred_positions, trailing=2)
    # lag=1 => delay 2 => latest available target index 8; mean [7,8]
    assert got.iloc[0, 0] == 7.5
    # lag=4 => delay 5 => latest available target index 5; mean [4,5]
    assert got.iloc[0, 1] == 104.5


def test_scalers_are_finite_and_round_trip_positive_data() -> None:
    x = np.array([[1.0, 10.0], [2.0, np.nan], [4.0, 40.0], [8.0, 80.0]])
    for space in ["raw", "log"]:
        scale = fit_scale(x, space)
        z = scale.transform(x)
        assert np.isfinite(z).all()
        restored = scale.inverse(z)
        assert np.isfinite(restored).all()
        assert (restored > 0).all()


def test_rank_priors_have_correct_schema_and_finite_values() -> None:
    y = pd.DataFrame(
        {
            "target_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target_1": [3.0, 2.0, 1.0, 4.0, 6.0],
            "target_2": [2.0, 1.0, 4.0, 3.0, 7.0],
        }
    )
    index = pd.Index([20, 21])
    for maker in [mean_rank_prior, regularized_kelly_rank_prior]:
        pred = maker(y, index)
        assert pred.shape == (2, 3)
        assert pred.index.equals(index)
        assert pred.columns.equals(y.columns)
        assert np.isfinite(pred.to_numpy()).all()
