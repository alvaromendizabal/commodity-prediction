"""Leakage, missingness, symmetry, scale, and ablation contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commodity_prediction.domain.market_normalization.features import asset_channels, feature_block


def market():
    rng = np.random.default_rng(20260911)
    n = 90
    x = pd.DataFrame(index=pd.Index(range(n), name="date_id"))
    for stem in ["US_A_adj_", "US_B_adj_"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        opening = close * np.exp(rng.normal(0, 0.01, n))
        x[stem + "close"] = close
        x[stem + "open"] = opening
        x[stem + "high"] = np.maximum(opening, close) * 1.01
        x[stem + "low"] = np.minimum(opening, close) / 1.01
        x[stem + "volume"] = rng.lognormal(8, 0.3, n)
    x["FX_AB"] = np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    pairs = pd.DataFrame(
        {
            "target": ["target_0", "target_1", "target_2", "target_3"],
            "pair": [
                "US_A_adj_close - US_B_adj_close",
                "US_B_adj_close - US_A_adj_close",
                "US_A_adj_close",
                "FX_AB",
            ],
            "lag": [1, 1, 2, 1],
        }
    )
    return x, pairs


@pytest.mark.parametrize(
    "variant,count", [("normalized_price", 7), ("volume_confirmation", 7), ("normalized_joint", 14)]
)
def test_declared_size_symmetry_and_structural_absence(variant, count):
    x, pairs = market()
    values, names, coverage = feature_block(x, pairs, variant)
    assert values.shape == (90, 4, count)
    assert coverage["added_templates"] == count
    assert not np.isinf(values).any()
    for i, name in enumerate(names):
        if name.endswith("difference"):
            np.testing.assert_allclose(values[:, 0, i], -values[:, 1, i], equal_nan=True)
        elif name.endswith("sum"):
            np.testing.assert_allclose(values[:, 0, i], values[:, 1, i], equal_nan=True)
    np.testing.assert_array_equal(values[:, 3], 0)


def test_future_perturbation_cannot_change_past_features():
    x, pairs = market()
    changed = x.copy()
    changed.iloc[61:] *= 17
    first = feature_block(x, pairs, "normalized_joint")[0]
    second = feature_block(changed, pairs, "normalized_joint")[0]
    np.testing.assert_array_equal(first[:61], second[:61])


def test_price_units_do_not_change_features():
    x, pairs = market()
    changed = x.copy()
    for c in changed.columns:
        if c.endswith(("close", "open", "high", "low")):
            changed[c] *= 1000
    np.testing.assert_allclose(
        feature_block(x, pairs, "normalized_joint")[0],
        feature_block(changed, pairs, "normalized_joint")[0],
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )


def test_observed_missing_bar_is_not_structural_zero():
    x, pairs = market()
    x.loc[50, "US_A_adj_open"] = np.nan
    values, names, _ = feature_block(x, pairs, "normalized_price")
    assert np.isnan(values[50, 2, names.index("market_normalization__intraday_difference")])
    assert values[50, 2, 0] == 1


def test_current_return_does_not_enter_its_own_volatility_denominator():
    x, _ = market()
    channels = asset_channels(x, ["US_A_adj_close"])
    expected = np.log(x.US_A_adj_close / x.US_A_adj_open)
    scale = (
        np.log(x.US_A_adj_close)
        .diff()
        .shift(1)
        .rolling(21, min_periods=14)
        .std(ddof=0)
        .clip(lower=1e-6)
    )
    np.testing.assert_allclose(
        channels["intraday"]["US_A_adj_close"],
        (expected / scale).clip(-12, 12),
        atol=1e-10,
        equal_nan=True,
    )


def test_current_volume_excluded_from_volume_reference():
    x, _ = market()
    c = asset_channels(x, ["US_A_adj_close"])
    lv = np.log1p(x.US_A_adj_volume)
    h = lv.shift(1).rolling(21, min_periods=14)
    s = np.tanh(((lv - h.median()) / h.std(ddof=0).clip(lower=1e-6)).clip(-12, 12) / 3)
    np.testing.assert_allclose(
        c["direction_volume"]["US_A_adj_close"], c["intraday"]["US_A_adj_close"] * s, equal_nan=True
    )


def test_joint_is_exact_union_not_different_base():
    x, p = market()
    a, na, _ = feature_block(x, p, "normalized_price")
    b, nb, _ = feature_block(x, p, "volume_confirmation")
    joint, names, _ = feature_block(x, p, "normalized_joint")
    assert names == na + nb
    np.testing.assert_array_equal(joint, np.concatenate([a, b], axis=2))


def test_bad_dates_and_duplicate_targets_fail_closed():
    x, pairs = market()
    with pytest.raises(ValueError, match="Contiguous"):
        feature_block(x.drop(index=20), pairs, "normalized_price")
    pairs.loc[1, "target"] = "target_0"
    with pytest.raises(ValueError, match="duplicated"):
        feature_block(x, pairs, "normalized_price")


def test_invalid_ohlc_and_negative_volume_remain_missing():
    x, pairs = market()
    x.loc[50, "US_A_adj_high"] = 1
    x.loc[51, "US_A_adj_volume"] = -1
    a, names, _ = feature_block(x, pairs, "normalized_joint")
    assert np.isnan(a[50, 2, names.index("market_normalization__intraday_difference")])
    assert np.isnan(a[51, 2, names.index("market_normalization__direction_volume_difference")])


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_nonfinite_volume_cannot_become_clipped_confirmation(bad):
    x, _ = market()
    x.loc[50, "US_A_adj_volume"] = bad
    channels = asset_channels(x, ["US_A_adj_close"])
    for channel in ["direction_volume", "location_volume", "range_volume"]:
        assert np.isnan(channels[channel].loc[50, "US_A_adj_close"])
    assert np.isfinite(channels["intraday"].loc[50, "US_A_adj_close"])


@pytest.mark.parametrize("field", ["close", "open", "high", "low"])
def test_infinite_price_is_missing_before_any_clipping(field):
    x, _ = market()
    x.loc[50, "US_A_adj_" + field] = np.inf
    channels = asset_channels(x, ["US_A_adj_close"])
    for frame in channels.values():
        assert np.isnan(frame.loc[50, "US_A_adj_close"])


def test_invalid_previous_close_cannot_become_extreme_overnight_signal():
    x, _ = market()
    x.loc[49, "US_A_adj_close"] = np.inf
    channels = asset_channels(x, ["US_A_adj_close"])
    assert np.isnan(channels["overnight"].loc[50, "US_A_adj_close"])


def test_duplicate_input_columns_are_rejected():
    x, pairs = market()
    ambiguous = pd.concat([x, x[["US_A_adj_volume"]]], axis=1)
    with pytest.raises(ValueError, match="unique input columns"):
        feature_block(ambiguous, pairs, "normalized_joint")


@pytest.mark.parametrize("column", ["target", "pair", "lag"])
def test_missing_target_metadata_is_rejected(column):
    x, pairs = market()
    with pytest.raises(ValueError, match="Target metadata"):
        feature_block(x, pairs.drop(columns=column), "normalized_price")


def test_empty_target_metadata_is_rejected():
    x, pairs = market()
    with pytest.raises(ValueError, match="Target metadata"):
        feature_block(x, pairs.iloc[:0], "normalized_price")


@pytest.mark.parametrize("lag", [0, 5, 1.5, np.nan])
def test_unsupported_target_horizons_are_rejected(lag):
    x, pairs = market()
    pairs["lag"] = pairs["lag"].astype(float)
    pairs.loc[0, "lag"] = lag
    with pytest.raises(ValueError, match="Target metadata"):
        feature_block(x, pairs, "normalized_price")


def test_streaming_prefix_matches_full_history_at_each_origin():
    x, pairs = market()
    full, _, _ = feature_block(x, pairs, "normalized_joint")
    for stop in [30, 51, 62, 90]:
        prefix, _, _ = feature_block(x.iloc[:stop], pairs, "normalized_joint")
        np.testing.assert_array_equal(prefix[-1], full[stop - 1])


def test_partial_pair_applicability_does_not_fill_observed_missingness():
    x, pairs = market()
    pairs = pairs.iloc[:1].copy()
    pairs.loc[0, "pair"] = "US_A_adj_close - FX_AB"
    x.loc[50, "US_A_adj_open"] = np.nan
    values, names, _ = feature_block(x, pairs, "normalized_joint")
    assert values[50, 0, names.index("market_normalization__price_applicable_legs")] == 1
    assert np.isnan(values[50, 0, names.index("market_normalization__intraday_difference")])
