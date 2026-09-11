"""Point-in-time price normalization and unusual-volume confirmation.

No labels, fitted full-sample moments, fills, calendar joins, or raw-price levels.
The current bar is excluded from every rolling reference distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIANTS = {
    "normalized_price": ("price",),
    "volume_confirmation": ("confirmation",),
    "normalized_joint": ("price", "confirmation"),
}
WINDOW = 21
MIN_PERIODS = 14
FLOOR = 1e-6


def asset_channels(x: pd.DataFrame, assets: list[str]) -> dict[str, pd.DataFrame]:
    """Return three normalized price channels and three volume interactions."""
    if not x.columns.is_unique:
        raise ValueError("Market observations require unique input columns")
    dates = x.index.to_numpy()
    if (
        len(dates) < WINDOW + 2
        or not np.issubdtype(dates.dtype, np.integer)
        or not np.all(np.diff(dates) == 1)
        or len(set(assets)) != len(assets)
        or not set(assets).issubset(x.columns)
    ):
        raise ValueError("Contiguous dates and complete unique asset metadata required")
    names = (
        "intraday",
        "overnight",
        "range",
        "direction_volume",
        "location_volume",
        "range_volume",
    )
    output: dict[str, dict[str, pd.Series]] = {name: {} for name in names}
    for asset in assets:
        us = asset.endswith("_adj_close")
        stem = asset.removesuffix("close" if us else "Close")
        fields = [
            stem + name for name in (["open", "high", "low"] if us else ["Open", "High", "Low"])
        ]
        if not set(fields).issubset(x.columns):
            continue
        # Mask nonfinite raw observations before logs and clipping. Otherwise an
        # infinite bar can become a finite, saturated signal or contaminate lags.
        close = np.log(x[asset].where(np.isfinite(x[asset]) & (x[asset] > 0)))
        opening, high, low = [
            np.log(x[name].where(np.isfinite(x[name]) & (x[name] > 0))) for name in fields
        ]
        valid = (
            close.notna()
            & opening.notna()
            & high.notna()
            & low.notna()
            & (high >= low)
            & (high >= pd.concat([close, opening], axis=1).max(axis=1))
            & (low <= pd.concat([close, opening], axis=1).min(axis=1))
        )
        # The denominator at t uses close-to-close returns through t-1 only.
        scale = close.diff().shift(1).rolling(WINDOW, min_periods=MIN_PERIODS).std(ddof=0)
        scale = scale.clip(lower=FLOOR)
        intraday = ((close - opening) / scale).where(valid).clip(-12, 12)
        overnight = ((opening - close.shift(1)) / scale).where(valid).clip(-12, 12)
        width = (high - low).where(valid)
        scaled_range = (width / scale).clip(0, 12)
        location = ((2 * close - high - low) / width.where(width > 0)).clip(-1, 1)
        for name, series in [
            ("intraday", intraday),
            ("overnight", overnight),
            ("range", scaled_range),
        ]:
            output[name][asset] = series
        volume_name = stem + ("volume" if us else "Volume")
        if volume_name not in x:
            continue
        volume = x[volume_name]
        log_volume = np.log1p(volume.where(np.isfinite(volume) & (volume >= 0)))
        history = log_volume.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS)
        baseline = history.median()
        volume_scale = history.std(ddof=0).clip(lower=FLOOR)
        surprise = np.tanh(((log_volume - baseline) / volume_scale).clip(-12, 12) / 3)
        output["direction_volume"][asset] = intraday * surprise
        output["location_volume"][asset] = location * surprise
        output["range_volume"][asset] = scaled_range * surprise
    return {
        name: pd.DataFrame(values, index=x.index).replace([np.inf, -np.inf], np.nan)
        for name, values in output.items()
    }


def feature_block(
    x: pd.DataFrame, pairs: pd.DataFrame, variant: str
) -> tuple[np.ndarray, list[str], dict]:
    """Project each asset mechanism as directed-pair difference and common sum."""
    if (
        pairs.empty
        or not pairs.columns.is_unique
        or not {"target", "pair", "lag"}.issubset(pairs.columns)
        or pairs[["target", "pair", "lag"]].isna().any().any()
        or not pairs["lag"].isin([1, 2, 3, 4]).all()
    ):
        raise ValueError("Target metadata requires nonempty target/pair columns and horizons 1-4")
    if variant not in VARIANTS or pairs.target.duplicated().any():
        raise ValueError("Unknown variant or duplicated target")
    legs = [str(value).split(" - ") for value in pairs.pair]
    if any(len(row) not in (1, 2) or len(set(row)) != len(row) for row in legs):
        raise ValueError("Invalid target legs")
    assets = sorted({asset for row in legs for asset in row})
    channels = asset_channels(x, assets)
    groups = {
        "price": ("intraday", "overnight", "range"),
        "confirmation": ("direction_volume", "location_volume", "range_volume"),
    }
    arrays, names, coverage = [], [], {}
    for group in VARIANTS[variant]:
        members = groups[group]
        supported = set(channels[members[0]].columns)
        if any(set(channels[name].columns) != supported for name in members):
            raise ValueError("Structural support differs within a mechanism")
        coverage[group] = len(supported)
        eligible = np.asarray(
            [sum(asset in supported for asset in row) for row in legs], dtype=np.float32
        )
        arrays.append(np.broadcast_to(eligible, (len(x), len(pairs))))
        names.append(f"market_normalization__{group}_applicable_legs")
        for name in members:
            frame = channels[name]
            left = np.stack(
                [
                    frame[row[0]].to_numpy() if row[0] in supported else np.zeros(len(x))
                    for row in legs
                ],
                axis=1,
            )
            right = np.stack(
                [
                    frame[row[1]].to_numpy()
                    if len(row) == 2 and row[1] in supported
                    else np.zeros(len(x))
                    for row in legs
                ],
                axis=1,
            )
            for suffix, values in [("difference", left - right), ("sum", left + right)]:
                arrays.append(values)
                names.append(f"market_normalization__{name}_{suffix}")
    values = np.stack(arrays, axis=2).astype(np.float32)
    if np.isinf(values).any() or len(names) != len(set(names)):
        raise ValueError("Invalid normalized feature block")
    return values, names, {"supported_assets": coverage, "added_templates": len(names)}


def augment(base, x: pd.DataFrame, pairs: pd.DataFrame, variant: str):
    from commodity_prediction.domain.catalog import Panel

    if base.dates != x.index.tolist() or base.targets != pairs.target.tolist():
        raise ValueError("Frozen panel and normalized features are not aligned")
    block, names, coverage = feature_block(x, pairs, variant)
    if set(names).intersection(base.names):
        raise ValueError("Normalized feature family already exists in the control")
    panel = Panel(
        np.concatenate([base.values, block], axis=2),
        base.dates,
        base.targets,
        base.names + names,
        dict(base.source_series),
    )
    panel.validate()
    return panel, coverage
