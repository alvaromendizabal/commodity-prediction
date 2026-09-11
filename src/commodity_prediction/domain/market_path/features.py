"""Four-date OHLC and trading-activity paths; no target labels or fitted statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = ("intraday", "overnight", "range", "close_location")
ACTIVITY = ("relative_volume", "volume_change")
VARIANTS = {
    "current_market": (PRICE + ACTIVITY, (0,)),
    "price_path": (PRICE, (0, 1, 2, 3)),
    "activity_path": (ACTIVITY, (0, 1, 2, 3)),
    "joint_path": (PRICE + ACTIVITY, (0, 1, 2, 3)),
}


def primitive_series(x: pd.DataFrame, assets: list[str]) -> dict[str, pd.DataFrame]:
    """Structural absence stays out of each frame; observed missing bars remain NaN."""
    dates = x.index.to_numpy()
    if (
        len(dates) < 2
        or not np.issubdtype(dates.dtype, np.integer)
        or not np.all(np.diff(dates) == 1)
    ):
        raise ValueError("Contiguous integer observation dates required")
    if len(set(assets)) != len(assets) or not set(assets).issubset(x.columns):
        raise ValueError("Asset metadata is incomplete or duplicated")
    output: dict[str, dict[str, pd.Series]] = {k: {} for k in PRICE + ACTIVITY}
    for asset in assets:
        close = np.log(x[asset].where(x[asset] > 0))
        us = asset.endswith("_adj_close")
        stem = asset.removesuffix("close" if us else "Close")
        names = [stem + k for k in (["open", "high", "low"] if us else ["Open", "High", "Low"])]
        if set(names).issubset(x.columns):
            o, h, low = [np.log(x[k].where(x[k] > 0)) for k in names]
            width = (h - low).where(h > low)
            output["intraday"][asset] = close - o
            output["overnight"][asset] = o - close.shift(1)
            output["range"][asset] = h - low
            output["close_location"][asset] = (2 * close - h - low) / width
        volume_name = stem + ("volume" if us else "Volume")
        if volume_name in x:
            v = np.log1p(x[volume_name].where(x[volume_name] >= 0))
            # Exclude current activity from its own baseline; trailing observed rows only.
            baseline = v.shift(1).rolling(21, min_periods=14).median()
            output["relative_volume"][asset] = v - baseline
            output["volume_change"][asset] = v.diff()
    return {
        k: pd.DataFrame(v, index=x.index).replace([np.inf, -np.inf], np.nan)
        for k, v in output.items()
    }


def path_block(
    x: pd.DataFrame, pairs: pd.DataFrame, variant: str
) -> tuple[np.ndarray, list[str], dict]:
    if variant not in VARIANTS or pairs.target.duplicated().any():
        raise ValueError("Undeclared variant or duplicate target")
    legs = [str(p).split(" - ") for p in pairs.pair]
    if any(len(p) not in (1, 2) or len(set(p)) != len(p) for p in legs):
        raise ValueError("Invalid target legs")
    assets = sorted({a for row in legs for a in row})
    primitive = primitive_series(x, assets)
    channels, lags = VARIANTS[variant]
    arrays, names, coverage = [], [], {}
    for channel in channels:
        frame = primitive[channel]
        supported = set(frame.columns)
        coverage[channel] = len(supported)
        # Static eligibility is identical for every lag; never forward-fill missing observations.
        applicable_count = np.asarray(
            [sum(a in supported for a in row) for row in legs], dtype=np.float32
        )
        names.append("market_path__" + channel + "_applicable_legs")
        arrays.append(np.broadcast_to(applicable_count, (len(x), len(pairs))))
        for lag in lags:
            shifted = frame.shift(lag)
            left = np.stack(
                [shifted[p[0]].to_numpy() if p[0] in supported else np.zeros(len(x)) for p in legs],
                axis=1,
            )
            right = np.stack(
                [
                    shifted[p[1]].to_numpy()
                    if len(p) == 2 and p[1] in supported
                    else np.zeros(len(x))
                    for p in legs
                ],
                axis=1,
            )
            for suffix, values in [("difference", left - right), ("sum", left + right)]:
                names.append(f"market_path__{channel}_lag_{lag}_{suffix}")
                arrays.append(values)
    values = np.stack(arrays, axis=2).astype(np.float32)
    if np.isinf(values).any() or len(set(names)) != len(names):
        raise ValueError("Invalid market path values")
    return (
        values,
        names,
        {
            "supported_assets_by_channel": coverage,
            "templates_added": len(names),
            "count_note": "Includes static applicability columns; duplicates are screened on training only.",
        },
    )


def build_panel(parent, x: pd.DataFrame, pairs: pd.DataFrame, variant: str):
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.compact.features import BASE, Variant
    from commodity_prediction.domain.compact.features import build_panel as compact_panel

    if parent.dates != x.index.tolist() or parent.targets != pairs.target.tolist():
        raise ValueError("Frozen feature panel and raw inputs are not aligned")
    baseline = compact_panel(parent, pairs, Variant(variant, (*BASE, "tail_risk"), admit=True))
    values, names, coverage = path_block(x, pairs, variant)
    panel = Panel(
        np.concatenate([baseline.values, values], axis=2),
        baseline.dates,
        baseline.targets,
        baseline.names + names,
        dict(baseline.source_series),
    )
    panel.validate()
    return panel, coverage
