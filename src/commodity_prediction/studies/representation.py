"""Point-in-time feature extensions and metadata-directed candidate routing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from commodity_prediction.features import price_columns


@dataclass(frozen=True)
class FeatureInfo:
    family: str
    assets: tuple[str, ...] = ()
    pair: str = ""
    target: str = ""
    context: bool = False


def asset_stem(asset: str) -> str:
    for suffix in ["adj_close", "Close"]:
        if asset.endswith(suffix):
            return asset.removesuffix(suffix)
    return asset


def existing_metadata(
    columns: list[str], prices: list[str], pairs: pd.DataFrame
) -> dict[str, FeatureInfo]:
    unique_pairs = pairs.pair.drop_duplicates().tolist()
    metadata = {}
    for name in columns:
        family, body = name.split("__", 1)
        if family == "pairs":
            match = re.match(r"pair_(\d+)_", body)
            if match is None:
                raise ValueError(f"Unrecognized pair feature: {name}")
            pair = unique_pairs[int(match.group(1))]
            info = FeatureInfo(family, tuple(pair.split(" - ")), pair=pair)
        else:
            assets = tuple(a for a in prices if body.startswith(asset_stem(a)))
            info = FeatureInfo(family, assets, context=not assets and family == "cross_market")
            if not assets and not info.context:
                raise ValueError(f"Unmapped candidate: {name}")
        metadata[name] = info
    return metadata


def validate_time_index(frame: pd.DataFrame) -> None:
    index = frame.index.to_numpy()
    if (
        not np.issubdtype(index.dtype, np.integer)
        or len(index) < 2
        or not np.all(np.diff(index) == 1)
    ):
        raise ValueError("Feature history requires contiguous integer date IDs")


def release_history(y: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Expose labels only on their h+1 release date, including during validation.

    The model remains frozen in a validation fold. Previously released labels are
    nevertheless available inputs, matching the official streaming data contract.
    This is chronological history, not random-fold or full-sample target encoding.
    """
    validate_time_index(y)
    if set(pairs.target) != set(y.columns) or pairs.target.duplicated().any():
        raise ValueError("Historical-label metadata must cover each target once")
    if not pairs.lag.isin([1, 2, 3, 4]).all():
        raise ValueError("Unsupported release horizon")
    output = {}
    for target, lag, _ in pairs.itertuples(index=False, name=None):
        known = y[target].replace(-999999, np.nan).shift(int(lag) + 1)
        prefix = f"release_history__{target}__"
        output[prefix + "latest"] = known
        output[prefix + "expanding_mean"] = known.expanding(min_periods=40).mean()
        output[prefix + "ewm_126"] = known.ewm(span=126, min_periods=40, adjust=False).mean()
        for window in [21, 63]:
            roll = known.rolling(window, min_periods=window * 2 // 3)
            output[prefix + f"mean_{window}"] = roll.mean()
            output[prefix + f"vol_{window}"] = roll.std(ddof=0)
        output[prefix + "availability_21"] = known.notna().rolling(21, min_periods=21).mean()
    return pd.DataFrame(output, index=y.index, dtype="float32")


def extend_candidates(
    x: pd.DataFrame, y: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, FeatureInfo]]:
    validate_time_index(x)
    if not x.index.equals(y.index):
        raise ValueError("Market and released-label histories must align")
    prices = price_columns(x)
    logp = np.log(x[prices].where(x[prices] > 0))
    returns = logp.diff()
    output: dict[str, pd.Series] = {}
    metadata: dict[str, FeatureInfo] = {}

    def add(family: str, asset: str, suffix: str, values: pd.Series) -> None:
        name = f"{family}__{asset}_{suffix}"
        output[name] = values.replace([np.inf, -np.inf], np.nan).astype("float32")
        metadata[name] = FeatureInfo(family, (asset,))

    for asset in prices:
        p, r = logp[asset], returns[asset]
        vol21 = r.rolling(21, min_periods=15).std(ddof=0).replace(0, np.nan)
        vol126 = r.rolling(126, min_periods=84).std(ddof=0).replace(0, np.nan)
        risk_state = vol21 / vol126
        for window in [126, 252]:
            add("regime", asset, f"return_{window}", p.diff(window))
        add("regime", asset, "vol_126", vol126)
        add("regime", asset, "risk_state", risk_state)
        add("regime", asset, "skew_63", r.rolling(63, min_periods=42).skew())
        add("regime", asset, "kurtosis_63", r.rolling(63, min_periods=42).kurt())
        add("regime", asset, "sign_persistence_21", np.sign(r).rolling(21, min_periods=15).mean())
        add("regime", asset, "autocorrelation_63", r.rolling(63, min_periods=42).corr(r.shift()))
        add("regime", asset, "drawdown_126", p - p.rolling(126, min_periods=84).max())
        add("regime", asset, "return_percentile_126", r.rolling(126, min_periods=84).rank(pct=True))
        for window in [1, 5, 21]:
            add(
                "interactions",
                asset,
                f"risk_scaled_return_{window}",
                p.diff(window) / (vol21 * np.sqrt(window)),
            )
        add("interactions", asset, "momentum_by_risk_state", p.diff(21) * risk_state)
        recent = p.rolling(21, min_periods=15)
        z = (p - recent.mean()) / recent.std(ddof=0).replace(0, np.nan)
        add("interactions", asset, "short_momentum_by_reversion", p.diff(5) * np.tanh(z))
        market = asset.split("_")[0]
        peers = [c for c in prices if c.startswith(market + "_")]
        relative = r - returns[peers].median(axis=1)
        add("interactions", asset, "relative_return_by_risk_state", relative * risk_state)
    historical = release_history(y, pairs)
    for row in pairs.itertuples(index=False):
        for name in historical:
            if name.startswith(f"release_history__{row.target}__"):
                metadata[name] = FeatureInfo(
                    "release_history", tuple(row.pair.split(" - ")), target=row.target
                )
    extended = pd.concat([pd.DataFrame(output, index=x.index), historical], axis=1)
    if set(extended.columns) != set(metadata):
        raise ValueError("Extended feature metadata is incomplete")
    return extended, metadata


def routed_columns(
    metadata: dict[str, FeatureInfo], assets: tuple[str, ...], pair: str = "", target: str = ""
) -> list[str]:
    """Keep own instruments, the exact pair, market context, and own released history."""
    selected = []
    for name, info in metadata.items():
        if info.target:
            keep = bool(target) and info.target == target
        elif info.pair:
            keep = bool(pair) and info.pair == pair
        else:
            keep = info.context or bool(set(info.assets) & set(assets))
        if keep:
            selected.append(name)
    return selected


def asset_labels(x: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Supervision only. At t, an h-horizon label needs data through t+h+1."""
    validate_time_index(x)
    logp = np.log(x[price_columns(x)].where(x[price_columns(x)] > 0))
    frames = []
    for horizon in horizons:
        if horizon not in [1, 2, 3, 4]:
            raise ValueError("Unsupported asset horizon")
        frame = logp.shift(-horizon - 1) - logp.shift(-1)
        frame.columns = [f"h{horizon}:{asset}" for asset in frame]
        frames.append(frame)
    return pd.concat(frames, axis=1)


def metadata_to_dict(metadata: dict[str, FeatureInfo]) -> dict:
    return {name: asdict(info) for name, info in metadata.items()}


def metadata_from_dict(value: dict) -> dict[str, FeatureInfo]:
    return {
        name: FeatureInfo(**{**info, "assets": tuple(info["assets"])})
        for name, info in value.items()
    }
