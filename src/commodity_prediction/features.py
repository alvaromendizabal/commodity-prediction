"""Causal, domain-motivated candidate generation and training-only screening."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


def price_columns(x: pd.DataFrame) -> list[str]:
    return [c for c in x if c.startswith("FX_") or c.endswith(("_Close", "_adj_close"))]


def build_candidates(x: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Only market observations at or before row t are read; no labels are accepted.

    Missing market observations remain missing. There is no backward filling, global
    normalization, learned transform, or centered window. Daily market data is
    available in the official API before the prediction for that row.
    """
    if not x.index.is_unique or not x.index.is_monotonic_increasing:
        raise ValueError("Features require sorted unique dates")
    out: dict[str, pd.Series] = {}
    families: dict[str, str] = {}

    def add(family: str, name: str, value: pd.Series) -> None:
        key = family + "__" + name
        if key in out:
            raise ValueError(f"Duplicate feature name: {key}")
        out[key] = value.replace([np.inf, -np.inf], np.nan).astype("float32")
        families[key] = family

    prices = price_columns(x)
    logp = np.log(x[prices].where(x[prices] > 0))
    returns = logp.diff()
    for col in prices:
        s, r = logp[col], returns[col]
        add("reference", col + "_return_1", r)
        for lag in [2, 3, 4, 5, 10, 21, 63]:
            add("momentum", f"{col}_return_{lag}", s.diff(lag))
        for lag in [1, 2, 5]:
            add("momentum", f"{col}_return_1_lag_{lag}", r.shift(lag))
        add("momentum", col + "_acceleration", r.diff())
        add("momentum", col + "_ewm_5", r.ewm(span=5, min_periods=5, adjust=False).mean())
        vol21 = r.rolling(21, min_periods=15).std(ddof=0)
        for window in [5, 21, 63]:
            roll = r.rolling(window, min_periods=max(3, window * 2 // 3))
            add("volatility", f"{col}_vol_{window}", roll.std(ddof=0))
            downside = (
                r.clip(upper=0)
                .pow(2)
                .rolling(window, min_periods=max(3, window * 2 // 3))
                .mean()
                .pow(0.5)
            )
            add("volatility", f"{col}_downside_{window}", downside)
            levels = s.rolling(window, min_periods=max(3, window * 2 // 3))
            add(
                "reversion",
                f"{col}_z_{window}",
                (s - levels.mean()) / levels.std(ddof=0).replace(0, np.nan),
            )
        add(
            "volatility",
            col + "_vol_ratio",
            r.rolling(5, min_periods=3).std(ddof=0) / vol21.replace(0, np.nan),
        )
        add(
            "reversion",
            col + "_risk_scaled_momentum",
            s.diff(21) / (vol21 * np.sqrt(21)).replace(0, np.nan),
        )
    positions = pd.Series(np.arange(len(x)), index=x.index)
    for col in x:
        missing = x[col].isna()
        add("missingness", col + "_missing", missing.astype(float))
        age = positions - positions.where(~missing).ffill()
        add("missingness", col + "_observation_age", age)
    for col in x:
        if col.lower().endswith(("volume", "open_interest")):
            v = np.log1p(x[col].where(x[col] >= 0))
            add("liquidity", col + "_log", v)
            add("liquidity", col + "_change", v.diff())
            for window in [5, 21]:
                roll = v.rolling(window, min_periods=max(3, window * 2 // 3))
                add(
                    "liquidity",
                    f"{col}_z_{window}",
                    (v - roll.mean()) / roll.std(ddof=0).replace(0, np.nan),
                )
    for col in prices:
        suffix = "close" if col.endswith("close") else "Close"
        base = col.removesuffix(suffix)
        names = [
            base + s
            for s in (["open", "high", "low"] if suffix == "close" else ["Open", "High", "Low"])
        ]
        if not set(names).issubset(x.columns):
            continue
        opening, high, low = [x[n] for n in names]
        close = x[col]
        add("ohlc", col + "_range", np.log(high.where(high > 0) / low.where(low > 0)))
        add("ohlc", col + "_body", (close - opening) / opening.replace(0, np.nan))
        add("ohlc", col + "_close_location", (close - low) / (high - low).replace(0, np.nan))
        add(
            "ohlc",
            col + "_overnight_gap",
            np.log(opening.where(opening > 0) / close.shift().where(close.shift() > 0)),
        )
    market_returns = {}
    for prefix in ["LME", "JPX", "US", "FX"]:
        cols = [c for c in prices if c.startswith(prefix + "_")]
        if not cols:
            continue
        daily = returns[cols]
        median = daily.median(axis=1)
        market_returns[prefix] = median
        add("cross_market", prefix + "_median", median)
        add("cross_market", prefix + "_dispersion", daily.std(axis=1, ddof=0))
        add(
            "cross_market",
            prefix + "_breadth",
            (daily > 0).sum(axis=1) / daily.notna().sum(axis=1).replace(0, np.nan),
        )
        add("cross_market", prefix + "_trend_5", median.rolling(5, min_periods=3).mean())
        ranks = daily.rank(axis=1, pct=True)
        for col in cols:
            add("cross_market", col + "_peer_rank", ranks[col])
            add("cross_market", col + "_relative_return", daily[col] - median)
    for a, series_a in market_returns.items():
        for b, series_b in market_returns.items():
            if a < b:
                add("cross_market", a + "_minus_" + b, series_a - series_b)
    for number, pair in enumerate(pairs.pair.drop_duplicates()):
        cols = pair.split(" - ")
        spread = logp[cols[0]] if len(cols) == 1 else logp[cols[0]] - logp[cols[1]]
        for lag in [1, 2, 3, 4, 5, 21]:
            add("pairs", f"pair_{number}_return_{lag}", spread.diff(lag))
        roll = spread.rolling(21, min_periods=15)
        add(
            "pairs",
            f"pair_{number}_z_21",
            (spread - roll.mean()) / roll.std(ddof=0).replace(0, np.nan),
        )
        add("pairs", f"pair_{number}_vol_21", spread.diff().rolling(21, min_periods=15).std(ddof=0))
        if len(cols) == 2:
            a, b = returns[cols[0]], returns[cols[1]]
            add("pairs", f"pair_{number}_correlation_63", a.rolling(63, min_periods=42).corr(b))
            covariance = a.rolling(63, min_periods=42).cov(b, ddof=0)
            beta = covariance / b.rolling(63, min_periods=42).var(ddof=0).replace(0, np.nan)
            add("pairs", f"pair_{number}_hedged_return", a - beta * b)
    return pd.DataFrame(out, index=x.index), families


@dataclass
class Screen:
    columns: list[str]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series
    audit: pd.DataFrame

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not set(self.columns).issubset(x.columns):
            raise ValueError("Selected feature schema is missing columns")
        values = x[self.columns].fillna(self.medians)
        result = ((values - self.means) / self.scales).to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized features")
        return result


def screen_features(x_train: pd.DataFrame, y_train: pd.DataFrame, config: dict) -> Screen:
    """Supervised ranking and redundancy checks use only the fit interval.

    Screening relevance is mean absolute Pearson correlation across available
    targets, using pairwise nonmissing labels. The outer validation never enters.
    """
    if not x_train.index.equals(y_train.index):
        raise ValueError("Training dates differ")
    rejected: dict[str, str] = {}
    missing = x_train.isna().mean()
    for c in x_train:
        if missing[c] > config["max_missing_fraction"]:
            rejected[c] = "missingness"
        elif x_train[c].nunique(dropna=True) < 2:
            rejected[c] = "constant"
    usable = [c for c in x_train if c not in rejected]
    medians = x_train[usable].median()
    filled = x_train[usable].fillna(medians)
    seen: dict[bytes, str] = {}
    unique = []
    for c in usable:
        h = hashlib.sha256(filled[c].to_numpy(dtype="float64").tobytes()).digest()
        if h in seen:
            rejected[c] = "duplicate:" + seen[h]
        else:
            seen[h] = c
            unique.append(c)
    if not unique:
        raise ValueError("No usable training features")
    filled = filled[unique]
    means, scales = filled.mean(), filled.std(ddof=0)
    z = ((filled - means) / scales).to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    observed = np.isfinite(y)
    count = observed.sum(axis=0)
    sums = np.where(observed, y, 0).sum(axis=0)
    mean_y = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    centered_y = np.where(observed, y - mean_y, 0)
    mask = observed.astype(float)
    # Center each feature on exactly the observed dates of each target.
    sx = z.T @ mask
    sx2 = (z * z).T @ mask
    variance_x = sx2 - np.divide(sx * sx, count, out=np.zeros_like(sx), where=count > 0)
    denominator = np.sqrt(np.maximum(variance_x, 0) * (centered_y * centered_y).sum(axis=0))
    correlations = np.divide(
        z.T @ centered_y, denominator, out=np.zeros_like(denominator), where=denominator > 1e-12
    )
    valid_targets = count >= config["min_target_observations"]
    if not valid_targets.any():
        raise ValueError("Insufficient observed targets for screening")
    relevance = np.abs(correlations[:, valid_targets]).mean(axis=1)
    order = np.argsort(-relevance, kind="stable")
    selected: list[int] = []
    for idx in order:
        c = unique[idx]
        if len(selected) >= config["max_features"]:
            rejected[c] = "feature_budget"
        elif (
            selected
            and np.max(np.abs(z[:, selected].T @ z[:, idx] / len(z)))
            > config["max_abs_correlation"]
        ):
            rejected[c] = "correlated"
        else:
            selected.append(int(idx))
    columns = [unique[i] for i in selected]
    relevance_map = dict(zip(unique, relevance.tolist(), strict=True))
    audit = pd.DataFrame(
        [
            {
                "feature": c,
                "missing_fraction": float(missing[c]),
                "status": "retained" if c in columns else "rejected",
                "reason": rejected.get(c, "retained"),
                "training_relevance": relevance_map.get(c, 0.0),
            }
            for c in x_train
        ]
    )
    return Screen(columns, medians[columns], means[columns], scales[columns], audit)
