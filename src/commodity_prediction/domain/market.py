"""Causal market-path, activity, risk, and asynchronous-observation features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import Builder


def roll(frame: pd.DataFrame, window: int):
    return frame.rolling(window, min_periods=max(3, window * 2 // 3))


def nonzero(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.where(frame.abs() > 1e-12)


def market_features(b: Builder, x: pd.DataFrame, logp: pd.DataFrame) -> None:
    r = logp.diff()
    vol = nonzero(roll(r, 63).std(ddof=0))
    for lag in [0, 1, 2, 5]:
        b.asset("reference", f"return_lag_{lag}", r.shift(lag))
        b.asset("reference", f"risk_scaled_return_lag_{lag}", r.shift(lag) / vol)
    b.asset("reference", "risk_63", vol)
    for lo, hi in [(1, 5), (5, 21), (21, 63), (63, 126), (126, 252)]:
        b.asset(
            "trend_shape",
            f"nonoverlap_{lo}_{hi}",
            (logp.shift(lo) - logp.shift(hi)) / (vol * np.sqrt(hi - lo)),
        )
    for window in [5, 21, 63, 126]:
        minimum = max(3, window * 2 // 3)
        path = r.abs().rolling(window, min_periods=minimum).sum()
        b.asset("trend_shape", f"efficiency_{window}", logp.diff(window) / nonzero(path))
        fast = logp.ewm(span=max(2, window // 3), adjust=False, min_periods=minimum).mean()
        slow = logp.ewm(span=window, adjust=False, min_periods=minimum).mean()
        b.asset("trend_shape", f"ema_separation_{window}", (fast - slow) / vol)
        b.asset(
            "trend_shape",
            f"peer_momentum_rank_{window}",
            logp.diff(window).rank(axis=1, pct=True) - 0.5,
        )
        b.asset(
            "trend_shape",
            f"peer_risk_scaled_momentum_rank_{window}",
            (logp.diff(window) / (vol * np.sqrt(window))).rank(axis=1, pct=True) - 0.5,
        )
    # The denominator is lagged so a current jump cannot inflate its own threshold.
    shock = r / vol.shift(1)
    b.asset(
        "tail_risk",
        "robust_shock",
        (r - roll(r, 63).median())
        / nonzero(roll(r, 63).quantile(0.75) - roll(r, 63).quantile(0.25)),
    )
    for window in [21, 63]:
        down = roll(r.clip(upper=0).pow(2), window).mean()
        up = roll(r.clip(lower=0).pow(2), window).mean()
        b.asset("tail_risk", f"semivariance_balance_{window}", (up - down) / nonzero(up + down))
        b.asset(
            "tail_risk",
            f"jump_frequency_{window}",
            roll((shock.abs() > 3).where(shock.notna()).astype(float), window).mean(),
        )
        b.asset(
            "tail_risk",
            f"jump_pressure_{window}",
            roll(shock.where(shock.abs() > 3, 0).where(shock.notna()), window).mean(),
        )
        b.asset("tail_risk", f"left_quantile_{window}", roll(r, window).quantile(0.1) / vol)
        b.asset("tail_risk", f"right_quantile_{window}", roll(r, window).quantile(0.9) / vol)
        b.asset("tail_risk", f"drawdown_{window}", (logp - roll(logp, window).max()) / vol)
        b.asset("tail_risk", f"drawup_{window}", (logp - roll(logp, window).min()) / vol)
    positions = pd.DataFrame(
        np.broadcast_to(np.arange(len(x))[:, None], logp.shape), index=x.index, columns=logp.columns
    )
    last_seen = positions.where(logp.notna()).ffill()
    age = positions - last_seen
    gap = positions - last_seen.shift()
    observed_move = (logp - logp.ffill().shift()).where(logp.notna())
    b.asset("asynchrony", "observation_age", age)
    b.asset("asynchrony", "gap_to_previous_observation", gap.where(logp.notna()))
    b.asset("asynchrony", "resumed_after_gap", ((gap > 1) & logp.notna()).astype(float))
    b.asset("asynchrony", "observed_move_per_date", observed_move / nonzero(gap))
    b.asset("asynchrony", "observed_move_per_risk", observed_move / (vol * np.sqrt(gap)))
    b.asset(
        "asynchrony",
        "zero_return_fraction",
        roll((r == 0).where(r.notna()).astype(float), 21).mean(),
    )
    # Forward filling above is confined to explicit observation-age features.
    # Ordinary market returns and price-path features never fill missing prices.
    activity: dict[str, dict[str, pd.Series]] = {}
    path_features: dict[str, dict[str, pd.Series]] = {}
    for asset in logp:
        us = asset.endswith("_adj_close")
        stem = asset.removesuffix("close") if us else asset.removesuffix("Close")
        volume_name = stem + ("volume" if us else "Volume")
        if volume_name in x:
            v = x[volume_name].where(x[volume_name] >= 0)
            rel = v / v.rolling(21, min_periods=14).median().replace(0, np.nan)
            amount = (x[asset] * v).where(v > 0)
            impact = r[asset].abs() / amount
            values = {
                "relative_volume": np.log1p(rel),
                "signed_volume_pressure": np.sign(r[asset]) * np.log1p(rel),
                "normalized_activity_impact": impact
                / impact.rolling(63, min_periods=42).median().replace(0, np.nan),
                "volume_acceleration": np.log1p(v).diff().diff(),
            }
            for window in [5, 21]:
                valid_volume = v.where(r[asset].notna())
                weighted = (r[asset] * valid_volume).rolling(
                    window, min_periods=max(3, window * 2 // 3)
                ).sum() / valid_volume.rolling(
                    window, min_periods=max(3, window * 2 // 3)
                ).sum().replace(0, np.nan)
                values[f"volume_weighted_return_{window}"] = weighted / vol[asset]
            oi_name = stem + "open_interest"
            if oi_name in x:
                oi = x[oi_name].where(x[oi_name] > 0)
                values["volume_to_open_interest"] = v / oi
                values["position_price_pressure"] = np.sign(r[asset]) * np.log(oi).diff()
            for name, value in values.items():
                activity.setdefault(name, {})[asset] = value
        columns = [stem + z for z in (["open", "high", "low"] if us else ["Open", "High", "Low"])]
        if not set(columns).issubset(x):
            continue
        o, h, low = [np.log(x[c].where(x[c] > 0)) for c in columns]
        close = logp[asset]
        intra, overnight = close - o, o - close.shift()
        range_sq = (h - low).pow(2) / (4 * np.log(2))
        rs = ((h - o) * (h - close) + (low - o) * (low - close)).clip(lower=0)
        values = {
            "intraday": intra / vol[asset],
            "overnight": overnight / vol[asset],
            "gap_reversal": (intra * overnight) / vol[asset].pow(2),
        }
        for window in [5, 21]:
            minimum = max(3, window * 2 // 3)
            values[f"intraday_trend_{window}"] = (
                intra.rolling(window, min_periods=minimum).mean() / vol[asset]
            )
            values[f"overnight_trend_{window}"] = (
                overnight.rolling(window, min_periods=minimum).mean() / vol[asset]
            )
            values[f"range_risk_ratio_{window}"] = (
                np.sqrt(range_sq.rolling(window, min_periods=minimum).mean()) / vol[asset]
            )
            values[f"path_risk_ratio_{window}"] = (
                np.sqrt(rs.rolling(window, min_periods=minimum).mean()) / vol[asset]
            )
        for name, value in values.items():
            path_features.setdefault(name, {})[asset] = value
    for family, values_by_name in [
        ("trading_activity", activity),
        ("intraday_path", path_features),
    ]:
        for name, values in values_by_name.items():
            b.asset(family, name, pd.DataFrame(values, index=x.index))


def latent_features(b: Builder, logp: pd.DataFrame) -> None:
    """Trailing PCA refit every 21 dates using data strictly before the current date."""
    returns = logp.diff().to_numpy(dtype=float)
    residual = np.full_like(returns, np.nan)
    systematic = np.full_like(returns, np.nan)
    loading = np.full_like(returns, np.nan)
    concentration = np.full(len(logp), np.nan)
    means = scales = vectors = None
    fraction = np.nan
    for t in range(126, len(logp)):
        if (t - 126) % 21 == 0:
            history = returns[t - 126 : t]
            finite = np.isfinite(history)
            count = finite.sum(axis=0)
            means = np.divide(
                np.where(finite, history, 0).sum(axis=0),
                count,
                out=np.zeros(history.shape[1]),
                where=count > 0,
            )
            filled = np.where(finite, history, means)
            scales = filled.std(axis=0)
            scales = np.where(scales > 1e-9, scales, 1.0)
            z = (filled - means) / scales
            z[:, count < 84] = 0
            covariance = z.T @ z / len(z)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            vectors = eigenvectors[:, -3:]
            fraction = float(eigenvalues[-3:].sum() / max(eigenvalues.sum(), 1e-12))
        assert means is not None and scales is not None and vectors is not None
        current = np.where(np.isfinite(returns[t]), returns[t], means)
        z = (current - means) / scales
        common = vectors @ (vectors.T @ z)
        residual[t] = (z - common) * scales
        systematic[t] = common * scales
        residual[t, ~np.isfinite(returns[t])] = np.nan
        systematic[t, ~np.isfinite(returns[t])] = np.nan
        loading[t] = np.sum(vectors**2, axis=1)
        concentration[t] = fraction
    residual_frame = pd.DataFrame(residual, index=logp.index, columns=logp.columns)
    for window in [1, 5, 21]:
        value = residual_frame if window == 1 else roll(residual_frame, window).mean()
        b.asset("latent_factors", f"idiosyncratic_{window}", value)
    b.asset(
        "latent_factors",
        "systematic_return",
        pd.DataFrame(systematic, index=logp.index, columns=logp.columns),
    )
    b.asset(
        "latent_factors",
        "loading_concentration",
        pd.DataFrame(loading, index=logp.index, columns=logp.columns),
    )
    b.context("latent_factors", "market_concentration", pd.Series(concentration, index=logp.index))
