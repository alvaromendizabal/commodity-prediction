"""Observed economic proxies, rolling factor exposures, FX graph, and pair dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import Builder
from .market import nonzero, roll

PROXIES = {
    "global_equity": "US_Stock_VT_adj_close",
    "gold": "US_Stock_GLD_adj_close",
    "energy": "US_Stock_XLE_adj_close",
    "materials": "US_Stock_XLB_adj_close",
    "treasury": "US_Stock_IEF_adj_close",
    "credit": "US_Stock_LQD_adj_close",
    "emerging_equity": "US_Stock_EEM_adj_close",
    "japan_equity": "US_Stock_EWJ_adj_close",
    "yen": "FX_USDJPY",
    "commodity_currency": "FX_AUDUSD",
}


def factor_features(b: Builder, x: pd.DataFrame, logp: pd.DataFrame) -> None:
    r = logp.diff()
    source = np.log(x.where(x > 0))
    for label, column in PROXIES.items():
        if column not in source:
            continue
        factor = source[column].diff()
        for lag in [0, 1, 2]:
            b.context("macro_links", f"{label}_lag_{lag}", factor.shift(lag))
        for window in [5, 21]:
            b.context(
                "macro_links",
                f"{label}_trend_{window}",
                factor.rolling(window, min_periods=max(3, window * 2 // 3)).sum(),
            )
        if label not in {
            "global_equity",
            "gold",
            "energy",
            "materials",
            "treasury",
            "yen",
            "commodity_currency",
        }:
            continue
        # Coefficients are frozen at t-1 before constructing current innovations.
        beta = (
            r.rolling(126, min_periods=84)
            .cov(factor, ddof=0)
            .div(factor.rolling(126, min_periods=84).var(ddof=0).replace(0, np.nan), axis=0)
            .shift()
        )
        beta = beta.clip(-10, 10)
        residual = r - beta.mul(factor, axis=0)
        b.asset("factor_relative", f"{label}_beta", beta)
        b.asset("factor_relative", f"{label}_innovation", residual)
        b.asset("factor_relative", f"{label}_residual_trend_21", roll(residual, 21).mean())
        lead_beta = (
            r.rolling(126, min_periods=84)
            .cov(factor.shift(), ddof=0)
            .div(factor.shift().rolling(126, min_periods=84).var(ddof=0).replace(0, np.nan), axis=0)
            .shift()
            .clip(-10, 10)
        )
        b.asset("factor_relative", f"{label}_lagged_exposure", lead_beta.mul(factor, axis=0))
    links = {
        "miners_gold": ("US_Stock_GDX_adj_close", "US_Stock_GLD_adj_close"),
        "silver_gold": ("US_Stock_SLV_adj_close", "US_Stock_GLD_adj_close"),
        "energy_materials": ("US_Stock_XLE_adj_close", "US_Stock_XLB_adj_close"),
        "high_yield_treasury": ("US_Stock_JNK_adj_close", "US_Stock_IEF_adj_close"),
        "credit_treasury": ("US_Stock_LQD_adj_close", "US_Stock_IEF_adj_close"),
        "duration": ("US_Stock_IEF_adj_close", "US_Stock_SHY_adj_close"),
        "inflation_linked_nominal": ("US_Stock_TIP_adj_close", "US_Stock_IEF_adj_close"),
        "emerging_global": ("US_Stock_EEM_adj_close", "US_Stock_VT_adj_close"),
    }
    for label, (a, c) in links.items():
        if a not in source or c not in source:
            continue
        spread = source[a] - source[c]
        for window in [1, 5, 21]:
            b.context("macro_links", f"{label}_change_{window}", spread.diff(window))
        b.context(
            "macro_links",
            f"{label}_relative_level",
            (spread - spread.rolling(126, min_periods=84).mean())
            / spread.rolling(126, min_periods=84).std(ddof=0).replace(0, np.nan),
        )


def currency_features(b: Builder, x: pd.DataFrame) -> None:
    columns = [c for c in x if c.startswith("FX_")]
    currencies = sorted({c[3:6] for c in columns} | {c[6:9] for c in columns})
    if len(currencies) < 3:
        return
    anchors = [c for c in currencies if c != "USD"]
    incidence = np.zeros((len(columns), len(anchors)))
    for j, c in enumerate(columns):
        for currency, sign in [(c[3:6], 1), (c[6:9], -1)]:
            if currency in anchors:
                incidence[j, anchors.index(currency)] = sign
    returns = np.log(x[columns].where(x[columns] > 0)).diff().to_numpy()
    fitted = np.full_like(returns, np.nan)
    strengths = np.full((len(x), len(anchors)), np.nan)
    for t, row in enumerate(returns):
        observed = np.isfinite(row)
        if observed.sum() < len(anchors):
            continue
        matrix = incidence[observed]
        solution, _, rank, _ = np.linalg.lstsq(matrix, row[observed], rcond=None)
        if rank < len(anchors):
            continue
        strengths[t] = solution
        fitted[t] = incidence @ solution
    observed_returns = pd.DataFrame(returns, index=x.index, columns=columns)
    network = pd.DataFrame(fitted, index=x.index, columns=columns)
    residual = observed_returns - network
    b.asset("currency_graph", "graph_implied_return", network)
    b.asset("currency_graph", "cycle_residual", residual)
    for window in [5, 21]:
        b.asset("currency_graph", f"currency_trend_{window}", roll(network, window).mean())
        b.asset("currency_graph", f"cycle_residual_{window}", roll(residual, window).mean())
    for currency in ["JPY", "AUD", "CAD", "EUR", "CHF"]:
        if currency in anchors:
            s = pd.Series(strengths[:, anchors.index(currency)], index=x.index)
            b.context("currency_graph", f"{currency}_strength", s)
    b.context("currency_graph", "quote_inconsistency", residual.abs().median(axis=1))


def contract_features(b: Builder, x: pd.DataFrame, logp: pd.DataFrame) -> None:
    source = np.log(x.where(x > 0))
    contrasts = {
        "gold_mini_standard": ("JPX_Gold_Mini_Futures_Close", "JPX_Gold_Standard_Futures_Close"),
        "gold_rolling_standard": (
            "JPX_Gold_Rolling-Spot_Futures_Close",
            "JPX_Gold_Standard_Futures_Close",
        ),
        "platinum_mini_standard": (
            "JPX_Platinum_Mini_Futures_Close",
            "JPX_Platinum_Standard_Futures_Close",
        ),
    }
    for name, (a, c) in contrasts.items():
        if a not in source or c not in source:
            continue
        spread = source[a] - source[c]
        b.context("contract_basis", name + "_log_difference", spread)
        b.context("contract_basis", name + "_change", spread.diff())
        b.context(
            "contract_basis",
            name + "_z_63",
            (spread - spread.rolling(63, min_periods=42).mean())
            / spread.rolling(63, min_periods=42).std(ddof=0).replace(0, np.nan),
        )
    settlement = {}
    for a in logp:
        column = a.removesuffix("Close") + "settlement_price"
        if column in source:
            settlement[a] = logp[a] - source[column]
    if settlement:
        frame = pd.DataFrame(settlement, index=x.index)
        b.asset("contract_basis", "close_settlement_difference", frame)
        b.asset("contract_basis", "close_settlement_change", frame.diff())
    if "FX_USDJPY" in source:
        yen = source.FX_USDJPY
        translated = {a: logp[a] - yen for a in logp if a.startswith("JPX_")}
        if translated:
            frame = pd.DataFrame(translated, index=x.index)
            b.asset("contract_basis", "usd_translated_return", frame.diff())
            b.asset("contract_basis", "usd_translated_trend_21", frame.diff(21))
        a, c = "JPX_Gold_Standard_Futures_Close", "US_Stock_GLD_adj_close"
        if a in source and c in source:
            spread = source[a] - yen - source[c]
            b.context("contract_basis", "japan_us_gold_relative_trend", spread.diff(21))
            b.context(
                "contract_basis",
                "japan_us_gold_relative_level",
                (spread - spread.rolling(126, min_periods=84).mean())
                / spread.rolling(126, min_periods=84).std(ddof=0).replace(0, np.nan),
            )


def pair_features(b: Builder, logp: pd.DataFrame) -> None:
    left = logp.reindex(columns=b.left).set_axis(b.pairs.target, axis=1)
    right = logp.reindex(columns=b.right, fill_value=0).set_axis(b.pairs.target, axis=1)
    spread = left - right
    a, c, difference = left.diff(), right.diff(), spread.diff()
    for window in [21, 63, 126]:
        var_a, var_c = roll(a, window).var(ddof=0), roll(c, window).var(ddof=0)
        covariance = a.rolling(window, min_periods=window * 2 // 3).cov(c, ddof=0)
        correlation = covariance / nonzero(np.sqrt(var_a * var_c))
        beta = (covariance / nonzero(var_c)).shift().clip(-10, 10)
        b.target("pair_dynamics", f"correlation_{window}", correlation)
        b.target("pair_dynamics", f"risk_ratio_{window}", np.sqrt(var_a / nonzero(var_c)))
        b.target(
            "pair_dynamics", f"spread_risk_{window}", np.sqrt(roll(difference, window).var(ddof=0))
        )
        b.target("pair_dynamics", f"lagged_hedge_innovation_{window}", a - beta * c)
        b.target(
            "pair_dynamics",
            f"relative_level_{window}",
            (spread - roll(spread, window).mean()) / nonzero(roll(spread, window).std(ddof=0)),
        )
        level = spread.shift()
        slope = (
            level.rolling(window, min_periods=window * 2 // 3).cov(difference, ddof=0)
            / nonzero(roll(level, window).var(ddof=0))
        ).shift()
        b.target("pair_dynamics", f"error_correction_slope_{window}", slope)
        b.target(
            "pair_dynamics",
            f"reversion_pressure_{window}",
            slope.clip(-1, 0) * (spread - roll(spread, window).mean()),
        )
    for lag in [1, 2, 5]:
        b.target(
            "pair_dynamics", f"asymmetric_cross_lag_{lag}", a * c.shift(lag) - c * a.shift(lag)
        )
    for h in [1, 2, 3, 4]:
        past = spread.diff(h)
        b.target("horizon_structure", f"observed_{h}_date_return", past)
        b.target(
            "horizon_structure",
            f"variance_ratio_{h}",
            roll(past, 63).var(ddof=0) / nonzero(h * roll(difference, 63).var(ddof=0)),
        )
    horizon = b.pairs.lag.to_numpy(dtype=float)
    for name, values in {
        "horizon": horizon,
        "sqrt_horizon": np.sqrt(horizon),
        "is_pair": np.asarray(b.right) != "",
    }.items():
        b.add("reference", name, np.broadcast_to(values, (len(logp), len(b.pairs))).copy())
    for market in ["LME", "JPX", "US", "FX"]:
        exposure = np.array(
            [
                int(a.startswith(market + "_")) - int(c.startswith(market + "_"))
                for a, c in zip(b.left, b.right, strict=True)
            ]
        )
        b.add(
            "reference",
            market + "_signed_exposure",
            np.broadcast_to(exposure, (len(logp), len(b.pairs))).copy(),
        )
