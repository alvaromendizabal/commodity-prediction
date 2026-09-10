"""Generate a comprehensive causal feature panel without touching final-test rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.runtime import RunLog
from commodity_prediction.studies.representation import validate_time_index

from .catalog import Builder, Panel
from .market import latent_features, market_features
from .relationships import contract_features, currency_features, factor_features, pair_features

FAMILIES = [
    "trend_shape",
    "tail_risk",
    "trading_activity",
    "intraday_path",
    "asynchrony",
    "factor_relative",
    "macro_links",
    "currency_graph",
    "contract_basis",
    "pair_dynamics",
    "released_priors",
    "horizon_structure",
    "latent_factors",
]


def released_features(b: Builder, y: pd.DataFrame) -> None:
    known = pd.DataFrame(
        {r.target: y[r.target].shift(int(r.lag) + 1) for r in b.pairs.itertuples(index=False)},
        index=y.index,
    )
    for window in [63, 126, 252]:
        ewm = known.ewm(span=window, min_periods=40, adjust=False)
        b.target("released_priors", f"location_{window}", ewm.mean())
        b.target("released_priors", f"risk_{window}", ewm.std(bias=True))
    median = known.rolling(63, min_periods=40).median()
    iqr = known.rolling(63, min_periods=40).quantile(0.75) - known.rolling(
        63, min_periods=40
    ).quantile(0.25)
    b.target("released_priors", "median_63", median)
    b.target("released_priors", "iqr_63", iqr)
    b.target(
        "released_priors", "standardized_innovation", (known - median) / iqr.where(iqr > 1e-12)
    )
    b.target("released_priors", "availability_63", known.notna().rolling(63, min_periods=40).mean())
    b.target(
        "released_priors",
        "positive_frequency_126",
        (known > 0).where(known.notna()).astype(float).rolling(126, min_periods=84).mean(),
    )
    group_frame = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    keys: dict[tuple, list[str]] = {}
    canonical: dict[tuple, list[tuple[str, float, int]]] = {}
    for row in b.pairs.itertuples(index=False):
        assets = row.pair.split(" - ")
        key = (
            row.lag,
            assets[0].split("_")[0],
            assets[1].split("_")[0] if len(assets) == 2 else "single",
        )
        keys.setdefault(key, []).append(row.target)
        sign = 1.0 if len(assets) == 1 or assets == sorted(assets) else -1.0
        canonical.setdefault(tuple(sorted(assets)), []).append((row.target, sign, row.lag))
    for columns in keys.values():
        # All constituents are released before aggregation; group membership is metadata only.
        series = known[columns].mean(axis=1).ewm(span=126, min_periods=40, adjust=False).mean()
        group_frame[columns] = np.broadcast_to(series.to_numpy()[:, None], (len(y), len(columns)))
    b.target("released_priors", "market_pair_pool", group_frame)
    count = known.notna().rolling(126, min_periods=40).sum()
    weight = count / (count + 63)
    own = known.ewm(span=126, min_periods=40, adjust=False).mean()
    b.target(
        "released_priors", "shrunk_market_pair_pool", own * weight + group_frame * (1 - weight)
    )
    cross_horizon = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for records in canonical.values():
        daily_rates = pd.concat(
            [known[target] * sign / h for target, sign, h in records], axis=1
        ).mean(axis=1)
        prior = daily_rates.ewm(span=126, min_periods=40, adjust=False).mean()
        for target, sign, h in records:
            cross_horizon[target] = prior * sign * h
    b.target("released_priors", "canonical_cross_horizon_pool", cross_horizon)


def build_panel(x: pd.DataFrame, y: pd.DataFrame, pairs: pd.DataFrame, log: RunLog) -> Panel:
    validate_time_index(x)
    if not x.index.equals(y.index) or list(y.columns) != pairs.target.tolist():
        raise ValueError("Panel data must have aligned dates and ordered target metadata")
    assets = sorted({a for pair in pairs.pair for a in pair.split(" - ")})
    if not set(assets).issubset(x):
        raise ValueError("Target assets are missing from market data")
    logp = np.log(x[assets].where(x[assets] > 0))
    builder = Builder(x.index, pairs)
    with log.stage("domain_market_paths"):
        market_features(builder, x, logp)
    with log.stage("domain_economic_relationships"):
        factor_features(builder, x, logp)
        currency_features(builder, x)
        contract_features(builder, x, logp)
        pair_features(builder, logp)
    with log.stage("domain_released_information"):
        released_features(builder, y)
    with log.stage("domain_causal_latent_factors"):
        latent_features(builder, logp)
    panel = builder.finish()
    actual = set(name.split("__", 1)[0] for name in panel.names)
    if not actual.issubset({"reference", *FAMILIES}):
        raise ValueError("Undeclared feature family")
    return panel
