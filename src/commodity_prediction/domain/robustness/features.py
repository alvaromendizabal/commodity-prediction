"""Causal sensitivity transforms and compact economic interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel


@dataclass(frozen=True)
class Variant:
    name: str
    transform: str = "identity"
    max_features: int = 64
    max_correlation: float = 0.98
    families: tuple[str, ...] = ()


def experiment_plan() -> list[Variant]:
    compact = ("reference", "released_priors")
    return [
        Variant("budget_32", max_features=32),
        Variant("budget_128", max_features=128),
        Variant("budget_all", max_features=380),
        Variant("redundancy_relaxed", max_features=380, max_correlation=1.01),
        Variant("released_delay_1", "released_delay_1"),
        Variant("released_delay_5", "released_delay_5"),
        Variant("market_delay_1", "market_delay_1"),
        Variant("conditional_priors", "conditional_priors"),
        Variant("mechanism_products", "mechanism_products"),
        Variant("compact_conditional", "conditional_priors", families=compact),
        Variant("compact_factor", families=(*compact, "factor_relative")),
        Variant("compact_risk_freshness", families=(*compact, "tail_risk", "asynchrony")),
    ]


def transformed_panel(parent: Panel, pairs: pd.DataFrame, variant: Variant) -> Panel:
    if pairs.target.tolist() != parent.targets:
        raise ValueError("Target order differs from the parent panel")
    keep = [
        i
        for i, name in enumerate(parent.names)
        if not variant.families or name.split("__", 1)[0] in variant.families
    ]
    values = parent.values if len(keep) == len(parent.names) else parent.values[:, :, keep]
    names = [parent.names[i] for i in keep]
    mode = variant.transform
    additions: dict[str, np.ndarray] = {}
    if mode.startswith("released_delay_") or mode == "market_delay_1":
        delay = int(mode.rsplit("_", 1)[1])
        static = {"reference__horizon", "reference__sqrt_horizon", "reference__is_pair"}
        static.update("reference__" + m + "_signed_exposure" for m in ["LME", "JPX", "US", "FX"])
        positions = [
            i
            for i, n in enumerate(names)
            if (
                n.startswith("released_priors__")
                if mode.startswith("released_delay_")
                else not n.startswith("released_priors__") and n not in static
            )
        ]
        values = values.copy()
        values[delay:, :, positions] = values[:-delay, :, positions]
        values[:delay, :, positions] = np.nan
    elif mode == "conditional_priors":
        assets = [p.split(" - ") for p in pairs.pair]
        gates = {"long_horizon": (pairs.lag.to_numpy() >= 3).astype(float)}
        for market in ["FX", "US", "metals"]:
            markets = {"LME", "JPX"} if market == "metals" else {market}
            gates[market + "_leg_share"] = np.array(
                [sum(a.split("_")[0] in markets for a in legs) / len(legs) for legs in assets]
            )
        for i, name in enumerate(names):
            if name.startswith("released_priors__"):
                for gate, weight in gates.items():
                    # Unsupported groups are structural zero; applicable missing priors stay missing.
                    additions["conditional_priors__" + name.split("__", 1)[1] + "_" + gate] = (
                        np.where(weight[None, :] == 0, 0, values[:, :, i] * weight[None, :])
                    )
    elif mode == "mechanism_products":
        for suffix in ["difference", "sum"]:
            drivers = [
                "reference__risk_scaled_return_lag_0_",
                "trend_shape__peer_momentum_rank_21_",
            ]
            states = [
                "tail_risk__robust_shock_",
                "trading_activity__relative_volume_",
                "asynchrony__resumed_after_gap_",
            ]
            for d, driver in enumerate(drivers):
                for s, state in enumerate(states):
                    a, b = (
                        values[:, :, names.index(driver + suffix)],
                        values[:, :, names.index(state + suffix)],
                    )
                    additions[f"mechanism_products__driver_{d}_state_{s}_{suffix}"] = np.tanh(
                        a
                    ) * np.tanh(b)
    elif mode != "identity":
        raise ValueError("Unknown robustness transformation")
    if additions:
        values = np.concatenate(
            [values, np.stack(list(additions.values()), axis=2).astype(np.float32)], axis=2
        )
        names += list(additions)
    panel = Panel(values, parent.dates, parent.targets, names, dict(parent.source_series))
    panel.validate()
    return panel
