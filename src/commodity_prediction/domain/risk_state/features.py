"""Causal risk states and normalized priors with fixed, matched ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.compact.features import BASE, JOINT
from commodity_prediction.domain.compact.features import Variant as ParentVariant
from commodity_prediction.domain.compact.features import build_panel as parent_panel

TAIL = (*BASE, "tail_risk")


@dataclass(frozen=True)
class Variant:
    name: str
    families: tuple[str, ...] = TAIL
    states: bool = True
    drivers: bool = True
    products: bool = True
    admit: bool = True
    prior_delay: int = 0

    def settings(self, config: dict, panel: Panel) -> dict:
        return {
            **config,
            "max_features": len(panel.names) if self.admit else 64,
            "max_abs_correlation": 1.01 if self.admit else 0.98,
        }


def experiment_plan() -> list[Variant]:
    return [
        Variant("tail_states", drivers=False, products=False),
        Variant("tail_scaled_priors", states=False, products=False),
        Variant("tail_states_and_priors", products=False),
        Variant("tail_state_products"),
        Variant("joint_state_products", families=JOINT),
        Variant("base_state_products", families=BASE),
        Variant("screened_state_products", admit=False),
        Variant("delayed_state_products", prior_delay=1),
    ]


def feature_blocks(panel: Panel) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All inputs already obey horizon+1 release; rolling windows end at the origin."""

    def prior(name: str) -> pd.DataFrame:
        values = panel.values[:, :, panel.names.index("released_priors__" + name)]
        return pd.DataFrame(values.astype(float), index=panel.dates, columns=panel.targets)

    risks = {w: prior(f"risk_{w}") for w in [63, 126, 252]}
    risks = {w: v.where(v > 1e-12) for w, v in risks.items()}
    log_risk = np.log(risks[63])
    states = {
        "risk_ratio_63_252": np.tanh(np.log(risks[63] / risks[252])),
        "risk_ratio_126_252": np.tanh(np.log(risks[126] / risks[252])),
        "risk_change_5": np.tanh(log_risk.diff(5)),
        "risk_change_21": np.tanh(log_risk.diff(21)),
        "risk_instability_21": np.tanh(log_risk.rolling(21, min_periods=14).std(ddof=0)),
        "risk_percentile_126": 2 * log_risk.rolling(126, min_periods=84).rank(pct=True) - 1,
    }
    drivers = {
        "location_63_per_risk": np.tanh(prior("location_63") / risks[63]),
        "location_126_per_risk": np.tanh(prior("location_126") / risks[126]),
        "pool_per_risk": np.tanh(prior("shrunk_market_pair_pool") / risks[63]),
        "location_change_per_risk": np.tanh(
            (prior("location_63") - prior("location_252")) / risks[252]
        ),
    }
    return (
        {name: value.to_numpy(dtype=np.float32) for name, value in states.items()},
        {name: value.to_numpy(dtype=np.float32) for name, value in drivers.items()},
    )


def build_panel(parent: Panel, pairs: pd.DataFrame, variant: Variant) -> Panel:
    if variant.products and not (variant.states and variant.drivers):
        raise ValueError("Products require their matched state and driver main effects")
    panel = parent_panel(
        parent,
        pairs,
        ParentVariant(variant.name, variant.families, prior_delay=variant.prior_delay),
    )
    states, drivers = feature_blocks(panel)
    additions = {}
    if variant.states:
        additions.update({"risk_state__" + k: v for k, v in states.items()})
    if variant.drivers:
        additions.update({"scaled_prior__" + k: v for k, v in drivers.items()})
    if variant.products:
        additions.update(
            {
                "state_prior_product__" + state + "_" + driver: a * b
                for state, a in states.items()
                for driver, b in drivers.items()
            }
        )
    result = Panel(
        np.concatenate([panel.values, np.stack(list(additions.values()), axis=2)], axis=2),
        panel.dates,
        panel.targets,
        [*panel.names, *additions],
        dict(panel.source_series),
    )
    result.validate()
    return result


def declared_comparisons() -> list[tuple[str, str]]:
    comparisons = [(v.name, "historical_mean") for v in experiment_plan()]
    comparisons += [
        (name, "admitted_tail")
        for name in [
            "tail_states",
            "tail_scaled_priors",
            "tail_states_and_priors",
            "tail_state_products",
        ]
    ]
    comparisons += [
        ("tail_state_products", "tail_states_and_priors"),
        ("tail_states_and_priors", "tail_states"),
        ("tail_states_and_priors", "tail_scaled_priors"),
        ("joint_state_products", "admitted_joint"),
        ("joint_state_products", "tail_state_products"),
        ("base_state_products", "admitted_base"),
        ("tail_state_products", "base_state_products"),
        ("screened_state_products", "screened_tail"),
        ("screened_state_products", "tail_state_products"),
        ("delayed_state_products", "tail_state_products"),
    ]
    return comparisons
