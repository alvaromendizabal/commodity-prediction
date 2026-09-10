"""Prespecified feature decomposition, information delays, and admitted interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.robustness.features import Variant as ParentVariant
from commodity_prediction.domain.robustness.features import transformed_panel

BASE = ("reference", "released_priors")
JOINT = (*BASE, "tail_risk", "asynchrony")


@dataclass(frozen=True)
class Variant:
    name: str
    families: tuple[str, ...] = JOINT
    admit: bool = False
    prior_delay: int = 0
    market_delay: int = 0
    interactions: str = ""

    def settings(self, config: dict, panel: Panel) -> dict:
        return {
            **config,
            "max_features": len(panel.names) if self.admit else 64,
            "max_abs_correlation": 1.01 if self.admit else 0.98,
        }


def experiment_plan() -> list[Variant]:
    return [
        Variant("screened_tail", (*BASE, "tail_risk")),
        Variant("screened_freshness", (*BASE, "asynchrony")),
        Variant("admitted_base", BASE, admit=True),
        Variant("admitted_tail", (*BASE, "tail_risk"), admit=True),
        Variant("admitted_freshness", (*BASE, "asynchrony"), admit=True),
        Variant("admitted_joint", admit=True),
        Variant("compact_prior_delay_1", prior_delay=1),
        Variant("compact_prior_delay_5", prior_delay=5),
        Variant("compact_market_delay_1", market_delay=1),
        Variant("compact_both_delay_1", prior_delay=1, market_delay=1),
        Variant("admitted_products", admit=True, interactions="products"),
        Variant("admitted_fx_states", admit=True, interactions="fx_states"),
    ]


def build_panel(parent: Panel, pairs: pd.DataFrame, variant: Variant) -> Panel:
    if variant.prior_delay not in {0, 1, 5} or variant.market_delay not in {0, 1}:
        raise ValueError("Undeclared information delay")
    panel = transformed_panel(parent, pairs, ParentVariant(variant.name, families=variant.families))
    if variant.prior_delay:
        mode = f"released_delay_{variant.prior_delay}"
        panel = transformed_panel(panel, pairs, ParentVariant(variant.name, mode))
    if variant.market_delay:
        panel = transformed_panel(panel, pairs, ParentVariant(variant.name, "market_delay_1"))
    additions: dict[str, np.ndarray] = {}
    if variant.interactions == "products":
        # Use the exact twelve frozen products; their sources need not be in the compact base.
        expanded = transformed_panel(parent, pairs, ParentVariant("products", "mechanism_products"))
        additions = {
            n: expanded.values[:, :, i]
            for i, n in enumerate(expanded.names)
            if n.startswith("mechanism_products__")
        }
    elif variant.interactions == "fx_states":
        assets = [p.split(" - ") for p in pairs.pair]
        gate = np.asarray([sum(a.startswith("FX_") for a in legs) / len(legs) for legs in assets])
        for i, name in enumerate(panel.names):
            if name.split("__", 1)[0] in {"tail_risk", "asynchrony"}:
                additions["fx_states__" + name.replace("__", "_")] = np.where(
                    gate[None, :] == 0, 0, panel.values[:, :, i] * gate[None, :]
                )
    elif variant.interactions:
        raise ValueError("Unknown interaction family")
    if additions:
        panel = Panel(
            np.concatenate(
                [panel.values, np.stack(list(additions.values()), axis=2).astype(np.float32)],
                axis=2,
            ),
            panel.dates,
            panel.targets,
            [*panel.names, *additions],
            dict(panel.source_series),
        )
    panel.validate()
    return panel


def declared_comparisons() -> list[tuple[str, str]]:
    pairs = [(v.name, "historical_mean") for v in experiment_plan()]
    pairs += [
        (n, "compact_control") for n in ["screened_tail", "screened_freshness", "admitted_base"]
    ]
    pairs += [("joint_control", n) for n in ["screened_tail", "screened_freshness"]]
    pairs += [
        (n, "admitted_base") for n in ["admitted_tail", "admitted_freshness", "admitted_joint"]
    ]
    pairs += [
        ("admitted_joint", n) for n in ["admitted_tail", "admitted_freshness", "joint_control"]
    ]
    pairs += [
        (n, "joint_control")
        for n in [
            "compact_prior_delay_1",
            "compact_prior_delay_5",
            "compact_market_delay_1",
            "compact_both_delay_1",
        ]
    ]
    pairs += [(n, "admitted_joint") for n in ["admitted_products", "admitted_fx_states"]]
    return pairs
