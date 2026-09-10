"""Recent label information with per-target releases and signed asset relationships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.compact.features import BASE
from commodity_prediction.domain.compact.features import Variant as ParentVariant
from commodity_prediction.domain.compact.features import build_panel as parent_panel
from commodity_prediction.studies.representation import validate_time_index


@dataclass(frozen=True)
class Variant:
    name: str
    own: bool = True
    peers: bool = True

    def settings(self, config: dict, panel: Panel) -> dict:
        return {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}


def experiment_plan() -> list[Variant]:
    return [
        Variant("short_own", peers=False),
        Variant("short_peers", own=False),
        Variant("short_joint"),
    ]


def released_values(y: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    validate_time_index(y)
    if list(y.columns) != pairs.target.tolist() or pairs.target.duplicated().any():
        raise ValueError("Released-history target metadata must match ordered labels")
    if not pairs.lag.isin([1, 2, 3, 4]).all():
        raise ValueError("Unsupported release horizon")
    clean = y.replace([-999999, np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {r.target: clean[r.target].shift(int(r.lag) + 1) for r in pairs.itertuples(index=False)},
        index=y.index,
    )


def peer_weights(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rows receive peers; shared-pair and shared-asset groups exclude the own target."""
    legs = [pair.split(" - ") for pair in pairs.pair]
    assets = sorted({asset for row in legs for asset in row})
    incidence = np.zeros((len(pairs), len(assets)), dtype=float)
    for i, row in enumerate(legs):
        if len(row) not in {1, 2} or len(set(row)) != len(row):
            raise ValueError("Invalid target asset expression")
        for j, asset in enumerate(row):
            incidence[i, assets.index(asset)] = 1 if j == 0 else -1
    overlap = incidence @ incidence.T
    canonical = [tuple(sorted(row)) for row in legs]
    same = np.asarray([[a == b for b in canonical] for a in canonical])
    same_pair = np.where(same, np.sign(overlap), 0.0)
    np.fill_diagonal(same_pair, 0.0)
    shared_asset = np.where(same, 0.0, overlap)
    return {"same_pair": same_pair, "shared_asset": shared_asset}


def short_blocks(
    y: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    known = released_values(y, pairs)
    rolling = known.rolling(4, min_periods=3)
    own = {f"snapshot_{k}": known.shift(k) for k in range(4)}
    own.update(
        mean_2=known.rolling(2, min_periods=2).mean(),
        mean_4=rolling.mean(),
        change_1=known.diff(),
        change_3=known.diff(3),
        innovation_4=known - rolling.mean(),
        volatility_4=rolling.std(ddof=0),
        sign_balance_4=np.sign(known).rolling(4, min_periods=3).mean(),
        availability_4=known.notna().rolling(4, min_periods=4).mean(),
    )
    horizon = pairs.lag.to_numpy(dtype=float)
    rates = known.to_numpy(dtype=float) / horizon
    available = np.isfinite(rates)
    filled = np.where(available, rates, 0.0)
    peer: dict[str, np.ndarray] = {}
    for group, weights in peer_weights(pairs).items():
        absolute = np.abs(weights)
        denominator = available.astype(float) @ absolute.T
        mean = np.divide(
            filled @ weights.T,
            denominator,
            out=np.full_like(rates, np.nan),
            where=denominator > 0,
        )
        second = np.divide(
            (filled * filled) @ absolute.T,
            denominator,
            out=np.full_like(rates, np.nan),
            where=denominator > 0,
        )
        dispersion = np.sqrt(np.maximum(second - mean * mean, 0)) * horizon
        values = pd.DataFrame(mean * horizon, index=y.index, columns=y.columns)
        maximum = absolute.sum(axis=1)
        coverage = np.divide(
            denominator,
            maximum[None, :],
            out=np.zeros_like(rates),
            where=maximum[None, :] > 0,
        )
        transforms = {
            "latest": values.to_numpy(),
            "mean_2": values.rolling(2, min_periods=2).mean().to_numpy(),
            "mean_4": values.rolling(4, min_periods=3).mean().to_numpy(),
            "change_1": values.diff().to_numpy(),
            "dispersion": dispersion,
            "coverage": coverage,
        }
        peer.update(
            {group + "_" + name: value.astype(np.float32) for name, value in transforms.items()}
        )
    return ({name: value.to_numpy(dtype=np.float32) for name, value in own.items()}, peer)


def build_panel(parent: Panel, y: pd.DataFrame, pairs: pd.DataFrame, variant: Variant) -> Panel:
    if parent.dates != y.index.tolist() or parent.targets != list(y.columns):
        raise ValueError("Short context must align with the frozen development panel")
    panel = parent_panel(
        parent, pairs, ParentVariant(variant.name, (*BASE, "tail_risk"), admit=True)
    )
    own, peers = short_blocks(y, pairs)
    additions = {}
    if variant.own:
        additions.update({"short_own__" + name: value for name, value in own.items()})
    if variant.peers:
        additions.update({"short_peers__" + name: value for name, value in peers.items()})
    if not additions:
        raise ValueError("A context variant must contain an added family")
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
    return [
        (v.name, c) for v in experiment_plan() for c in ["admitted_tail", "historical_mean"]
    ] + [
        ("short_joint", "short_own"),
        ("short_joint", "short_peers"),
    ]
