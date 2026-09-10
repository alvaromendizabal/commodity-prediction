"""Prespecified matched feature experiments and training-only residual calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_prediction.data import Fold
from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations

from .features import FAMILIES
from .model import Model, graph_mean, target_moments


@dataclass(frozen=True)
class Experiment:
    name: str
    include: str = "all"
    drop: str = ""
    stable: bool = False
    algorithm: str = "ridge"

    def candidates(self, names: list[str]) -> list[str]:
        return [
            name
            for name in names
            if (self.include == "all" or name.split("__", 1)[0] in {"reference", self.include})
            and name.split("__", 1)[0] != self.drop
        ]


def experiment_plan() -> list[Experiment]:
    return [
        Experiment("pooled_reference", include="reference"),
        *[Experiment(f"add_{family}", include=family) for family in FAMILIES],
        Experiment("pooled_all"),
        Experiment("pooled_stable", stable=True),
        *[Experiment(f"drop_{family}", drop=family) for family in FAMILIES],
        Experiment("tree_reference", include="reference", algorithm="histogram"),
        Experiment("tree_all", algorithm="histogram"),
    ]


def inner_folds(stop: int, config: dict) -> list[Fold]:
    width, count = config["inner_validation_dates"], config["inner_folds"]
    folds = []
    for number in range(count):
        start = stop - (count - number) * width
        fit_stop = start - config["purge_dates"]
        if fit_stop <= config["warmup_dates"] + 40 or config["purge_dates"] < 5:
            raise ValueError("Insufficient history or unsafe inner purge")
        folds.append(Fold(number, fit_stop, start, start + width))
    return folds


def choose_weight(
    records: list[tuple[pd.DataFrame, pd.DataFrame, np.ndarray]], config: dict
) -> dict:
    """Records contain only inner validation truth, predictions, and inner-fit means."""
    scores = []
    for weight in config["residual_weights"]:
        daily = []
        for truth, raw, mean in records:
            prediction = pd.DataFrame(
                mean + weight * (raw.to_numpy() - mean), index=raw.index, columns=raw.columns
            )
            daily.append(daily_rank_correlations(truth, prediction))
        scores.append(
            {"weight": weight, "official_metric": correlation_sharpe(np.concatenate(daily))}
        )
    chosen = max(scores, key=lambda row: (row["official_metric"], -row["weight"]))
    return {"selected_weight": chosen["weight"], "inner_scores": scores}


def control_model(
    name: str, y: pd.DataFrame, pairs: pd.DataFrame, stop: int, config: dict
) -> Model:
    mean, scale = target_moments(y, stop)
    if name == "graph_mean":
        mean = graph_mean(y, pairs, stop)
    elif name == "winsorized_mean":
        values = y.iloc[:stop]
        mean = (
            values.clip(values.quantile(0.01), values.quantile(0.99), axis=1)
            .mean()
            .fillna(0)
            .to_numpy()
        )
    elif name != "historical_mean":
        raise ValueError("Unknown control")
    return Model(
        [],
        np.array([], dtype=int),
        np.array([]),
        np.zeros((2, 0)),
        np.array([]),
        mean,
        scale,
        np.array([]),
        0,
        None,
        config["standardized_clip"],
        0,
    )


def declared_comparisons(names: list[str]) -> list[tuple[str, str]]:
    result = [(name, "historical_mean") for name in names if name != "historical_mean"]
    for prefix in ["", "raw__"]:
        result += [(prefix + "add_" + family, prefix + "pooled_reference") for family in FAMILIES]
        result += [(prefix + "pooled_all", prefix + "drop_" + family) for family in FAMILIES]
        result += [
            (prefix + "pooled_all", prefix + "pooled_reference"),
            (prefix + "pooled_stable", prefix + "pooled_all"),
            (prefix + "tree_all", prefix + "tree_reference"),
        ]
    return list(dict.fromkeys((a, b) for a, b in result if a in names and b in names))
