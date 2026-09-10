"""Dataset contracts and horizon-aware chronological validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def validate_dates(frame: pd.DataFrame) -> None:
    if "date_id" not in frame or not pd.api.types.is_integer_dtype(frame.date_id):
        raise ValueError("date_id must be an integer column")
    if not frame.date_id.is_unique or not frame.date_id.is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing")
    if len(frame) < 2 or not np.all(np.diff(frame.date_id) == 1):
        raise ValueError("Date gaps require an explicit release calendar")


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = root / "data/raw"
    x = pd.read_csv(raw / "train.csv")
    y = pd.read_csv(raw / "train_labels.csv").replace(-999999, np.nan)
    pairs = pd.read_csv(raw / "target_pairs.csv")
    validate_dates(x)
    validate_dates(y)
    if not x.date_id.equals(y.date_id):
        raise ValueError("Feature and label date alignment differs")
    expected = [f"target_{i}" for i in range(424)]
    if list(y.columns) != ["date_id", *expected]:
        raise ValueError("Expected the official ordered 424-target schema")
    if set(pairs.columns) != {"target", "lag", "pair"} or pairs.target.duplicated().any():
        raise ValueError("Invalid target-pair metadata")
    if set(pairs.target) != set(expected) or not pairs.lag.isin([1, 2, 3, 4]).all():
        raise ValueError("Target metadata coverage/horizons differ")
    for pair in pairs.pair:
        columns = pair.split(" - ")
        if not 1 <= len(columns) <= 2 or not set(columns).issubset(x.columns):
            raise ValueError(f"Unknown target inputs: {pair}")
    for frame in [x, y]:
        if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
            raise ValueError("Non-numeric market data")
        if np.isinf(frame.to_numpy(dtype=float)).any():
            raise ValueError("Infinite input values")
    for lag in range(1, 5):
        released = pd.read_csv(raw / f"lagged_test_labels/test_labels_lag_{lag}.csv")
        if not (released.date_id - released.label_date_id).eq(lag + 1).all():
            raise ValueError("Observed label releases differ from the horizon + 1 rule")
    return x.set_index("date_id"), y.set_index("date_id"), pairs


@dataclass(frozen=True)
class Fold:
    number: int
    train_stop: int
    validation_start: int
    validation_stop: int


def make_folds(n_dates: int, config: dict) -> tuple[list[Fold], int]:
    development_stop = n_dates - config["holdout_dates"]
    if config["purge_dates"] < 5:
        raise ValueError("At least five dates must be purged for label release timing")
    first = development_stop - config["n_folds"] * config["validation_dates"]
    if first - config["purge_dates"] < config["min_train_dates"]:
        raise ValueError("Not enough history for the declared validation protocol")
    folds = []
    for number in range(config["n_folds"]):
        start = first + number * config["validation_dates"]
        folds.append(
            Fold(number, start - config["purge_dates"], start, start + config["validation_dates"])
        )
    return folds, development_stop


def reconstruct_targets(x: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Audit only: these future-looking values must never become features."""
    out = {}
    for target, lag, pair in pairs.itertuples(index=False, name=None):
        components = []
        for column in pair.split(" - "):
            log_price = np.log(x[column].where(x[column] > 0))
            components.append(log_price.shift(-lag - 1) - log_price.shift(-1))
        out[target] = components[0] if len(components) == 1 else components[0] - components[1]
    return pd.DataFrame(out, index=x.index)
