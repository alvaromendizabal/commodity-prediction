"""Fixed diagnostic estimators, structural projections, and replayable model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .screening import PreparedFeatures


@dataclass
class ControlBundle:
    target_names: list[str]
    historical_means: np.ndarray
    released: bool = False

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = np.tile(self.historical_means, (len(frame), 1))
        if self.released:
            names = [f"release_history__{target}__expanding_mean" for target in self.target_names]
            known = frame[names].to_numpy(dtype=float)
            values = np.where(np.isnan(known), values, known)
        return pd.DataFrame(values, index=frame.index, columns=self.target_names)


@dataclass
class OutputModel:
    selected: list[int]
    estimator: Any
    constant: float

    def predict(self, z: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return np.full(len(z), self.constant)
        return np.asarray(self.estimator.predict(z[:, self.selected]))


@dataclass
class ModelBundle:
    feature_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    output_names: list[str]
    target_names: list[str]
    models: list[OutputModel]
    projection: list[list[tuple[int, float]]]

    def standardize(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.medians, values)
        z = (values - self.means) / self.scales
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite replay features")
        return z

    def project(self, output: np.ndarray) -> np.ndarray:
        if output.shape[1] != len(self.output_names):
            raise ValueError("Structural output schema differs")
        prediction = np.zeros((len(output), len(self.target_names)))
        for j, components in enumerate(self.projection):
            for i, weight in components:
                prediction[:, j] += weight * output[:, i]
        return prediction

    def predict_standardized(self, z: np.ndarray) -> np.ndarray:
        return self.project(np.column_stack([m.predict(z) for m in self.models]))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        prediction = self.predict_standardized(self.standardize(frame))
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite replay predictions")
        return pd.DataFrame(prediction, index=frame.index, columns=self.target_names)


def projection_for(pairs: pd.DataFrame, output_names: list[str], structural: bool) -> list:
    positions = {name: i for i, name in enumerate(output_names)}
    projection = []
    for target, lag, pair in pairs.itertuples(index=False, name=None):
        if structural:
            components = pair.split(" - ")
            projection.append(
                [
                    (positions[f"h{lag}:{asset}"], 1.0 if j == 0 else -1.0)
                    for j, asset in enumerate(components)
                ]
            )
        else:
            projection.append([(positions[target], 1.0)])
    return projection


def fit_output(
    prepared: PreparedFeatures,
    values: np.ndarray,
    selected: list[int],
    algorithm: str,
    config: dict,
) -> OutputModel:
    observed = np.isfinite(values)
    mean = float(values[observed].mean()) if observed.any() else 0.0
    if observed.sum() < config["minimum_output_observations"] or not selected:
        return OutputModel(selected, None, mean)
    model: Any
    if algorithm == "ridge":
        model = Ridge(alpha=config["ridge_alpha"], solver="cholesky")
    elif algorithm == "histogram":
        # Disable automatic random validation; the outer folds are chronological.
        model = HistGradientBoostingRegressor(
            max_iter=config["histogram_iterations"],
            learning_rate=config["histogram_learning_rate"],
            max_leaf_nodes=config["histogram_max_leaf_nodes"],
            min_samples_leaf=config["histogram_min_samples_leaf"],
            l2_regularization=config["histogram_l2_regularization"],
            early_stopping=False,
            random_state=config["seed"],
        )
    else:
        raise ValueError(f"Unsupported diagnostic algorithm: {algorithm}")
    model.fit(prepared.train[observed][:, selected], values[observed])
    return OutputModel(selected, model, mean)
