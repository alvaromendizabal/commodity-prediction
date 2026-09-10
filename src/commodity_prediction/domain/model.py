"""Fixed pooled residual models with training-only statistics and traceable screening."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .catalog import Panel


def target_moments(y: pd.DataFrame, stop: int) -> tuple[np.ndarray, np.ndarray]:
    fit = y.iloc[:stop]
    mean = fit.mean().fillna(0).to_numpy()
    scale = fit.std(ddof=0).to_numpy()
    eligible = scale[np.isfinite(scale) & (scale > 1e-12)]
    floor = max(float(np.median(eligible)) * 0.05, 1e-8) if len(eligible) else 1e-8
    return mean, np.where(np.isfinite(scale), np.maximum(scale, floor), floor)


def graph_mean(y: pd.DataFrame, pairs: pd.DataFrame, stop: int) -> np.ndarray:
    """Project training target means onto a shared per-date asset-drift graph."""
    assets = sorted({a for pair in pairs.pair for a in pair.split(" - ")})
    matrix = np.zeros((len(pairs), len(assets)))
    horizon = pairs.lag.to_numpy(dtype=float)
    for i, pair in enumerate(pairs.pair):
        for j, asset in enumerate(pair.split(" - ")):
            matrix[i, assets.index(asset)] = 1 if j == 0 else -1
    counts = y.iloc[:stop].notna().sum().to_numpy()
    means = y.iloc[:stop].mean().fillna(0).to_numpy()
    weight = np.sqrt(counts)
    daily, _, _, _ = np.linalg.lstsq(matrix * weight[:, None], means / horizon * weight, rcond=None)
    return (matrix @ daily) * horizon


@dataclass
class Statistics:
    names: list[str]
    positions: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    relevance: np.ndarray
    stable_relevance: np.ndarray
    signatures: list[str]
    rejected: dict[str, str]
    outcome_mean: float
    target_mean: np.ndarray
    target_scale: np.ndarray
    train_start: int
    train_stop: int


def weighted_correlation(z: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    mx = weights @ z
    my = weights @ outcome
    numerator = (z * weights[:, None]).T @ outcome - mx * my
    vx = weights @ (z * z) - mx * mx
    vy = weights @ (outcome * outcome) - my * my
    denominator = np.sqrt(np.maximum(vx, 0) * max(float(vy), 0))
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )


def prepare(panel: Panel, y: pd.DataFrame, stop: int, config: dict) -> Statistics:
    """Every value here is estimated solely from the fitting interval."""
    start = config["warmup_dates"]
    if stop <= start + 40 or panel.targets != list(y.columns):
        raise ValueError("Insufficient training history or mismatched target order")
    target_mean, target_scale = target_moments(y, stop)
    labels = y.iloc[start:stop].to_numpy()
    observed = np.isfinite(labels)
    mask = observed.ravel()
    if not mask.any():
        raise ValueError("No observed fitting labels")
    values = panel.values[start:stop].reshape(-1, len(panel.names))[mask].astype(float)
    outcome = np.clip(
        (labels - target_mean) / target_scale,
        -config["standardized_clip"],
        config["standardized_clip"],
    ).ravel()[mask]
    # Each observed date has total weight one; missing-label counts do not change its influence.
    counts = observed.sum(axis=1)
    date_weights = np.divide(1.0, counts, out=np.zeros(len(counts)), where=counts > 0)
    weights = np.repeat(date_weights, len(panel.targets))[mask]
    weights /= weights.sum()
    missing = np.isnan(values).mean(axis=0)
    usable = np.flatnonzero(missing <= config["max_missing_fraction"])
    rejected = {
        n: "missingness"
        for n, m in zip(panel.names, missing, strict=True)
        if m > config["max_missing_fraction"]
    }
    medians = np.nanmedian(values[:, usable], axis=0)
    filled = np.where(np.isnan(values[:, usable]), medians, values[:, usable])
    means = weights @ filled
    scales = np.sqrt(np.maximum(weights @ (filled * filled) - means * means, 0))
    keep = scales > 1e-10
    for j in usable[~keep]:
        rejected[panel.names[j]] = "constant"
    usable, medians, means, scales = usable[keep], medians[keep], means[keep], scales[keep]
    filled = filled[:, keep]
    signatures = [
        hashlib.sha256(filled[:, i].tobytes()).hexdigest() for i in range(filled.shape[1])
    ]
    z = np.clip(
        (filled - means) / scales, -config["standardized_clip"], config["standardized_clip"]
    )
    # Recenter the clipped representation for the exact weighted ridge normal equations.
    clipped_mean = weights @ z
    z -= clipped_mean
    means = np.vstack([means, clipped_mean])
    outcome_mean = float(weights @ outcome)
    centered = outcome - outcome_mean
    gram = (z * weights[:, None]).T @ z
    rhs = (z * weights[:, None]).T @ centered
    middle = ((stop - start) // 2) * len(panel.targets)
    first = np.flatnonzero(mask) < middle
    full = weighted_correlation(z, outcome, weights)
    a = weighted_correlation(z[first], outcome[first], weights[first])
    c = weighted_correlation(z[~first], outcome[~first], weights[~first])
    stable = np.where(a * c > 0, np.minimum(np.abs(a), np.abs(c)) + 0.25 * np.abs(full), 0)
    return Statistics(
        [panel.names[i] for i in usable],
        usable,
        medians,
        means,
        scales,
        gram,
        rhs,
        np.abs(full),
        stable,
        signatures,
        rejected,
        outcome_mean,
        target_mean,
        target_scale,
        start,
        stop,
    )


def select(
    stats: Statistics, candidate_names: list[str], config: dict, stable: bool
) -> tuple[list[int], dict]:
    positions = {n: i for i, n in enumerate(stats.names)}
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    available = []
    for name in candidate_names:
        if name in stats.rejected:
            reasons[stats.rejected[name]] += 1
            continue
        i = positions[name]
        if stats.signatures[i] in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(stats.signatures[i])
        available.append(i)
    relevance = stats.stable_relevance if stable else stats.relevance
    selected: list[int] = []
    for i in sorted(available, key=lambda k: (-float(relevance[k]), k)):
        if stable and relevance[i] <= 0:
            reasons["unstable_sign"] += 1
        elif len(selected) >= config["max_features"]:
            reasons["feature_budget"] += 1
        elif (
            selected
            and np.max(
                np.abs(stats.gram[selected, i])
                / np.sqrt(np.maximum(np.diag(stats.gram)[selected] * stats.gram[i, i], 1e-20))
            )
            > config["max_abs_correlation"]
        ):
            reasons["correlated"] += 1
        else:
            selected.append(i)
    audit = {
        "candidate_templates": len(candidate_names),
        "retained_templates": len(selected),
        "rejected_templates": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "selected_names": [stats.names[i] for i in selected],
        "selected_relevance": [float(relevance[i]) for i in selected],
    }
    if len(selected) + sum(reasons.values()) != len(candidate_names):
        raise ValueError("Screening counts do not balance")
    return selected, audit


@dataclass
class Model:
    feature_names: list[str]
    feature_positions: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    estimator: Any
    clip: float
    weight: float = 1.0

    def transform(self, panel: Panel, start: int, stop: int) -> np.ndarray:
        if [panel.names[i] for i in self.feature_positions] != self.feature_names:
            raise ValueError("Model feature schema differs")
        value = (
            panel.values[start:stop, :, self.feature_positions]
            .reshape(-1, len(self.feature_names))
            .astype(float)
        )
        value = np.where(np.isnan(value), self.medians, value)
        z = np.clip((value - self.means[0]) / self.scales, -self.clip, self.clip) - self.means[1]
        if not np.isfinite(z).all():
            raise ValueError("Invalid model inputs")
        return z

    def residual(self, panel: Panel, start: int, stop: int) -> np.ndarray:
        if not self.feature_names:
            return np.zeros((stop - start, len(panel.targets)))
        z = self.transform(panel, start, stop)
        value = (
            z @ self.coefficient + self.intercept
            if self.estimator is None
            else self.estimator.predict(z)
        )
        return np.asarray(value).reshape(stop - start, len(panel.targets))

    def predict(
        self, panel: Panel, start: int, stop: int, weight: float | None = None
    ) -> pd.DataFrame:
        w = self.weight if weight is None else weight
        values = self.target_mean + w * self.target_scale * self.residual(panel, start, stop)
        return pd.DataFrame(values, index=panel.dates[start:stop], columns=panel.targets)


def fit(
    stats: Statistics,
    selected: list[int],
    panel: Panel,
    y: pd.DataFrame,
    algorithm: str,
    config: dict,
) -> Model:
    model = Model(
        [stats.names[i] for i in selected],
        stats.positions[selected],
        stats.medians[selected],
        stats.means[:, selected],
        stats.scales[selected],
        stats.target_mean,
        stats.target_scale,
        np.zeros(len(selected)),
        stats.outcome_mean,
        None,
        config["standardized_clip"],
    )
    if not selected:
        return model
    if algorithm == "ridge":
        gram = stats.gram[np.ix_(selected, selected)]
        model.coefficient = np.linalg.solve(
            gram + config["ridge_penalty"] * np.eye(len(selected)), stats.rhs[selected]
        )
    elif algorithm == "histogram":
        z = model.transform(panel, stats.train_start, stats.train_stop)
        labels = y.iloc[stats.train_start : stats.train_stop].to_numpy()
        mask = np.isfinite(labels).ravel()
        outcome = np.clip(
            (labels - stats.target_mean) / stats.target_scale, -model.clip, model.clip
        ).ravel()[mask]
        counts = np.isfinite(labels).sum(axis=1)
        weights = np.repeat(
            np.divide(1.0, counts, out=np.zeros(len(counts)), where=counts > 0), len(panel.targets)
        )[mask]
        weights *= len(weights) / weights.sum()
        estimator = HistGradientBoostingRegressor(
            max_iter=config["histogram_iterations"],
            learning_rate=config["histogram_learning_rate"],
            max_leaf_nodes=config["histogram_max_leaf_nodes"],
            min_samples_leaf=config["histogram_min_samples_leaf"],
            l2_regularization=config["histogram_l2_regularization"],
            early_stopping=False,
            random_state=config["seed"],
        )
        estimator.fit(z[mask], outcome, sample_weight=weights)
        model.estimator = estimator
    else:
        raise ValueError("Unknown diagnostic model")
    return model
