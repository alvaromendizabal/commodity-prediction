"""Final forecast comparisons and fold-respecting paired uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations


def evaluate(truth: pd.DataFrame, prediction: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    daily = daily_rank_correlations(truth, prediction)
    observed = truth.notna().to_numpy()
    errors = prediction.to_numpy()[observed] - truth.to_numpy()[observed]
    horizons = {}
    for horizon in sorted(pairs.lag.unique()):
        columns = pairs.loc[pairs.lag == horizon, "target"].tolist()
        eligible = truth[columns].notna().sum(axis=1) >= 2
        values = daily_rank_correlations(
            truth.loc[eligible, columns], prediction.loc[eligible, columns]
        )
        horizons[str(horizon)] = {
            "official_metric": correlation_sharpe(values),
            "mean_daily_rank_correlation": float(values.mean()),
            "eligible_dates": int(eligible.sum()),
            "excluded_dates": int((~eligible).sum()),
        }
    return {
        "official_metric": correlation_sharpe(daily),
        "mean_daily_rank_correlation": float(daily.mean()),
        "std_daily_rank_correlation": float(daily.std(ddof=0)),
        "positive_correlation_fraction": float((daily > 0).mean()),
        "daily_rank_correlations": daily.tolist(),
        "date_ids": truth.index.tolist(),
        "mean_absolute_return_error": float(np.abs(errors).mean()),
        "root_mean_squared_return_error": float(np.sqrt(np.square(errors).mean())),
        "horizon_metrics": horizons,
    }


def block_indices(fold_lengths: list[int], block: int, repetitions: int, seed: int) -> np.ndarray:
    """Resample each fold separately; blocks never cross fitted-model boundaries."""
    if block < 1 or repetitions < 1 or not fold_lengths or min(fold_lengths) < 2:
        raise ValueError("Invalid block bootstrap configuration")
    rng = np.random.default_rng(seed)
    samples = []
    offset = 0
    for length in fold_lengths:
        starts = rng.integers(0, length, size=(repetitions, int(np.ceil(length / block))))
        indices = (starts[:, :, None] + np.arange(block)[None, None, :]) % length
        samples.append(indices.reshape(repetitions, -1)[:, :length] + offset)
        offset += length
    return np.concatenate(samples, axis=1)


def metric_draws(daily: np.ndarray, indices: np.ndarray) -> np.ndarray:
    samples = daily[indices]
    deviations = samples.std(axis=1, ddof=0)
    if np.any(deviations <= 1e-12):
        raise ValueError("Degenerate bootstrap correlation distribution")
    return samples.mean(axis=1) / deviations


def compare_predictions(
    daily_by_variant: dict[str, np.ndarray],
    folds: list[int],
    comparisons: list[tuple[str, str]],
    config: dict,
) -> list[dict]:
    """Conditional intervals and simultaneous centered max-error bounds.

    The bounds cover the declared comparisons jointly under the block resampling
    approximation. They condition on the fitted models, not on new feature searches.
    """
    rows = []
    observed = np.asarray(
        [
            correlation_sharpe(daily_by_variant[a]) - correlation_sharpe(daily_by_variant[b])
            for a, b in comparisons
        ]
    )
    for block in config["bootstrap_block_dates"]:
        indices = block_indices(folds, block, config["bootstrap_repetitions"], config["seed"])
        draws = {name: metric_draws(daily, indices) for name, daily in daily_by_variant.items()}
        differences = np.column_stack([draws[a] - draws[b] for a, b in comparisons])
        max_error = np.max(np.abs(differences - observed), axis=1)
        critical = float(np.quantile(max_error, 0.95))
        for i, (variant, reference) in enumerate(comparisons):
            rows.append(
                {
                    "variant": variant,
                    "reference": reference,
                    "block_dates": block,
                    "delta": float(observed[i]),
                    "conditional_95_interval": np.quantile(
                        differences[:, i], [0.025, 0.975]
                    ).tolist(),
                    "simultaneous_95_interval": [
                        float(observed[i] - critical),
                        float(observed[i] + critical),
                    ],
                }
            )
    return rows
