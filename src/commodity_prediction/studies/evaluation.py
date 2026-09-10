"""Official-metric comparisons, fold-respecting uncertainty, and group diagnostics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations

from .models import ModelBundle
from .representation import FeatureInfo


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


def block_permutation(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    chunks = [np.arange(start, min(length, start + block)) for start in range(0, length, block)]
    return np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])


def group_permutations(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    metadata: dict[str, FeatureInfo],
    config: dict,
) -> dict:
    """Jointly permute each family's date blocks; keep cross-target alignment intact."""
    z = bundle.standardize(frame)
    original = bundle.predict_standardized(z)
    baseline = correlation_sharpe(
        daily_rank_correlations(
            truth, pd.DataFrame(original, index=truth.index, columns=truth.columns)
        )
    )
    selected_families = sorted(
        {
            metadata[bundle.feature_names[i]].family
            for model in bundle.models
            for i in model.selected
        }
    )
    result = {}
    rng = np.random.default_rng(config["seed"])
    for family in selected_families:
        drops = []
        for _ in range(config["permutation_repetitions"]):
            order = block_permutation(len(frame), config["permutation_block_dates"], rng)
            output = []
            for model in bundle.models:
                if model.estimator is None:
                    output.append(model.predict(z))
                    continue
                local = z[:, model.selected].copy()
                affected = [
                    j
                    for j, idx in enumerate(model.selected)
                    if metadata[bundle.feature_names[idx]].family == family
                ]
                if affected:
                    local[:, affected] = local[order][:, affected]
                output.append(model.estimator.predict(local))
            changed = pd.DataFrame(
                bundle.project(np.column_stack(output)), index=truth.index, columns=truth.columns
            )
            drops.append(baseline - correlation_sharpe(daily_rank_correlations(truth, changed)))
        result[family] = {
            "metric_drop_repetitions": drops,
            "mean_metric_drop": float(np.mean(drops)),
            "std_metric_drop": float(np.std(drops, ddof=0)),
        }
    return result


def selection_summary(audits: list[dict], metadata: dict[str, FeatureInfo]) -> dict:
    chosen = {name for audit in audits for name in audit["selected"]}
    retained = [audit["retained_count"] for audit in audits]
    return {
        "output_models": len(audits),
        "candidate_output_assignments": sum(audit["candidate_count"] for audit in audits),
        "retained_output_assignments": sum(retained),
        "rejected_output_assignments": sum(audit["rejected_count"] for audit in audits),
        "unique_retained_features": len(chosen),
        "retained_per_output_min": min(retained),
        "retained_per_output_median": float(np.median(retained)),
        "retained_per_output_max": max(retained),
        "retained_family_assignments": dict(
            Counter(metadata[name].family for audit in audits for name in audit["selected"])
        ),
        "rejection_reasons": dict(
            sum((Counter(audit["rejection_reasons"]) for audit in audits), Counter())
        ),
        "unique_features_by_family": dict(Counter(metadata[name].family for name in chosen)),
    }


def selection_stability(audits_by_fold: list[list[dict]]) -> dict:
    values = []
    for first, second in zip(audits_by_fold[:-1], audits_by_fold[1:], strict=True):
        left = {audit["output"]: set(audit["selected"]) for audit in first}
        right = {audit["output"]: set(audit["selected"]) for audit in second}
        for name in left.keys() & right.keys():
            union = left[name] | right[name]
            if union:
                values.append(len(left[name] & right[name]) / len(union))
    return {
        "adjacent_fold_output_comparisons": len(values),
        "median_jaccard": float(np.median(values)) if values else None,
        "mean_jaccard": float(np.mean(values)) if values else None,
    }
