"""Stress tests for the causal sensitivity transformations and fixed-model reuse."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold
from commodity_prediction.domain.attribution.run import fit_stage
from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.experiment import Experiment
from commodity_prediction.domain.model import prepare
from commodity_prediction.domain.robustness.features import (
    Variant,
    experiment_plan,
    transformed_panel,
)
from commodity_prediction.runtime import digest, fingerprint


@pytest.fixture
def panel():
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root / "reports/domain_study.json").read_text())["inventory"]["names"]
    rng = np.random.default_rng(17)
    targets = [f"target_{i}" for i in range(8)]
    values = rng.normal(size=(260, 8, len(names))).astype(np.float32)
    pairs = pd.DataFrame(
        {
            "target": targets,
            "lag": [1, 2, 3, 4] * 2,
            "pair": ["FX_USDJPY", "US_A - FX_USDJPY", "LME_Copper", "JPX_Gold"] * 2,
        }
    )
    return Panel(values, list(range(260)), targets, names, {}), pairs


@pytest.mark.parametrize(
    "mode,delay", [("released_delay_1", 1), ("released_delay_5", 5), ("market_delay_1", 1)]
)
def test_delay_has_exact_boundary_and_does_not_mutate_parent(panel, mode, delay):
    original, pairs = panel
    before = original.values.copy()
    variant = Variant(mode, mode)
    current = transformed_panel(original, pairs, variant)
    name = (
        "released_priors__location_63"
        if mode.startswith("released")
        else "reference__return_lag_0_difference"
    )
    j = original.names.index(name)
    assert np.isnan(current.values[:delay, :, j]).all()
    np.testing.assert_array_equal(current.values[delay:, :, j], before[:-delay, :, j])
    np.testing.assert_array_equal(original.values, before)
    changed = replace(original, values=before.copy())
    changed.values[200:] = 1e5
    later = transformed_panel(changed, pairs, variant)
    np.testing.assert_array_equal(current.values[:200], later.values[:200])


def test_all_interactions_ignore_future_and_conditioning_preserves_missingness(panel):
    original, pairs = panel
    j = original.names.index("released_priors__location_63")
    original.values[100, :, j] = np.nan
    for mode in ["conditional_priors", "mechanism_products"]:
        variant = Variant(mode, mode)
        current = transformed_panel(original, pairs, variant)
        changed = replace(original, values=original.values.copy())
        changed.values[200:] *= 100
        later = transformed_panel(changed, pairs, variant)
        np.testing.assert_array_equal(current.values[:200], later.values[:200])
        assert len(current.names) - len(original.names) == (
            56 if mode == "conditional_priors" else 12
        )
        if mode == "conditional_priors":
            k = current.names.index("conditional_priors__location_63_long_horizon")
            np.testing.assert_array_equal(current.values[100, [0, 1, 4, 5], k], 0)
            assert np.isnan(current.values[100, [2, 3, 6, 7], k]).all()
        else:
            assert np.nanmax(np.abs(current.values[:, :, len(original.names) :])) <= 1


def test_compact_conditional_model_roundtrip_and_no_fit_resume(panel, tmp_path, monkeypatch):
    original, pairs = panel
    variant = next(v for v in experiment_plan() if v.name == "compact_conditional")
    transformed = transformed_panel(original, pairs, variant)
    assert len(transformed.names) == 95
    y = pd.DataFrame(
        original.values[:, :, 0] * 0.01 + np.arange(8) * 0.002,
        index=pd.Index(original.dates, name="date_id"),
        columns=original.targets,
    )
    root = Path(__file__).resolve().parents[1]
    config = {
        **json.loads((root / "configs/domain_study.json").read_text()),
        "warmup_dates": 20,
        "histogram_iterations": 3,
        "histogram_min_samples_leaf": 16,
    }
    fold = Fold(0, 200, 205, 235)
    with threadpool_limits(limits=1):
        stats = prepare(transformed, y, fold.train_stop, config)
        result = fit_stage(
            transformed, y, pairs, fold, Experiment(variant.name), config, stats, tmp_path, "toy"
        )
    audit = result["selection"]
    assert (
        audit["candidate_templates"]
        == audit["retained_templates"] + audit["rejected_templates"]
        == 95
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A valid checkpoint must not refit")

    monkeypatch.setattr("commodity_prediction.domain.attribution.run.fit", forbidden)
    assert (
        fit_stage(
            transformed, y, pairs, fold, Experiment(variant.name), config, None, tmp_path, "toy"
        )
        == result
    )


def test_parent_fingerprints_and_fixed_experiment_budget_are_preserved():
    root = Path(__file__).resolve().parents[1]
    for name in ["domain_lineage", "tree_attribution_lineage"]:
        evidence = json.loads((root / f"reports/{name}.json").read_text())
        for path, expected in evidence["files"].items():
            assert digest(root / path) == expected
        assert "/robustness/" not in str(evidence["files"])
    attr = json.loads((root / "reports/tree_attribution_lineage.json").read_text())
    config = json.loads((root / "configs/domain_robustness.json").read_text())
    assert config["parent_lineage"] == fingerprint(attr)
    assert len(experiment_plan()) * 3 == config["max_new_fits"] == 36


def test_subgroup_coverage_excludes_missing_dates_and_single_target_horizons():
    from commodity_prediction.domain.robustness.reporting.run import subgroup_score

    truth = pd.DataFrame({"a": [1.0, 3.0, np.nan, 4.0], "b": [2.0, 1.0, np.nan, 3.0]})
    prediction = pd.DataFrame({"a": [1.0, 3.0, 2.0, 2.0], "b": [2.0, 1.0, 1.0, 3.0]})
    result = subgroup_score(truth, prediction)
    assert result["eligible_dates"] == 3 and result["excluded_dates"] == 1
    assert result["official_metric"] == pytest.approx(
        np.mean([1.0, 1.0, -1.0]) / np.std([1.0, 1.0, -1.0])
    )
    unavailable = subgroup_score(truth[["a"]], prediction[["a"]])
    assert unavailable["official_metric"] is None
    assert unavailable["eligible_dates"] == 0 and unavailable["excluded_dates"] == 4


def test_subgroup_constant_predictions_are_explicitly_undefined():
    from commodity_prediction.domain.robustness.reporting.run import subgroup_score

    truth = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    result = subgroup_score(truth, truth * 0)
    assert result["official_metric"] is None
    assert result["undefined_reason"] == "constant_daily_target_or_prediction"
