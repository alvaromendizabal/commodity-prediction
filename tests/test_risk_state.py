"""Causal features, aligned delays, main-effect controls and immutable study resume."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold
from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.risk_state.features import Variant, build_panel, experiment_plan
from commodity_prediction.domain.risk_state.run import run_study
from commodity_prediction.runtime import atomic_json, seal_checkpoint


@pytest.fixture
def panel():
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root / "reports/domain_study.json").read_text())["inventory"]["names"]
    rng = np.random.default_rng(382)
    targets = [f"target_{i}" for i in range(8)]
    values = rng.normal(size=(260, 8, len(names))).astype(np.float32)
    for i, name in enumerate(names):
        if name.startswith("released_priors__risk_"):
            values[:, :, i] = np.exp(values[:, :, i])
    pairs = pd.DataFrame(
        {
            "target": targets,
            "lag": [1, 2, 3, 4] * 2,
            "pair": ["FX_USDJPY", "US_A - FX_USDJPY", "LME_Copper", "JPX_Gold"] * 2,
        }
    )
    return Panel(values, list(range(260)), targets, names, {}), pairs


@pytest.mark.parametrize("variant", experiment_plan(), ids=lambda v: v.name)
def test_risk_features_are_causal_and_preserve_parent(panel, variant):
    original, pairs = panel
    snapshot = original.values.copy()
    current = build_panel(original, pairs, variant)
    changed = replace(original, values=snapshot.copy())
    changed.values[200:] *= 100
    later = build_panel(changed, pairs, variant)
    np.testing.assert_array_equal(current.values[:200], later.values[:200])
    np.testing.assert_array_equal(original.values, snapshot)
    shorter = replace(original, values=snapshot[:200], dates=original.dates[:200])
    np.testing.assert_array_equal(current.values[:200], build_panel(shorter, pairs, variant).values)


def test_all_products_include_main_effects_and_are_bounded(panel):
    original, pairs = panel
    current = build_panel(original, pairs, Variant("all"))
    names = current.names
    assert len(names) == 103
    assert sum(n.startswith("risk_state__") for n in names) == 6
    assert sum(n.startswith("scaled_prior__") for n in names) == 4
    products = [i for i, n in enumerate(names) if n.startswith("state_prior_product__")]
    assert len(products) == 24
    assert np.nanmax(np.abs(current.values[:, :, products])) <= 1
    with pytest.raises(ValueError, match="main effects"):
        build_panel(original, pairs, Variant("bad", states=False))


def test_extra_delay_shifts_states_and_priors_but_not_tail_inputs(panel):
    original, pairs = panel
    current = build_panel(original, pairs, Variant("base"))
    delayed = build_panel(original, pairs, Variant("delay", prior_delay=1))
    for i, name in enumerate(current.names):
        if name.startswith(
            ("released_priors__", "risk_state__", "scaled_prior__", "state_prior_product__")
        ):
            np.testing.assert_array_equal(delayed.values[1:, :, i], current.values[:-1, :, i])
        else:
            np.testing.assert_array_equal(delayed.values[:, :, i], current.values[:, :, i])


def test_invalid_risk_stays_missing_and_never_creates_infinite_features(panel):
    original, pairs = panel
    original.values[:, 0, original.names.index("released_priors__risk_63")] = 0
    current = build_panel(original, pairs, Variant("missing"))
    assert not np.isinf(current.values).any()
    i = current.names.index("scaled_prior__location_63_per_risk")
    assert np.isnan(current.values[:, 0, i]).all()


def test_full_study_reuses_valid_completed_work_and_detects_tampering(panel, tmp_path, monkeypatch):
    original, pairs = panel
    root = Path(__file__).resolve().parents[1]
    config = {
        **json.loads((root / "configs/domain_study.json").read_text()),
        "warmup_dates": 100,
        "histogram_iterations": 2,
        "histogram_min_samples_leaf": 16,
        "bootstrap_repetitions": 30,
        "threads": 1,
    }
    for name, value in [("domain_study", config), ("research", {}), ("feature_study", {})]:
        atomic_json(tmp_path / "configs" / (name + ".json"), value)
    y = pd.DataFrame(
        np.random.default_rng(6).normal(size=(260, 8)),
        index=pd.Index(original.dates, name="date_id"),
        columns=original.targets,
    )
    folds = [Fold(i, 150 + i * 30, 155 + i * 30, 175 + i * 30) for i in range(3)]
    daily = np.random.default_rng(8).normal(0.05, 0.1, 60).tolist()
    summary = {
        "daily_rank_correlations": daily,
        "official_metric": float(np.mean(daily) / np.std(daily)),
    }
    parent = {
        "joint_comparisons": [],
        "joint_comparison_count": 0,
        "summaries": {
            n: summary
            for n in [
                "historical_mean",
                "admitted_base",
                "admitted_tail",
                "admitted_joint",
                "screened_tail",
            ]
        },
    }
    stage = tmp_path / "artifacts/parent/summary"
    atomic_json(stage / "summary.json", parent)
    seal_checkpoint(stage, "parent", ["summary.json"])
    stage = tmp_path / "artifacts/feature/features"
    atomic_json(stage / "toy.json", {})
    seal_checkpoint(stage, "feature", ["toy.json"])
    module = "commodity_prediction.domain.risk_state.run."
    monkeypatch.setattr(
        module + "study_lineage",
        lambda root: (
            "risk_state",
            {
                "parent_lineage": "parent",
                "feature_lineage": "feature",
                "config": {"max_new_fits": 24, "max_run_seconds": 1200},
            },
        ),
    )
    monkeypatch.setattr(
        module + "load_data", lambda root: (pd.DataFrame(index=original.dates), y, pairs)
    )
    monkeypatch.setattr(module + "load_panel", lambda path: original)
    monkeypatch.setattr(module + "study_folds", lambda *args: (folds, 260))
    monkeypatch.setattr(module + "joint_history", lambda *args: {})
    persisted = []
    with threadpool_limits(limits=1):
        report = run_study(tmp_path, checkpoint_hook=lambda p: persisted.append(p))
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 24
    assert report["maximum_prediction_replay_error"] == 0
    assert len(persisted) == 25

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete resume must not load data or refit")

    monkeypatch.setattr(module + "load_data", forbidden)
    assert run_study(tmp_path) == report
    model = tmp_path / "artifacts/risk_state/fold_0/tail_states/model.joblib"
    model.write_bytes(model.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        run_study(tmp_path)
