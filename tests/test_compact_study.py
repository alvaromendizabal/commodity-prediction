"""Causal transformations, fair feature admission, exact resume, and preserved lineage."""

import importlib.metadata
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold
from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.compact.features import Variant, build_panel, experiment_plan
from commodity_prediction.domain.compact.run import compact_lineage, run_study
from commodity_prediction.domain.model import prepare, select
from commodity_prediction.runtime import atomic_json, digest, fingerprint, seal_checkpoint


@pytest.fixture
def panel():
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root / "reports/domain_study.json").read_text())["inventory"]["names"]
    rng = np.random.default_rng(29)
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


@pytest.mark.parametrize("variant", experiment_plan(), ids=lambda v: v.name)
def test_all_variants_are_causal_and_do_not_mutate_parent(panel, variant):
    original, pairs = panel
    snapshot = original.values.copy()
    current = build_panel(original, pairs, variant)
    changed = replace(original, values=snapshot.copy())
    changed.values[200:] *= 1000
    later = build_panel(changed, pairs, variant)
    np.testing.assert_array_equal(current.values[:200], later.values[:200])
    np.testing.assert_array_equal(original.values, snapshot)
    assert current.targets == pairs.target.tolist()


def test_compound_delay_shifts_dynamic_features_once_and_keeps_metadata(panel):
    original, pairs = panel
    current = build_panel(original, pairs, Variant("both", prior_delay=1, market_delay=1))
    for name in ["released_priors__location_63", "reference__return_lag_0_difference"]:
        i, j = current.names.index(name), original.names.index(name)
        np.testing.assert_array_equal(current.values[1:, :, i], original.values[:-1, :, j])
        assert np.isnan(current.values[0, :, i]).all()
    name = "reference__horizon"
    np.testing.assert_array_equal(
        current.values[:, :, current.names.index(name)],
        original.values[:, :, original.names.index(name)],
    )


def test_fx_gate_preserves_structural_absence_and_applicable_missingness(panel):
    original, pairs = panel
    name = "tail_risk__robust_shock_difference"
    original.values[100, :, original.names.index(name)] = np.nan
    result = build_panel(original, pairs, Variant("fx", admit=True, interactions="fx_states"))
    assert len(result.names) == 123
    j = result.names.index("fx_states__" + name.replace("__", "_"))
    np.testing.assert_array_equal(result.values[100, [2, 3, 6, 7], j], 0)
    assert np.isnan(result.values[100, [0, 1, 4, 5], j]).all()
    np.testing.assert_allclose(
        result.values[99, 1, j], original.values[99, 1, original.names.index(name)] * 0.5
    )


def test_admission_retains_low_relevance_products_and_statistics_ignore_future(panel):
    original, pairs = panel
    variant = Variant("products", admit=True, interactions="products")
    current = build_panel(original, pairs, variant)
    root = Path(__file__).resolve().parents[1]
    config = variant.settings(
        {**json.loads((root / "configs/domain_study.json").read_text()), "warmup_dates": 20},
        current,
    )
    y = pd.DataFrame(np.random.default_rng(4).normal(size=(260, 8)), columns=original.targets)
    with threadpool_limits(limits=1):
        stats = prepare(current, y, 200, config)
        stats.relevance[:] = 0
        selected, audit = select(stats, current.names, config, False)
        assert len(selected) == len(current.names) == 93
        assert sum(n.startswith("mechanism_products__") for n in audit["selected_names"]) == 12
        changed_y = y.copy()
        changed_y.iloc[200:] *= 1e6
        changed_panel = replace(current, values=current.values.copy())
        changed_panel.values[200:] *= 1e6
        later = prepare(changed_panel, changed_y, 200, config)
    for attribute in ["medians", "means", "scales", "gram", "rhs", "target_mean", "target_scale"]:
        np.testing.assert_array_equal(getattr(stats, attribute), getattr(later, attribute))


def test_study_runs_all_stages_and_complete_resume_never_loads_data(panel, tmp_path, monkeypatch):
    original, pairs = panel
    root = Path(__file__).resolve().parents[1]
    config = {
        **json.loads((root / "configs/domain_study.json").read_text()),
        "warmup_dates": 20,
        "histogram_iterations": 2,
        "histogram_min_samples_leaf": 16,
        "bootstrap_repetitions": 30,
        "threads": 1,
    }
    for name, value in [("domain_study", config), ("research", {}), ("feature_study", {})]:
        atomic_json(tmp_path / "configs" / (name + ".json"), value)
    y = pd.DataFrame(
        np.random.default_rng(13).normal(size=(260, 8)),
        index=pd.Index(original.dates, name="date_id"),
        columns=original.targets,
    )
    folds = [Fold(i, 140 + i * 30, 145 + i * 30, 165 + i * 30) for i in range(3)]
    daily = np.random.default_rng(8).normal(0.05, 0.1, 60).tolist()
    summary = {
        "daily_rank_correlations": daily,
        "official_metric": float(np.mean(daily) / np.std(daily)),
    }
    parent = {
        "parent_lineage": "attribution",
        "joint_comparisons": [],
        "summaries": {
            n: summary
            for n in ["historical_mean", "compact_control", "all_control", "compact_risk_freshness"]
        },
    }
    for lineage in ["parent", "feature", "attribution"]:
        stage = tmp_path / "artifacts" / lineage / "summary"
        atomic_json(stage / "summary.json", parent)
        seal_checkpoint(stage, lineage, ["summary.json"])
    stage = tmp_path / "artifacts/feature/features"
    atomic_json(stage / "toy.json", {})
    seal_checkpoint(stage, "feature", ["toy.json"])
    monkeypatch.setattr(
        "commodity_prediction.domain.compact.run.compact_lineage",
        lambda root: (
            "compact",
            {
                "parent_lineage": "parent",
                "feature_lineage": "feature",
                "config": {"max_new_fits": 36},
            },
        ),
    )
    monkeypatch.setattr(
        "commodity_prediction.domain.compact.run.load_data",
        lambda root: (pd.DataFrame(index=original.dates), y, pairs),
    )
    monkeypatch.setattr("commodity_prediction.domain.compact.run.load_panel", lambda path: original)
    monkeypatch.setattr(
        "commodity_prediction.domain.compact.run.study_folds", lambda *args: (folds, 260)
    )
    report = run_study(tmp_path)
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 36
    assert report["maximum_prediction_replay_error"] == 0
    assert len(list((tmp_path / "artifacts/compact").rglob("manifest.json"))) == 37

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete verified study must not load data or refit")

    monkeypatch.setattr("commodity_prediction.domain.compact.run.load_data", forbidden)
    assert run_study(tmp_path) == report
    model = tmp_path / "artifacts/compact/fold_0/screened_tail/model.joblib"
    model.write_bytes(model.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Checkpoint integrity failure"):
        run_study(tmp_path)


def test_all_parent_lineages_remain_frozen(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    initial = json.loads((root / "reports/lineage.json").read_text())
    assert initial["config"] == json.loads((root / "configs/research.json").read_text())
    for name, expected in initial["environment"].items():
        assert importlib.metadata.version(name) == expected
    for path, expected in initial["files"].items():
        if path.startswith("data/raw/") and not (root / path).exists():
            continue  # Public CI has the recorded digest, not the restricted bytes.
        assert digest(root / path) == expected
    initial_id = fingerprint(initial)
    assert initial_id == json.loads((root / "reports/research.json").read_text())["lineage"]
    # Inject only the already verified archive at the private-data boundary. All
    # downstream source/config/experiment fingerprints are recomputed normally.
    monkeypatch.setattr(
        "commodity_prediction.studies.run.lineage_for", lambda *_: (initial_id, initial)
    )
    lineage, evidence = compact_lineage(root)
    assert len(lineage) == 64 and len(evidence["experiments"]) == 12
    for filename in ["domain_lineage", "tree_attribution_lineage", "domain_robustness_lineage"]:
        report = json.loads((root / f"reports/{filename}.json").read_text())
        for nested in [report, report.get("fitting_evidence", {"files": {}})]:
            for path, expected in nested["files"].items():
                assert digest(root / path) == expected


def test_runtime_still_rejects_missing_private_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "src", tmp_path / "src")
    shutil.copytree(root / "configs", tmp_path / "configs")
    with pytest.raises(ValueError, match="Initial source/data/configuration lineage changed"):
        compact_lineage(tmp_path)
