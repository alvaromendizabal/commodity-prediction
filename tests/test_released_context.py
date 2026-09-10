"""Release-time safety, signed relationships and interrupted-study recovery."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold
from commodity_prediction.domain.catalog import Panel
from commodity_prediction.domain.released_context.features import (
    peer_weights,
    released_values,
    short_blocks,
)
from commodity_prediction.domain.released_context.run import run_study
from commodity_prediction.runtime import atomic_json, seal_checkpoint


@pytest.fixture
def inputs():
    pairs = pd.DataFrame(
        {
            "target": [f"target_{i}" for i in range(8)],
            "lag": [1, 4, 2, 3, 1, 4, 2, 3],
            "pair": [
                "LME_A - LME_B",
                "LME_B - LME_A",
                "LME_A - LME_C",
                "LME_B",
                "LME_C",
                "FX_D",
                "LME_C - LME_A",
                "US_E",
            ],
        }
    )
    y = pd.DataFrame(
        np.random.default_rng(76).normal(size=(260, 8)),
        index=pd.Index(range(260), name="date_id"),
        columns=pairs.target,
    )
    return y, pairs


@pytest.mark.parametrize("target", range(8))
def test_unreleased_labels_cannot_change_any_context(inputs, target):
    y, pairs = inputs
    changed = y.copy()
    changed.iloc[100 - int(pairs.lag.iloc[target]) :, target] += 1e5
    current, later = short_blocks(y, pairs), short_blocks(changed, pairs)
    for a, b in zip(current, later, strict=True):
        for name in a:
            np.testing.assert_array_equal(a[name][:101], b[name][:101])
    known = released_values(y, pairs)
    assert known.iloc[100, target] == y.iloc[99 - int(pairs.lag.iloc[target]), target]


def test_prefix_missingness_and_no_own_target_in_peers(inputs):
    y, pairs = inputs
    y.iloc[10, 0] = -999999
    assert pd.isna(released_values(y, pairs).iloc[12, 0])
    current, prefix = short_blocks(y, pairs), short_blocks(y.iloc[:160], pairs)
    for a, b in zip(current, prefix, strict=True):
        for name in a:
            np.testing.assert_array_equal(a[name][:160], b[name])
    changed = y.copy()
    changed.iloc[:, 0] *= 1000
    peers = short_blocks(changed, pairs)[1]
    for name in peers:
        np.testing.assert_array_equal(current[1][name][:, 0], peers[name][:, 0])
    for weights in peer_weights(pairs).values():
        assert np.all(np.diag(weights) == 0)
    assert np.isnan(current[1]["shared_asset_latest"][:, 5]).all()
    assert (current[1]["shared_asset_coverage"][:, 5] == 0).all()


def test_reversing_pair_orientation_preserves_other_targets(inputs):
    y, pairs = inputs
    current = short_blocks(y, pairs)[1]
    reversed_pairs = pairs.copy()
    reversed_pairs.loc[0, "pair"] = "LME_B - LME_A"
    changed = y.copy()
    changed.iloc[:, 0] *= -1
    reverse = short_blocks(changed, reversed_pairs)[1]
    for name in current:
        np.testing.assert_allclose(current[name][:, 1:], reverse[name][:, 1:], equal_nan=True)
        sign = 1 if name.endswith(("dispersion", "coverage")) else -1
        np.testing.assert_allclose(current[name][:, 0] * sign, reverse[name][:, 0], equal_nan=True)


def test_partial_checkpoint_and_complete_no_refit_resume(inputs, tmp_path, monkeypatch):
    y, pairs = inputs
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root / "reports/domain_study.json").read_text())["inventory"]["names"]
    original = Panel(
        np.random.default_rng(3).normal(size=(260, 8, len(names))).astype(np.float32),
        y.index.tolist(),
        y.columns.tolist(),
        names,
        {},
    )
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
    folds = [Fold(i, 150 + i * 30, 155 + i * 30, 175 + i * 30) for i in range(3)]
    summary = {
        "daily_rank_correlations": np.random.default_rng(8).normal(0.05, 0.1, 60).tolist(),
        "official_metric": 0.2,
    }
    parent = {
        "joint_comparisons": [],
        "joint_comparison_count": 0,
        "summaries": {
            n: summary
            for n in ["historical_mean", "admitted_tail", "screened_tail", "tail_states_and_priors"]
        },
    }
    for directory, filename, value, lineage in [
        ("parent/summary", "summary.json", parent, "parent"),
        ("feature/features", "toy.json", {}, "feature"),
    ]:
        stage = tmp_path / "artifacts" / directory
        atomic_json(stage / filename, value)
        seal_checkpoint(stage, lineage, [filename])
    module = "commodity_prediction.domain.released_context.run."
    monkeypatch.setattr(
        module + "study_lineage",
        lambda root: (
            "context",
            {
                "parent_lineage": "parent",
                "feature_lineage": "feature",
                "config": {"max_new_fits": 9, "max_run_seconds": 300},
            },
        ),
    )
    monkeypatch.setattr(module + "load_data", lambda root: (pd.DataFrame(index=y.index), y, pairs))
    monkeypatch.setattr(module + "load_panel", lambda path: original)
    monkeypatch.setattr(module + "study_folds", lambda *args: (folds, 260))
    monkeypatch.setattr(module + "joint_history", lambda *args: {})
    with threadpool_limits(limits=1):
        probe = run_study(tmp_path, fold_limit=1)
        assert len(probe["results"]) == 3
        stages = sorted((tmp_path / "artifacts/context").glob("fold_0/*/model.joblib"))
        timestamps = {str(p): p.stat().st_mtime_ns for p in stages}
        report = run_study(tmp_path)
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 9
    assert report["maximum_prediction_replay_error"] == 0
    assert timestamps == {str(p): p.stat().st_mtime_ns for p in stages}
    assert len(list((tmp_path / "artifacts/context").rglob("manifest.json"))) == 10

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete resume must not load data or refit")

    monkeypatch.setattr(module + "load_data", forbidden)
    assert run_study(tmp_path) == report
    stages[0].write_bytes(stages[0].read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run_study(tmp_path)
