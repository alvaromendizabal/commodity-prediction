"""Causality, ordered paths, missingness, and sign-preserving feature contracts."""

import numpy as np
import pandas as pd
import pytest

from commodity_prediction.domain.market_path.features import VARIANTS, path_block, primitive_series


@pytest.fixture
def inputs():
    rng = np.random.default_rng(21)
    x = pd.DataFrame(index=pd.Index(range(80), name="date_id"))
    for asset in ["US_A_adj_", "US_B_adj_"]:
        price = np.exp(4 + rng.normal(0, 0.01, 80).cumsum())
        x[asset + "close"] = price
        x[asset + "open"] = price * 0.99
        x[asset + "high"] = price * 1.02
        x[asset + "low"] = price * 0.98
        x[asset + "volume"] = rng.integers(100, 1000, 80).astype(float)
    x["FX_C"] = 1.1 + rng.normal(0, 0.001, 80)
    pairs = pd.DataFrame({"target": ["target_0", "target_1", "target_2"],
                          "lag": [1, 4, 2],
                          "pair": ["US_A_adj_close - US_B_adj_close", "US_A_adj_close", "FX_C"]})
    return x, pairs


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_prefix_and_future_perturbation(inputs, variant):
    x, pairs = inputs
    a, names, _ = path_block(x, pairs, variant)
    b, other_names, _ = path_block(x.iloc[:50], pairs, variant)
    np.testing.assert_array_equal(a[:50], b)
    assert names == other_names
    changed = x.copy()
    changed.iloc[50:] *= 8
    c, _, _ = path_block(changed, pairs, variant)
    np.testing.assert_array_equal(a[:50], c[:50])


def test_order_is_not_an_average(inputs):
    x, pairs = inputs
    a, names, _ = path_block(x, pairs, "joint_path")
    raw = primitive_series(x, ["US_A_adj_close", "US_B_adj_close", "FX_C"])
    for lag in range(4):
        k = names.index(f"market_path__overnight_lag_{lag}_difference")
        expected = raw["overnight"]["US_A_adj_close"].shift(lag).to_numpy()
        np.testing.assert_array_equal(a[:, 1, k], expected.astype(np.float32))


def test_reversed_pairs(inputs):
    x, pairs = inputs
    a, names, _ = path_block(x, pairs, "joint_path")
    changed = pairs.copy()
    changed.loc[0, "pair"] = "US_B_adj_close - US_A_adj_close"
    b, _, _ = path_block(x, changed, "joint_path")
    for i, name in enumerate(names):
        sign = -1 if name.endswith("difference") else 1
        np.testing.assert_array_equal(a[:, 0, i] * sign, b[:, 0, i])


def test_observed_missing_and_structural_absence_differ(inputs):
    x, pairs = inputs
    x.loc[40, "US_A_adj_open"] = np.nan
    a, names, _ = path_block(x, pairs, "joint_path")
    k = names.index("market_path__intraday_lag_0_difference")
    assert np.isnan(a[40, 1, k])
    assert a[40, 2, k] == 0
    eligibility = names.index("market_path__intraday_applicable_legs")
    assert a[40, 1, eligibility] == 1 and a[40, 2, eligibility] == 0


def test_metadata_and_date_rejection(inputs):
    x, pairs = inputs
    with pytest.raises(ValueError, match="Contiguous"):
        path_block(x.drop(index=20), pairs, "joint_path")
    with pytest.raises(ValueError, match="Undeclared"):
        path_block(x, pairs, "unreviewed")


def test_probe_seals_models_and_reuses_without_refit(inputs, tmp_path, monkeypatch):
    from pathlib import Path
    import json
    from commodity_prediction.data import Fold
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.market_path import run as runner
    from commodity_prediction.runtime import atomic_json, seal_checkpoint

    x, pairs = inputs
    x = pd.concat([x] * 4, ignore_index=True).iloc[:260]
    x.index.name = "date_id"
    source = Path(__file__).resolve().parents[1]
    names = json.loads((source / "reports/domain_study.json").read_text())["inventory"]["names"]
    original = Panel(np.random.default_rng(2).normal(size=(len(x), len(pairs), len(names))).astype(np.float32),
                     x.index.tolist(), pairs.target.tolist(), names, {})
    y = pd.DataFrame(np.random.default_rng(3).normal(size=(len(x), len(pairs))), index=x.index, columns=pairs.target)
    config = json.loads((source / "configs/domain_study.json").read_text())
    config.update(warmup_dates=100, histogram_iterations=2, histogram_min_samples_leaf=16, threads=1)
    atomic_json(tmp_path / "configs/domain_study.json", config)
    atomic_json(tmp_path / "configs/final_evaluation.json", {"evaluated": False, "final_test_start_date_id": 1714})
    for lineage, child, value in [("feature", "features", {}), ("prior", "summary", {"summaries": {"admitted_tail": {"fold_scores": [-100.0]}}})]:
        stage = tmp_path / "artifacts" / lineage / child
        atomic_json(stage / "summary.json", value)
        seal_checkpoint(stage, lineage, ["summary.json"])
    monkeypatch.setattr(runner, "study_lineage", lambda root: ("toy", {"parent_lineage": "prior", "feature_lineage": "feature", "config": {"max_run_seconds": 300}}))
    monkeypatch.setattr(runner, "load_inputs", lambda *args: (x, y, pairs, original, [Fold(0, 150, 155, 175)]))
    probe = runner.run_study(tmp_path, 1)
    assert probe["fits_this_invocation"] == 4
    assert probe["maximum_prediction_replay_error"] == 0
    models = list((tmp_path / "artifacts/toy").rglob("model.joblib"))
    timestamps = {str(p): p.stat().st_mtime_ns for p in models}
    def forbidden(*args, **kwargs):
        raise AssertionError("No fitting statistics should be recomputed for completed stages")
    monkeypatch.setattr(runner, "prepare", forbidden)
    again = runner.run_study(tmp_path, 1)
    assert again["fits_this_invocation"] == 0
    assert timestamps == {str(p): p.stat().st_mtime_ns for p in models}
    models[0].write_bytes(models[0].read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        runner.run_study(tmp_path, 1)
