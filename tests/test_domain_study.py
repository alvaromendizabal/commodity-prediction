"""Causal timing, economic identities, weighted fitting, and exact resumability."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from commodity_prediction.data import Fold, reconstruct_targets
from commodity_prediction.domain.catalog import Builder, Panel
from commodity_prediction.domain.experiment import Experiment, choose_weight, inner_folds
from commodity_prediction.domain.features import FAMILIES, build_panel, released_features
from commodity_prediction.domain.model import fit, graph_mean, prepare, select
from commodity_prediction.domain.relationships import contract_features, currency_features
from commodity_prediction.domain.run import run_experiments
from commodity_prediction.runtime import RunLog, verify_checkpoint


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return {
        **json.loads((root / "configs/domain_study.json").read_text()),
        "warmup_dates": 20,
        "inner_validation_dates": 40,
        "histogram_iterations": 3,
        "histogram_min_samples_leaf": 16,
        "max_features": 4,
    }


def example_panel():
    rng = np.random.default_rng(44)
    values = rng.normal(size=(260, 8, 4)).astype(np.float32)
    targets = [f"target_{i}" for i in range(8)]
    panel = Panel(
        values,
        list(range(260)),
        targets,
        ["reference__a", "trend_shape__b", "tail_risk__c", "reference__d"],
        {},
    )
    y = pd.DataFrame(
        values[:, :, 0] * 0.008 + rng.normal(0, 0.02, (260, 8)) + np.arange(8) * 0.001,
        index=pd.Index(range(260), name="date_id"),
        columns=targets,
    )
    y.iloc[30:70:3, 0] = np.nan
    pairs = pd.DataFrame(
        {
            "target": targets,
            "lag": [1] * 4 + [2] * 4,
            "pair": ["FX_USDJPY", "FX_AUDUSD", "FX_USDJPY - FX_AUDUSD", "FX_AUDUSD - FX_USDJPY"]
            * 2,
        }
    )
    return panel, y, pairs


def domain_market():
    rng = np.random.default_rng(23)
    assets = [
        "FX_USDJPY",
        "FX_AUDUSD",
        "FX_AUDJPY",
        "US_Stock_VT_adj_close",
        "US_Stock_GLD_adj_close",
        "JPX_Gold_Standard_Futures_Close",
        "JPX_Gold_Mini_Futures_Close",
    ]
    x = pd.DataFrame(
        np.exp(rng.normal(0, 0.01, (340, len(assets))).cumsum(axis=0)),
        columns=assets,
        index=pd.Index(range(340), name="date_id"),
    )
    for a in ["US_Stock_VT_adj_close", "JPX_Gold_Standard_Futures_Close"]:
        us = a.startswith("US")
        stem = a.removesuffix("close" if us else "Close")
        for suffix, multiple in [
            ("open" if us else "Open", 0.999),
            ("high" if us else "High", 1.03),
            ("low" if us else "Low", 0.97),
        ]:
            x[stem + suffix] = x[a] * multiple
        x[stem + ("volume" if us else "Volume")] = rng.uniform(100, 10000, len(x))
        if not us:
            x[stem + "open_interest"] = rng.uniform(10000, 30000, len(x))
            x[stem + "settlement_price"] = x[a] * 1.001
    combinations = [
        "FX_USDJPY - FX_AUDUSD",
        "US_Stock_VT_adj_close - US_Stock_GLD_adj_close",
        "JPX_Gold_Standard_Futures_Close - US_Stock_GLD_adj_close",
        "FX_AUDUSD",
    ]
    pairs = pd.DataFrame(
        [
            {"target": f"target_{4 * (h - 1) + i}", "lag": h, "pair": p}
            for h in range(1, 5)
            for i, p in enumerate(combinations)
        ]
    )
    return x, reconstruct_targets(x, pairs), pairs


def test_every_domain_family_ignores_future_and_unreleased_values(tmp_path):
    x, y, pairs = domain_market()
    original = build_panel(x, y, pairs, RunLog(tmp_path / "a.jsonl"))
    cutoff = 250
    changed_x, changed_y = x.copy(), y.copy()
    changed_x.loc[cutoff + 1 :] *= 10
    for row in pairs.itertuples(index=False):
        changed_y.loc[cutoff - row.lag :, row.target] = -100
    changed = build_panel(changed_x, changed_y, pairs, RunLog(tmp_path / "b.jsonl"))
    assert set(n.split("__", 1)[0] for n in original.names) == {"reference", *FAMILIES}
    np.testing.assert_allclose(
        original.values[: cutoff + 1], changed.values[: cutoff + 1], rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_released_pool_moves_exactly_at_release(horizon):
    y = pd.DataFrame({"target_0": np.sin(np.arange(180))})
    pairs = pd.DataFrame({"target": ["target_0"], "lag": [horizon], "pair": ["FX_USDJPY"]})
    first, second = Builder(y.index, pairs), Builder(y.index, pairs)
    released_features(first, y)
    y.loc[100, "target_0"] = 100
    released_features(second, y)
    a, b = first.finish(), second.finish()
    release = 100 + horizon + 1
    np.testing.assert_allclose(
        a.values[:release], b.values[:release], rtol=0, atol=0, equal_nan=True
    )
    column = a.names.index("released_priors__location_63")
    assert b.values[release, 0, column] > a.values[release, 0, column]


def test_currency_triangle_and_yen_conversion_have_correct_sign():
    t = np.arange(150)
    x = pd.DataFrame(
        {"FX_USDJPY": np.exp(0.01 * np.sin(t / 7)), "FX_AUDUSD": np.exp(0.02 * np.cos(t / 11))}
    )
    x["FX_AUDJPY"] = x.FX_AUDUSD * x.FX_USDJPY
    x["JPX_Gold_Standard_Futures_Close"] = np.exp(0.003 * t) * x.FX_USDJPY
    pairs = pd.DataFrame(
        {
            "target": ["a", "b"],
            "lag": [1, 1],
            "pair": ["FX_AUDJPY", "JPX_Gold_Standard_Futures_Close"],
        }
    )
    builder = Builder(x.index, pairs)
    currency_features(builder, x)
    contract_features(builder, x, np.log(x[["JPX_Gold_Standard_Futures_Close"]]))
    panel = builder.finish()
    residual = panel.names.index("currency_graph__cycle_residual_difference")
    conversion = panel.names.index("contract_basis__usd_translated_return_difference")
    np.testing.assert_allclose(panel.values[1:, 0, residual], 0, atol=1e-12)
    np.testing.assert_allclose(panel.values[1:, 1, conversion], 0.003, atol=1e-9)


def test_unsupported_leg_is_structural_zero_but_observed_missingness_survives():
    pairs = pd.DataFrame(
        {"target": ["a", "b", "c"], "lag": [1, 1, 1], "pair": ["A - B", "B - A", "A"]}
    )
    b = Builder(pd.RangeIndex(2), pairs)
    b.asset("trading_activity", "volume", pd.DataFrame({"A": [3, np.nan]}))
    panel = b.finish()
    position = panel.names.index("trading_activity__volume_difference")
    np.testing.assert_array_equal(panel.values[0, :, position], [3, -3, 3])
    assert np.isnan(panel.values[1, :, position]).all()


def test_future_mutation_cannot_change_statistics_selection_or_fitted_model(config):
    panel, y, _ = example_panel()
    a = prepare(panel, y, 200, config)
    changed, changed_y = replace(panel, values=panel.values.copy()), y.copy()
    changed.values[200:] = 1e6
    changed_y.iloc[200:] = -1e6
    b = prepare(changed, changed_y, 200, config)
    for field in [
        "medians",
        "means",
        "scales",
        "gram",
        "rhs",
        "relevance",
        "stable_relevance",
        "target_mean",
        "target_scale",
    ]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert select(a, panel.names, config, False) == select(b, panel.names, config, False)


def test_weighted_ridge_matches_independent_solver_and_panel_axis_order(config):
    panel, y, _ = example_panel()
    stats = prepare(panel, y, 200, config)
    positions, _ = select(stats, panel.names, config, False)
    model = fit(stats, positions, panel, y, "ridge", config)
    raw = (
        panel.values[20:200][:, :, model.feature_positions]
        .reshape(-1, len(positions))
        .astype(float)
    )
    expected = (
        np.clip((raw - model.means[0]) / model.scales, -model.clip, model.clip) - model.means[1]
    )
    np.testing.assert_array_equal(model.transform(panel, 20, 200), expected)
    labels = y.iloc[20:200].to_numpy()
    mask = np.isfinite(labels).ravel()
    outcome = np.clip(
        (labels - stats.target_mean) / stats.target_scale, -model.clip, model.clip
    ).ravel()[mask]
    weights = np.repeat(1 / np.isfinite(labels).sum(axis=1), len(panel.targets))[mask]
    weights /= weights.sum()
    reference = Ridge(alpha=config["ridge_penalty"], fit_intercept=True, solver="cholesky").fit(
        expected[mask], outcome, sample_weight=weights
    )
    np.testing.assert_allclose(model.coefficient, reference.coef_, atol=1e-12)
    np.testing.assert_allclose(model.intercept, reference.intercept_, atol=1e-12)


def test_graph_control_preserves_pair_and_horizon_identities(config):
    _, y, pairs = example_panel()
    actual = graph_mean(y, pairs, 200)
    assert actual[2] == pytest.approx(actual[0] - actual[1])
    assert actual[3] == pytest.approx(-actual[2])
    np.testing.assert_allclose(actual[4:], actual[:4] * 2)


def test_nested_boundaries_and_zero_weight_tie_break(config):
    folds = inner_folds(200, config)
    assert all(
        f.train_stop - 1 + 5 < f.validation_start and f.validation_stop <= 200 for f in folds
    )
    with pytest.raises(ValueError, match="unsafe"):
        inner_folds(200, {**config, "purge_dates": 4})
    _, y, _ = example_panel()
    records = []
    for f in folds:
        truth = y.iloc[f.validation_start : f.validation_stop]
        mean = y.iloc[: f.train_stop].mean().to_numpy()
        raw = pd.DataFrame(
            np.broadcast_to(mean, truth.shape), index=truth.index, columns=truth.columns
        )
        records.append((truth, raw, mean))
    assert choose_weight(records, config)["selected_weight"] == 0


def test_resumed_experiments_never_refit_or_prepare_and_tampering_fails(
    tmp_path, config, monkeypatch
):
    panel, y, pairs = example_panel()
    folds = [Fold(0, 200, 205, 235)]
    plan = [
        Experiment("pooled_reference", include="reference"),
        Experiment("pooled_all"),
        Experiment("tree_all", algorithm="histogram"),
    ]
    with threadpool_limits(limits=1):
        results = run_experiments(
            panel,
            y,
            pairs,
            folds,
            config,
            tmp_path,
            "toy",
            RunLog(tmp_path / "run.jsonl"),
            lambda _: None,
            plan,
        )
    assert len(results) == 6
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 12
    assert all(verify_checkpoint(p.parent, "toy") for p in manifests)

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume must not refit or preprocess")

    monkeypatch.setattr("commodity_prediction.domain.run.fit", forbidden)
    monkeypatch.setattr("commodity_prediction.domain.run.prepare", forbidden)
    repeated = run_experiments(
        panel,
        y,
        pairs,
        folds,
        config,
        tmp_path,
        "toy",
        RunLog(tmp_path / "resume.jsonl"),
        lambda _: None,
        plan,
    )
    assert repeated == results
    model = tmp_path / "fold_0/outer/pooled_all/model.joblib"
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run_experiments(
            panel,
            y,
            pairs,
            folds,
            config,
            tmp_path,
            "toy",
            RunLog(tmp_path / "corrupt.jsonl"),
            lambda _: None,
            plan,
        )


def test_tree_attribution_replays_and_reuses_parent_statistics(tmp_path, config, monkeypatch):
    from commodity_prediction.domain.attribution.run import fit_stage

    panel, y, pairs = example_panel()
    fold = Fold(0, 200, 205, 235)
    stats = prepare(panel, y, fold.train_stop, config)
    experiment = Experiment("drop_trend_shape", drop="trend_shape", algorithm="histogram")
    with threadpool_limits(limits=1):
        result = fit_stage(panel, y, pairs, fold, experiment, config, stats, tmp_path, "child")
    assert result["selected_weight"] == 1
    assert not any(
        name.startswith("trend_shape__") for name in result["selection"]["selected_names"]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed tree attribution must not refit")

    monkeypatch.setattr("commodity_prediction.domain.attribution.run.fit", forbidden)
    assert fit_stage(panel, y, pairs, fold, experiment, config, None, tmp_path, "child") == result
    (tmp_path / "predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        fit_stage(panel, y, pairs, fold, experiment, config, None, tmp_path, "child")


def test_child_source_does_not_invalidate_completed_domain_fingerprint():
    from commodity_prediction.runtime import digest, fingerprint

    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "reports/domain_lineage.json").read_text())
    assert (
        fingerprint(evidence) == "52d3650abd20481db1a88fe7793085360bb0b4b684c501518ae9f4cb1c9405ba"
    )
    for name, expected in evidence["files"].items():
        assert digest(root / name) == expected
    assert all("/attribution/" not in name for name in evidence["files"])
