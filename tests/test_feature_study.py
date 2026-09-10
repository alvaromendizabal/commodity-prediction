"""Leakage, structural identities, fold-local selection, replay, and resume contracts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

from commodity_prediction.data import reconstruct_targets
from commodity_prediction.features import build_candidates
from commodity_prediction.research import lineage_for
from commodity_prediction.runtime import atomic_json, seal_checkpoint, verify_checkpoint
from commodity_prediction.studies.evaluation import (
    block_indices,
    block_permutation,
    group_permutations,
)
from commodity_prediction.studies.models import (
    ControlBundle,
    ModelBundle,
    fit_output,
    projection_for,
)
from commodity_prediction.studies.representation import (
    FeatureInfo,
    asset_labels,
    extend_candidates,
    release_history,
    routed_columns,
)
from commodity_prediction.studies.run import experiment_plan, run_study, study_folds
from commodity_prediction.studies.screening import (
    correlations,
    prepare_features,
    relevance_scores,
    select_features,
)


@pytest.fixture
def study_config():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/feature_study.json").read_text())


def metadata(horizons=(1, 2, 3, 4)):
    return pd.DataFrame(
        [
            {"target": f"target_{j * 4 + i}", "lag": h, "pair": pair}
            for j, h in enumerate(horizons)
            for i, pair in enumerate(["FX_A", "FX_B", "FX_A - FX_B", "FX_B - FX_A"])
        ]
    )


def market_data(n=320):
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 2)), axis=0)),
        columns=["FX_A", "FX_B"],
        index=pd.Index(np.arange(n), name="date_id"),
    )


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_label_is_visible_exactly_on_its_release_date(horizon):
    pairs = pd.DataFrame({"target": ["target_0"], "lag": [horizon], "pair": ["FX_A"]})
    y = pd.DataFrame({"target_0": np.arange(200, dtype=float)})
    baseline = release_history(y, pairs)
    changed = y.copy()
    changed.loc[90, "target_0"] = 999
    result = release_history(changed, pairs)
    release = 90 + horizon + 1
    pd.testing.assert_frame_equal(baseline.iloc[:release], result.iloc[:release])
    assert result.loc[release, "release_history__target_0__latest"] == 999
    assert result.loc[release - 1, "release_history__target_0__latest"] == 89


def test_every_extended_family_ignores_unavailable_market_and_label_values():
    x = market_data()
    pairs = metadata()
    y = reconstruct_targets(x, pairs)
    original, _ = extend_candidates(x, y, pairs)
    cutoff = 200
    changed_x, changed_y = x.copy(), y.copy()
    changed_x.loc[cutoff + 1 :] *= 3
    for target, lag, _ in pairs.itertuples(index=False, name=None):
        changed_y.loc[cutoff - lag :, target] = -999
    changed, _ = extend_candidates(changed_x, changed_y, pairs)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])
    assert {name.split("__")[0] for name in original} == {
        "regime",
        "interactions",
        "release_history",
    }


def test_release_history_rejects_date_gaps_and_masks_missing_filler():
    pairs = metadata((1,))
    y = pd.DataFrame({name: np.arange(100, dtype=float) for name in pairs.target})
    y.loc[50, "target_0"] = -999999
    result = release_history(y, pairs)
    assert pd.isna(result.loc[52, "release_history__target_0__latest"])
    with pytest.raises(ValueError, match="contiguous"):
        release_history(y.drop(index=50), pairs)


def test_structural_targets_obey_the_official_identity_and_release_boundary():
    x = market_data()
    pairs = metadata()
    labels = asset_labels(x, [1, 2, 3, 4])
    bundle = ModelBundle(
        [],
        np.array([]),
        np.array([]),
        np.array([]),
        list(labels),
        list(pairs.target),
        [],
        projection_for(pairs, list(labels), True),
    )
    projected = bundle.project(labels.to_numpy())
    np.testing.assert_allclose(
        projected, reconstruct_targets(x, pairs).to_numpy(), rtol=0, atol=1e-12, equal_nan=True
    )
    changed = x.copy()
    changed.iloc[105:] *= 2
    np.testing.assert_allclose(
        labels.iloc[:100],
        asset_labels(changed, [1, 2, 3, 4]).iloc[:100],
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_terminal_embargo_prevents_any_validation_target_from_entering_holdout(study_config):
    parent = {
        "holdout_dates": 252,
        "n_folds": 3,
        "validation_dates": 180,
        "purge_dates": 5,
        "min_train_dates": 600,
    }
    folds, start = study_folds(1961, parent, study_config)
    assert [f.validation_stop - f.validation_start for f in folds] == [180, 180, 175]
    assert all(f.validation_stop - 1 + 5 < start for f in folds)
    with pytest.raises(ValueError, match="Terminal embargo"):
        study_folds(1961, parent, {**study_config, "terminal_embargo_dates": 4})


def test_correlations_match_independent_pairwise_pearson_with_missing_labels():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5))
    y = rng.normal(size=(200, 3))
    y[rng.uniform(size=y.shape) < 0.2] = np.nan
    actual = correlations(x, y, 40)
    for j in range(3):
        observed = np.isfinite(y[:, j])
        for i in range(5):
            assert actual[i, j] == pytest.approx(
                pearsonr(x[observed, i], y[observed, j]).statistic, abs=1e-12
            )


def test_temporal_sign_reversals_fail_stability_screen(study_config):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(600, 2))
    y = (np.r_[np.full(300, 3), np.full(300, -1)] * x[:, 0] + 1.5 * x[:, 1])[:, None]
    ordinary, stable = relevance_scores(x, y, study_config)
    assert ordinary[0, 0] > 0.1
    assert stable[0, 0] == 0
    assert stable[1, 0] > 0.1


def test_validation_mutation_cannot_change_preprocessing_or_selection(study_config):
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    train.iloc[3, 0] = np.nan
    valid = train.iloc[:30].copy()
    a = prepare_features(train, valid, study_config)
    b = prepare_features(train, valid * 1e9, study_config)
    for name in ["medians", "means", "scales", "train"]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    scores = np.arange(5, dtype=float)
    assert select_features(a, list(train), scores, study_config, False) == select_features(
        b, list(train), scores, study_config, False
    )


def test_duplicates_are_removed_within_each_ablation_pool(study_config):
    x = pd.DataFrame({"reference": np.arange(100), "pair": np.arange(100), "constant": 1})
    prepared = prepare_features(x, x.iloc[:10], study_config)
    selected, audit = select_features(prepared, list(x), np.ones(2), study_config, False)
    assert audit["rejection_reasons"] == {"duplicate": 1, "constant": 1}
    assert len(selected) == 1
    selected, audit = select_features(prepared, ["pair"], np.ones(2), study_config, False)
    assert audit["selected"] == ["pair"]
    assert audit["candidate_count"] == audit["retained_count"] + audit["rejected_count"]


def test_routing_excludes_unrelated_pairs_and_other_targets_history():
    info = {
        "a": FeatureInfo("reference", ("A",)),
        "b": FeatureInfo("reference", ("B",)),
        "context": FeatureInfo("cross_market", context=True),
        "ab": FeatureInfo("pairs", ("A", "B"), pair="A - B"),
        "ac": FeatureInfo("pairs", ("A", "C"), pair="A - C"),
        "own": FeatureInfo("release_history", ("A",), target="target_0"),
        "other": FeatureInfo("release_history", ("A",), target="target_1"),
    }
    assert set(routed_columns(info, ("A", "B"), "A - B", "target_0")) == {
        "a",
        "b",
        "context",
        "ab",
        "own",
    }
    assert set(routed_columns(info, ("A",))) == {"a", "context"}


@pytest.mark.parametrize("algorithm", ["ridge", "histogram"])
def test_saved_models_replay_predictions_and_do_not_randomly_hold_out_rows(
    tmp_path, study_config, algorithm
):
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(250, 3)), columns=["a", "b", "c"])
    y = 0.01 * x.a.to_numpy() + rng.normal(0, 0.002, len(x))
    prepared = prepare_features(x.iloc[:200], x.iloc[200:], study_config)
    with threadpool_limits(limits=1):
        model = fit_output(prepared, y[:200], [0, 1], algorithm, study_config)
        bundle = ModelBundle(
            prepared.columns,
            prepared.medians,
            prepared.means,
            prepared.scales,
            ["target_0"],
            ["target_0"],
            [model],
            [[(0, 1.0)]],
        )
        expected = bundle.predict(x.iloc[200:])
        joblib.dump(bundle, tmp_path / "model.joblib")
        pd.testing.assert_frame_equal(
            joblib.load(tmp_path / "model.joblib").predict(x.iloc[200:]), expected
        )
    if algorithm == "histogram":
        assert model.estimator.early_stopping is False
    assert pearsonr(expected.target_0, y[200:]).statistic > 0.8


def test_bootstrap_blocks_never_cross_model_boundaries_and_are_reproducible():
    a = block_indices([100, 95, 85], 20, 100, 42)
    np.testing.assert_array_equal(a, block_indices([100, 95, 85], 20, 100, 42))
    assert a.shape == (100, 280)
    assert np.all((a[:, :100] >= 0) & (a[:, :100] < 100))
    assert np.all((a[:, 100:195] >= 100) & (a[:, 100:195] < 195))
    assert np.all((a[:, 195:] >= 195) & (a[:, 195:] < 280))
    permutation = block_permutation(85, 20, np.random.default_rng(42))
    assert sorted(permutation.tolist()) == list(range(85))


def test_group_permutation_detects_reliance_on_predictive_family(study_config):
    rng = np.random.default_rng(10)
    x = pd.DataFrame(rng.normal(size=(400, 2)), columns=["signal", "noise"])
    slopes = np.array([-2, -0.5, 0.5, 2])
    y = x.signal.to_numpy()[:, None] * slopes + rng.normal(size=(400, 4))
    prepared = prepare_features(x.iloc[:200], x.iloc[200:], study_config)
    config = {**study_config, "permutation_repetitions": 3}
    with threadpool_limits(limits=1):
        models = [fit_output(prepared, y[:200, i], [0, 1], "ridge", config) for i in range(4)]
        names = [f"target_{i}" for i in range(4)]
        bundle = ModelBundle(
            prepared.columns,
            prepared.medians,
            prepared.means,
            prepared.scales,
            names,
            names,
            models,
            [[(i, 1.0)] for i in range(4)],
        )
        result = group_permutations(
            bundle,
            x.iloc[200:],
            pd.DataFrame(y[200:], index=x.index[200:], columns=names),
            {name: FeatureInfo(name) for name in x},
            config,
        )
    assert result["signal"]["mean_metric_drop"] > result["noise"]["mean_metric_drop"] + 0.3


def test_full_study_reuses_every_completed_fit(tmp_path, monkeypatch, study_config):
    import commodity_prediction.studies.run as runner

    x = market_data(800)
    pairs = metadata()
    y = reconstruct_targets(x, pairs)
    parent_config = {
        "holdout_dates": 20,
        "n_folds": 3,
        "validation_dates": 180,
        "purge_dates": 5,
        "min_train_dates": 100,
    }
    atomic_json(tmp_path / "configs/research.json", parent_config)
    parent, _ = lineage_for(tmp_path, parent_config)
    config = {
        **study_config,
        "parent_lineage": parent,
        "bootstrap_repetitions": 30,
        "bootstrap_block_dates": [20],
        "permutation_repetitions": 1,
        "histogram_iterations": 4,
        "threads": 1,
    }
    atomic_json(tmp_path / "configs/feature_study.json", config)
    folds, stop = study_folds(len(x), parent_config, config)
    base, _ = build_candidates(x.iloc[:stop], pairs)
    stage = tmp_path / "artifacts" / parent / "features"
    stage.mkdir(parents=True)
    base.to_parquet(stage / "candidates.parquet")
    seal_checkpoint(stage, parent, ["candidates.parquet"])
    for fold in folds:
        stage = tmp_path / "artifacts" / parent / f"fold_{fold.number}" / "reference"
        stage.mkdir(parents=True)
        block = base.iloc[fold.validation_start : fold.validation_stop]
        ControlBundle(list(y), y.iloc[: fold.train_stop].mean().to_numpy()).predict(
            block
        ).to_parquet(stage / "predictions.parquet")
        seal_checkpoint(stage, parent, ["predictions.parquet"])
    atomic_json(tmp_path / "reports/research.json", {"comparison": [{"variant": "reference"}]})
    monkeypatch.setattr(runner, "load_data", lambda root: (x, y, pairs))
    report = run_study(tmp_path)
    assert report["experiments_completed"] == len(experiment_plan()) * 3
    assert report["validation_dates"] == 535
    assert not report["holdout_evaluated"]
    assert (
        report["candidate_count"]
        == report["reused_candidate_count"] + report["new_candidate_count"]
    )
    directory = tmp_path / "artifacts" / report["lineage"]
    manifests = list(directory.rglob("manifest.json"))
    assert len(manifests) == 1 + 3 * len(experiment_plan()) + 3 + 1
    assert all(verify_checkpoint(path.parent, report["lineage"]) for path in manifests)

    def cannot_fit(*args, **kwargs):
        raise AssertionError("A valid completed fit must be reused")

    monkeypatch.setattr(runner, "fit_output", cannot_fit)
    monkeypatch.setattr(runner, "prepare_features", cannot_fit)
    resumed = run_study(tmp_path)
    assert resumed == report
