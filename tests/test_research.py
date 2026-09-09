"""Methodological and artifact-integrity tests, independent of competition data."""

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from commodity_prediction.cloud import extract_archive
from commodity_prediction.data import make_folds, reconstruct_targets, validate_dates
from commodity_prediction.features import build_candidates, screen_features
from commodity_prediction.metrics import (
    correlation_sharpe,
    daily_rank_correlations,
    paired_block_interval,
    score,
)
from commodity_prediction.research import fit_diagnostic
from commodity_prediction.runtime import RunLog, atomic_json, seal_checkpoint, verify_checkpoint


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/research.json").read_text())


@pytest.fixture
def market():
    rng = np.random.default_rng(42)
    n = 130
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    b = 80 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    frame = pd.DataFrame(
        {
            "LME_A_Close": a,
            "FX_USDJPY": b,
            "JPX_Gold_Close": a * 1.05,
            "JPX_Gold_Open": a * 1.04,
            "JPX_Gold_High": a * 1.07,
            "JPX_Gold_Low": a * 1.02,
            "JPX_Gold_Volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    frame.loc[7:9, "LME_A_Close"] = np.nan
    pairs = pd.DataFrame(
        {
            "target": ["target_0", "target_1"],
            "lag": [1, 4],
            "pair": ["LME_A_Close", "LME_A_Close - FX_USDJPY"],
        }
    )
    return frame, pairs


@pytest.mark.parametrize("dates", [[0, 0, 1], [2, 1, 0], [0, 2, 3], [0.0, 1.0]])
def test_invalid_date_contract(dates):
    with pytest.raises(ValueError):
        validate_dates(pd.DataFrame({"date_id": dates}))


def test_valid_dates():
    validate_dates(pd.DataFrame({"date_id": [0, 1, 2]}))


def test_purged_walk_forward_never_uses_unreleased_labels(config):
    folds, end = make_folds(1961, config)
    assert end == 1709
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_stop - 1 + 5 < fold.validation_start
        assert fold.validation_stop <= end
    assert folds[-1].validation_stop == end
    assert folds[0].validation_stop == folds[1].validation_start


@pytest.mark.parametrize("purge", [0, 1, 4])
def test_insufficient_purge_rejected(config, purge):
    with pytest.raises(ValueError, match="purged"):
        make_folds(1961, {**config, "purge_dates": purge})


def test_short_history_rejected(config):
    with pytest.raises(ValueError, match="history"):
        make_folds(500, config)


def test_target_starts_tomorrow_and_ends_horizon_plus_one(market):
    x, pairs = market
    result = reconstruct_targets(x, pairs)
    t = 30
    a, b = x.LME_A_Close, x.FX_USDJPY
    assert result.loc[t, "target_0"] == pytest.approx(np.log(a[t + 2] / a[t + 1]))
    expected = np.log(a[t + 5] / a[t + 1]) - np.log(b[t + 5] / b[t + 1])
    assert result.loc[t, "target_1"] == pytest.approx(expected)


def test_metric_matches_independent_spearman_reference_with_missing_and_ties():
    y = pd.DataFrame([[1.0, 2, 2, 4], [3, 2, np.nan, 1], [4, 1, 2, -999999]])
    p = pd.DataFrame([[3.0, 2, 2, 1], [4, 2, 9, 1], [2, 3, 1, 0]])
    expected = []
    for i in y.index:
        observed = y.loc[i].notna() & y.loc[i].ne(-999999)
        expected.append(spearmanr(y.loc[i, observed], p.loc[i, observed]).statistic)
    np.testing.assert_allclose(daily_rank_correlations(y, p), expected, atol=1e-14)
    assert score(y, p) == pytest.approx(np.mean(expected) / np.std(expected, ddof=0))


@pytest.mark.parametrize("mutation", ["columns", "rows", "nan", "inf"])
def test_metric_rejects_invalid_alignment_or_predictions(mutation):
    y = pd.DataFrame([[1.0, 2, 3], [3, 1, 2]], columns=["a", "b", "c"])
    p = y.copy()
    if mutation == "columns":
        p = p[["c", "b", "a"]]
    elif mutation == "rows":
        p.index = [2, 3]
    else:
        p.iloc[0, 0] = np.nan if mutation == "nan" else np.inf
    with pytest.raises(ValueError):
        score(y, p)


def test_constant_metric_is_an_error_not_a_fabricated_score():
    with pytest.raises(ZeroDivisionError):
        daily_rank_correlations(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[0, 0, 0]]))
    with pytest.raises(ZeroDivisionError):
        correlation_sharpe(np.ones(3))


def test_metric_is_invariant_to_strictly_monotonic_prediction_transform():
    rng = np.random.default_rng(4)
    y = pd.DataFrame(rng.normal(size=(8, 6)))
    p = pd.DataFrame(rng.normal(size=(8, 6)))
    assert score(y, p) == pytest.approx(score(y, np.exp(p)))


def test_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(2)
    values = rng.normal(0.05, 0.2, 70)
    assert paired_block_interval(values, values, 30) == [0.0, 0.0]
    assert paired_block_interval(values, values + 0.02, 30) == paired_block_interval(
        values, values + 0.02, 30
    )


def test_all_feature_families_are_invariant_to_future_mutation(market):
    x, pairs = market
    original, families = build_candidates(x, pairs)
    changed = x.copy()
    changed.loc[100:] = changed.loc[100:] * 7
    mutated, _ = build_candidates(changed, pairs)
    prefix, _ = build_candidates(x.iloc[:100], pairs)
    pd.testing.assert_frame_equal(original.iloc[:100], mutated.iloc[:100])
    pd.testing.assert_frame_equal(original.iloc[:100], prefix)
    assert len(set(families.values())) == 9
    assert original.columns.is_unique
    assert not np.isinf(original.to_numpy()).any()


def test_market_holidays_are_not_backfilled(market):
    x, pairs = market
    features, _ = build_candidates(x, pairs)
    assert features.loc[7:9, "reference__LME_A_Close_return_1"].isna().all()
    assert features.loc[7:9, "missingness__LME_A_Close_observation_age"].tolist() == [1.0, 2.0, 3.0]


def test_screen_records_rejections_and_uses_only_training_statistics(config):
    rng = np.random.default_rng(42)
    signal = rng.normal(size=150)
    x = pd.DataFrame(
        {
            "signal": signal,
            "duplicate": signal,
            "constant": 1.0,
            "missing": np.nan,
            "correlated": signal + rng.normal(0, 1e-5, 150),
            "noise": rng.normal(size=150),
        }
    )
    y = pd.DataFrame({"target_0": signal + rng.normal(0, 0.1, 150)})
    screen = screen_features(x, y, {**config, "max_features": 2})
    reasons = screen.audit.set_index("feature").reason
    assert reasons["duplicate"].startswith("duplicate:")
    assert reasons["constant"] == "constant"
    assert reasons["missing"] == "missingness"
    assert screen.audit.status.value_counts()["retained"] == 2
    before = screen.medians.copy()
    validation = x.iloc[:4].copy() * 1000
    validation.iloc[0] = np.nan
    assert np.isfinite(screen.transform(validation)).all()
    pd.testing.assert_series_equal(screen.medians, before)


def test_screen_rejects_empty_candidates(config):
    with pytest.raises(ValueError, match="No usable"):
        screen_features(pd.DataFrame({"a": [1.0] * 100}), pd.DataFrame({"b": [1.0] * 100}), config)


def test_fit_ignores_missing_labels_and_exports_replayable_coefficients(config):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(150, 5))
    y = pd.DataFrame({"a": x[:, 0] + 0.1, "b": x[:, 1] - 0.2})
    y.loc[:10, "a"] = np.nan
    predicted, coefficient, intercept, fallback = fit_diagnostic(x, x[:9], y, config)
    np.testing.assert_allclose(predicted, x[:9] @ coefficient.T + intercept)
    assert fallback == 0


@pytest.mark.parametrize("failure", ["tamper", "missing", "lineage"])
def test_checkpoint_rejects_invalid_artifacts(tmp_path, failure):
    atomic_json(tmp_path / "result.json", {"value": 3})
    seal_checkpoint(tmp_path, "abc", ["result.json"])
    assert verify_checkpoint(tmp_path, "abc")
    if failure == "tamper":
        atomic_json(tmp_path / "result.json", {"value": 4})
    elif failure == "missing":
        (tmp_path / "result.json").unlink()
    with pytest.raises(ValueError):
        verify_checkpoint(tmp_path, "other" if failure == "lineage" else "abc")


def test_no_manifest_is_incomplete(tmp_path):
    assert verify_checkpoint(tmp_path, "abc") is None


def test_atomic_json_does_not_publish_nonfinite_values(tmp_path):
    atomic_json(tmp_path / "result.json", {"value": 1})
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "result.json", {"value": float("nan")})
    assert json.loads((tmp_path / "result.json").read_text()) == {"value": 1}


def test_heartbeat_and_failure_are_logged_in_utc(tmp_path):
    log = RunLog(tmp_path / "run.jsonl", heartbeat_seconds=0.01)
    with pytest.raises(RuntimeError), log.stage("test"):
        time.sleep(0.03)
        raise RuntimeError("intentional")
    rows = [json.loads(row) for row in log.path.read_text().splitlines()]
    assert {"heartbeat", "stage_started", "stage_failed"}.issubset({r["event"] for r in rows})
    assert all(r["utc"].endswith("+00:00") for r in rows)
    assert all("stage_elapsed_seconds" in r and "total_elapsed_seconds" in r for r in rows)


def test_archive_path_traversal_rejected(tmp_path):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(path, tmp_path / "data")
    assert not (tmp_path / "outside.txt").exists()
