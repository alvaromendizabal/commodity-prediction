from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_prediction.competitive.third_place_reproduction import (
    build_causal_pair_features,
    build_forensic_source_described_features,
    choose_stratified_targets,
    parse_pair,
    prefix_invariant_feature_check,
    prepare_train_valid,
)


def test_parse_pair_single_and_pair() -> None:
    assert parse_pair("A") == ["A"]
    assert parse_pair("A - B") == ["A", "B"]


def test_causal_features_are_prefix_invariant() -> None:
    x = pd.DataFrame(
        {"A": np.arange(1, 81, dtype=float), "B": np.arange(101, 181, dtype=float)},
        index=pd.RangeIndex(80, name="date_id"),
    )
    assert prefix_invariant_feature_check(x, "A - B", 50)


def test_forensic_negative_shift_is_detectably_noncausal() -> None:
    x = pd.DataFrame(
        {"A": np.arange(1, 30, dtype=float)},
        index=pd.RangeIndex(29, name="date_id"),
    )
    full = build_forensic_source_described_features(x, "A")
    prefix = build_forensic_source_described_features(x.iloc[:20], "A")
    col = "A__neg_lag1"
    assert not full.iloc[:20][col].equals(prefix[col])


def test_train_only_imputation_and_scaling_are_finite() -> None:
    tr = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, np.nan, 4.0]})
    va = pd.DataFrame({"a": [np.nan, 8.0], "b": [9.0, np.nan]})
    p = prepare_train_valid(tr, va)
    assert np.isfinite(p.train).all()
    assert np.isfinite(p.valid).all()


def test_stratified_target_selection_is_deterministic() -> None:
    rows = []
    k = 0
    for lag in [1, 2, 3, 4]:
        for kind in ["single", "pair"]:
            for _ in range(3):
                rows.append({
                    "target": f"target_{k}",
                    "lag": lag,
                    "pair": "A" if kind == "single" else "A - B",
                })
                k += 1
    pairs = pd.DataFrame(rows)
    a = choose_stratified_targets(pairs, 8)
    b = choose_stratified_targets(pairs, 8)
    assert a == b
    assert len(a) == 8
