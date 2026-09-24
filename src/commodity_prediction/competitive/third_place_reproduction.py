"""Documented reproduction of the public MITSUI 3rd-place strategy.

This module intentionally separates:
1) source-described structure (pair routing, simple temporal features, LGBM/RF/XGB stack)
2) causal research-grade adaptation (no future shifts, chronological OOF meta-features)

The public writeup does not disclose every hyperparameter/window, so reconstruction
choices are explicit in configs/third_place_reproduction.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


def signed_log1p(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.sign(values) * np.log1p(np.abs(values))


def parse_pair(pair: str) -> list[str]:
    cols = pair.split(" - ")
    if not 1 <= len(cols) <= 2:
        raise ValueError(f"Unexpected pair specification: {pair!r}")
    return cols


def _base_frame(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in x.columns]
    if missing:
        raise KeyError(f"Pair columns absent from market data: {missing}")
    # Causal fill only. The source writeup describes ffill+bfill, but bfill can
    # carry future information backward in an offline matrix.
    return x[columns].astype(float).replace([np.inf, -np.inf], np.nan).ffill()


def build_causal_pair_features(
    x: pd.DataFrame,
    pair: str,
    *,
    positive_lags: Iterable[int] = (1, 2, 3, 4),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
) -> pd.DataFrame:
    """Recreate the documented 3rd-place feature families without future shifts."""
    cols = parse_pair(pair)
    base = _base_frame(x, cols)
    out: dict[str, pd.Series] = {}

    for col in cols:
        s = base[col]
        out[f"{col}__raw"] = s
        out[f"{col}__slog1p"] = signed_log1p(s)
        out[f"{col}__diff1"] = s.diff()
        out[f"{col}__slog1p_diff1"] = signed_log1p(s).diff()

        for lag in positive_lags:
            if lag <= 0:
                raise ValueError("Causal positive_lags must be > 0")
            out[f"{col}__lag{lag}"] = s.shift(lag)
            out[f"{col}__diff1_lag{lag}"] = s.diff().shift(lag)

        for window in rolling_windows:
            if window < 2:
                raise ValueError("Rolling windows must be >=2")
            out[f"{col}__rollmean{window}"] = s.rolling(window, min_periods=2).mean()
            out[f"{col}__rollmax{window}"] = s.rolling(window, min_periods=2).max()
            out[f"{col}__rollmin{window}"] = s.rolling(window, min_periods=2).min()
            out[f"{col}__rollstd{window}"] = s.rolling(window, min_periods=2).std()

    if len(cols) == 2:
        a, b = base[cols[0]], base[cols[1]]
        spread = a - b
        out["pair__spread"] = spread
        out["pair__spread_diff1"] = spread.diff()
        out["pair__slog_spread"] = signed_log1p(spread)
        denom = b.abs().replace(0.0, np.nan)
        out["pair__relative"] = a / denom
        out["pair__relative_diff1"] = (a / denom).diff()
        for lag in positive_lags:
            out[f"pair__spread_lag{lag}"] = spread.shift(lag)
            out[f"pair__spread_diff1_lag{lag}"] = spread.diff().shift(lag)
        for window in rolling_windows:
            out[f"pair__spread_rollmean{window}"] = spread.rolling(window, min_periods=2).mean()
            out[f"pair__spread_rollmax{window}"] = spread.rolling(window, min_periods=2).max()

    frame = pd.DataFrame(out, index=x.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_forensic_source_described_features(
    x: pd.DataFrame,
    pair: str,
    *,
    positive_lags: Iterable[int] = (1, 2, 3, 4),
    negative_lags: Iterable[int] = (-1, -2, -3, -4),
    rolling_windows: Iterable[int] = (3, 5, 10, 20),
) -> pd.DataFrame:
    """Forensic only: includes literal negative shifts described in the writeup.

    This is deliberately NOT used by default because shift(-k) uses future rows
    relative to an origin and is not acceptable for promotion.
    """
    frame = build_causal_pair_features(
        x, pair, positive_lags=positive_lags, rolling_windows=rolling_windows
    )
    cols = parse_pair(pair)
    base = x[cols].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    for col in cols:
        for lag in negative_lags:
            if lag >= 0:
                raise ValueError("forensic negative_lags must be <0")
            frame[f"{col}__neg_lag{abs(lag)}"] = base[col].shift(lag)
    return frame


@dataclass
class Prepared:
    imputer: SimpleImputer
    scaler: StandardScaler
    train: np.ndarray
    valid: np.ndarray


def prepare_train_valid(
    train: pd.DataFrame, valid: pd.DataFrame
) -> Prepared:
    """Median imputation + standardization, fit on training only."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_i = imputer.fit_transform(train)
    valid_i = imputer.transform(valid)
    train_s = scaler.fit_transform(train_i)
    valid_s = scaler.transform(valid_i)
    if not np.isfinite(train_s).all() or not np.isfinite(valid_s).all():
        raise ValueError("Non-finite prepared features")
    return Prepared(imputer, scaler, train_s, valid_s)


def make_base_models(seed: int, threads: int):
    """Publicly documented model families; exact hyperparameters were not disclosed."""
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError("LightGBM is required for the 3rd-place reproduction") from exc
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost is required for the 3rd-place reproduction") from exc

    return {
        "lightgbm": LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=threads,
        ),
        "xgboost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.3,
            max_depth=6,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=threads,
            tree_method="hist",
            verbosity=0,
        ),
    }


def make_meta_model(seed: int, threads: int):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost is required for the stack meta-model") from exc
    return XGBRegressor(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=threads,
        tree_method="hist",
        verbosity=0,
    )


def fit_source_described_stack(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    *,
    seed: int,
    threads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce the writeup's in-sample meta-feature stacking structure."""
    base = make_base_models(seed, threads)
    train_meta, valid_meta = [], []
    valid_parts: dict[str, np.ndarray] = {}
    for name, model in base.items():
        model.fit(x_train, y_train)
        train_pred = np.asarray(model.predict(x_train), dtype=float)
        valid_pred = np.asarray(model.predict(x_valid), dtype=float)
        train_meta.append(train_pred)
        valid_meta.append(valid_pred)
        valid_parts[name] = valid_pred
    train_m = np.column_stack(train_meta)
    valid_m = np.column_stack(valid_meta)
    meta = make_meta_model(seed, threads)
    meta.fit(train_m, y_train)
    final = np.asarray(meta.predict(valid_m), dtype=float)
    valid_parts["stack"] = final
    return final, valid_parts


def _time_series_splits(n_samples: int, n_splits: int, gap: int):
    if n_samples < 100:
        raise ValueError("Need at least 100 training rows for OOF stacking")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return list(splitter.split(np.arange(n_samples)))


def fit_causal_oof_stack(
    raw_train: pd.DataFrame,
    y_train: pd.Series,
    raw_valid: pd.DataFrame,
    *,
    seed: int,
    threads: int,
    n_splits: int,
    gap: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Leakage-safe chronological OOF stack using the same three base families."""
    valid_mask = y_train.notna()
    x = raw_train.loc[valid_mask]
    y = y_train.loc[valid_mask].to_numpy(dtype=float)
    if len(y) < 100:
        raise ValueError("Insufficient observed target rows")

    splits = _time_series_splits(len(x), n_splits, gap)
    names = ["lightgbm", "random_forest", "xgboost"]
    oof = np.full((len(x), len(names)), np.nan, dtype=float)

    for split_id, (tr, va) in enumerate(splits):
        prepared = prepare_train_valid(x.iloc[tr], x.iloc[va])
        models = make_base_models(seed + split_id, threads)
        for j, name in enumerate(names):
            models[name].fit(prepared.train, y[tr])
            oof[va, j] = np.asarray(models[name].predict(prepared.valid), dtype=float)

    meta_rows = np.isfinite(oof).all(axis=1)
    if meta_rows.sum() < 30:
        raise ValueError("Too few chronological OOF rows for meta-model")

    meta = make_meta_model(seed, threads)
    meta.fit(oof[meta_rows], y[meta_rows])

    prepared_full = prepare_train_valid(x, raw_valid)
    models = make_base_models(seed, threads)
    valid_meta = []
    valid_parts: dict[str, np.ndarray] = {}
    for name in names:
        models[name].fit(prepared_full.train, y)
        pred = np.asarray(models[name].predict(prepared_full.valid), dtype=float)
        valid_meta.append(pred)
        valid_parts[name] = pred

    final = np.asarray(meta.predict(np.column_stack(valid_meta)), dtype=float)
    valid_parts["stack_oof"] = final
    return final, valid_parts


def choose_stratified_targets(pairs: pd.DataFrame, n: int) -> list[str]:
    """Deterministic panel across horizon and single/pair targets."""
    if n >= len(pairs):
        return pairs["target"].tolist()
    table = pairs.copy()
    table["kind"] = table["pair"].str.contains(" - ").map({True: "pair", False: "single"})
    buckets = []
    for lag in sorted(table["lag"].unique()):
        for kind in ["single", "pair"]:
            subset = table[(table["lag"] == lag) & (table["kind"] == kind)]["target"].tolist()
            buckets.append(subset)

    selected: list[str] = []
    cursor = 0
    while len(selected) < n and any(buckets):
        bucket = buckets[cursor % len(buckets)]
        if bucket:
            selected.append(bucket.pop(0))
        cursor += 1
    return selected[:n]


def prefix_invariant_feature_check(
    x: pd.DataFrame,
    pair: str,
    cut: int,
    *,
    positive_lags=(1, 2, 3, 4),
    rolling_windows=(3, 5, 10, 20),
) -> bool:
    full = build_causal_pair_features(
        x, pair, positive_lags=positive_lags, rolling_windows=rolling_windows
    ).iloc[:cut]
    prefix = build_causal_pair_features(
        x.iloc[:cut], pair, positive_lags=positive_lags, rolling_windows=rolling_windows
    )
    return full.equals(prefix)
