#!/usr/bin/env python3
"""Install notebook 12; three bounded delayed-response representation ablations.

Default CLI: setup only. Notebook: maximum three fixed-learner fits on development
fold 1. Rolling, outcome-trained feature coefficients update only from labels
released by the prediction origin. No cloud/Git writes, installs, final test,
old model refits, unreviewed retries or automatic space shutdown.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-manual")
SHA = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
PARENT = "04544d4b1a1d63487a24e88749d03d3259326c63902f2d0c74214e711147d605"
FEATURE = "52d3650abd20481db1a88fe7793085360bb0b4b684c501518ae9f4cb1c9405ba"
PRIOR_LINEAGE = "c2d8078948d72467b75dc30f692ad9cbf4f38b4ff1f74600b00e2aea63413d2c"
PRIOR_HASH = "7a577e40c4c0c87de13da8de573e529214220fd5f5b46bae9ee8c42950f33bf5"
SESSION_HASH = "7efef4574d55bd9aad8da8ecf4e7f4ba7cc54907412dcfb4a754100d9e34f88f"
TRAIN_STOP, START, STOP, FOLD = 1344, 1349, 1529, 1
CONTROL = 0.16704165319061334
LIMIT, GAIN = 240.0, 0.002
WINDOWS = (63, 126)
DRIVER_LAGS = (1, 5)
NOTEBOOK = "12_delayed_response_ablation.ipynb"
VARIANTS = ("response_drivers", "response_encoded", "response_joint")
TERMINAL = {"RESPONSE_REVIEW_READY", "NOTEBOOK_AND_RESPONSE_READY"}
DRIVER_NAMES = [f"delayed_response__driver_{lag}" for lag in DRIVER_LAGS]
RESPONSE_NAMES = [
    f"delayed_response__encoded_{lag}_{window}" for lag in DRIVER_LAGS for window in WINDOWS
]


class Stop(RuntimeError):
    """Preserve evidence; do not retry interrupted experiments unchanged."""


def utc():
    return datetime.now(UTC).isoformat()


def emit(stage, **fields):
    print(json.dumps(dict(utc=utc(), stage=stage, **fields), allow_nan=False), flush=True)


def previous(root):
    p = Path(root) / "scripts/commodity_session_ablation.py"
    if (
        p.is_symlink()
        or not p.is_file()
        or hashlib.sha256(p.read_bytes()).hexdigest() != SESSION_HASH
    ):
        raise Stop("The unchanged notebook-07 support is missing or modified; preserve it.")
    spec = importlib.util.spec_from_file_location("response12_session_utilities", p)
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    older, u = session.previous(Path(root))
    return session, older, u


def output_dir(root):
    return Path(root) / "logs/manual_response_encoding"


def report_path(root):
    return output_dir(root) / "commodity_response_encoding_report.json"


def plan():
    return dict(
        study="horizon_delayed_conditional_response_encoding",
        source_commit=SHA,
        parent_lineage=PARENT,
        prior_context_lineage=PRIOR_LINEAGE,
        fold=FOLD,
        training_stop_exclusive=TRAIN_STOP,
        validation_start=START,
        validation_stop_exclusive=STOP,
        warmup_dates=252,
        worker_limit_seconds=LIMIT,
        new_fit_limit=3,
        control_refits=0,
        driver_lags=list(DRIVER_LAGS),
        response_windows=list(WINDOWS),
        minimum_pairs=[42, 84],
        driver_definition="Directed log-price difference over 1 or 5 rows / (sqrt(lag) * prior 63-row spread-return population SD, min42, floor1e-6), clipped +-12. No forward fill; all intermediate prices must be valid.",
        feature_update_policy="Each target j uses signal/outcome ORIGIN-aligned pairs (s[u,j],y[u,j]) only when u+h_j+1 <= t. Rolling coefficients update sequentially during validation; downstream forecasting model stays frozen.",
        response_definition="n/(n+63*h) * Corr(s,y) * (current_s-mean_s)/sd_s, clipped +-6; all moments use the SAME finite paired observations. No outcome-mean/intercept added; no missing target zero-fill.",
        shrinkage_note="63*h is a fixed conservative pseudo-count for noisy, overlapping-horizon estimates, not an exact effective sample-size calculation.",
        minimum_signal_variance=1e-10,
        minimum_outcome_variance=1e-16,
        numerical_counts=[2, 4, 6],
        coverage_columns_added=0,
        new_feature_labels_used=True,
        coefficient_learning="Supervised rolling feature coefficients, not zero learning. No optimizer or extra forecast-model training during feature generation.",
        variants=list(VARIANTS),
        forecast_learner="Existing histogram model, configuration and training preprocessing unchanged; usable nonduplicate inputs admitted.",
        gate="Drivers >=0.002 over saved current_market; encoded and joint >=0.002 over BOTH saved current_market and fitted drivers. Required intended inputs admitted and exact replay. Review only.",
        review_gain=GAIN,
        contrasts=[[v, "current_market"] for v in VARIANTS]
        + [["response_encoded", "response_drivers"], ["response_joint", "response_drivers"]],
        period_choice="Same previously inspected middle development fold; exploratory after negative context panels, not fresh holdout or adaptive-search correction.",
        maximum_feature_label_origins={str(h): STOP - 1 - (h + 1) for h in (1, 2, 3, 4)},
        other_folds=False,
        feature_gate="open",
        promotion_allowed=False,
        final_test_evaluations=0,
        aws_api_calls=0,
        github_writes=False,
        hyperparameter_search=False,
    )


def validate_prior(r):
    expected = {
        "context_identity": 0.15334865292560604,
        "context_shocks": 0.1336489611227173,
        "context_joint": 0.12888769504762737,
    }
    if (
        r.get("status") != "NOTEBOOK_AND_TARGET_CONTEXT_READY"
        or r.get("lineage") != PRIOR_LINEAGE
        or r.get("source_commit") != SHA
        or r.get("decision") != "STOP_TESTED_TARGET_CONTEXT_PANELS"
        or r.get("new_training_fits") != 3
        or r.get("control_refits") != 0
        or r.get("fold") != 1
        or r.get("parents_unchanged") is not True
        or r.get("maximum_prediction_replay_error") != 0
        or r.get("final_test_evaluations") != 0
        or r.get("candidates_for_review") != []
    ):
        raise Stop("Notebook-11 result differs from the reviewed completed negative experiment.")
    rows = r.get("comparison_rows", [])
    if len(rows) != 3 or {v["variant"] for v in rows} != set(expected):
        raise Stop("Prior result inventory differs.")
    if any(
        not math.isclose(v["official_metric"], expected[v["variant"]], abs_tol=1e-12, rel_tol=0)
        for v in rows
    ):
        raise Stop("Prior scores changed.")


def read_prior(root, u, verify_artifacts=False):
    path = u.safe_path(root, "logs/manual_target_context/commodity_target_context_report.json")
    if u.digest(path) != PRIOR_HASH:
        raise Stop("Notebook-11 report changed. Preserve it for review.")
    r = u.read_json(path)
    validate_prior(r)
    if verify_artifacts:
        directory = u.safe_path(root, "artifacts/target_context/" + PRIOR_LINEAGE)
        u.verify_stage(directory / "features", PRIOR_LINEAGE, r["feature_checkpoint_hashes"])
        for name, pins in r["checkpoint_hashes"].items():
            u.verify_stage(directory / "fold_1" / name, PRIOR_LINEAGE, pins)
    return r


def identity(root, u):
    if u.read_json(u.safe_path(root, "configs/manual_response_encoding.json")) != plan():
        raise Stop("Response declaration changed.")
    evidence = dict(
        plan=plan(),
        helper_sha256=u.digest(Path(__file__)),
        prior_context_report_sha256=PRIOR_HASH,
        session_helper_sha256=SESSION_HASH,
        domain_config_sha256=u.digest(root / "configs/domain_study.json"),
    )
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(), evidence


def metadata_legs(pairs):
    import pandas as pd

    if (
        not isinstance(pairs, pd.DataFrame)
        or pairs.empty
        or not pairs.columns.is_unique
        or not {"target", "pair", "lag"}.issubset(pairs.columns)
        or pairs[["target", "pair", "lag"]].isna().any().any()
        or pairs.target.duplicated().any()
        or pd.api.types.is_bool_dtype(pairs.lag)
        or not pairs.lag.isin([1, 2, 3, 4]).all()
        or not pairs.target.map(lambda v: isinstance(v, str) and bool(v)).all()
    ):
        raise Stop("Invalid unique target/pair/horizon metadata.")
    result = []
    for value in pairs.pair:
        if not isinstance(value, str):
            raise Stop("Pair must be a string.")
        legs = value.split(" - ")
        if (
            len(legs) not in (1, 2)
            or len(set(legs)) != len(legs)
            or any(not a or a != a.strip() for a in legs)
        ):
            raise Stop("Invalid or duplicate pair legs.")
        result.append(legs)
    return result


def build_drivers(x, pairs):
    """Two label-free target-spread signals; no filled prices or fitted labels."""
    import numpy as np
    import pandas as pd

    legs = metadata_legs(pairs)
    assets = sorted({a for row in legs for a in row})
    if (
        not isinstance(x, pd.DataFrame)
        or x.empty
        or len(x) > STOP
        or not x.columns.is_unique
        or not pd.api.types.is_integer_dtype(x.index)
        or x.index.tolist() != list(range(len(x)))
        or not set(assets).issubset(x.columns)
        or any(
            not pd.api.types.is_numeric_dtype(x[a]) or pd.api.types.is_bool_dtype(x[a])
            for a in assets
        )
    ):
        raise Stop("Bounded contiguous numeric market input is required.")
    raw = x[assets].to_numpy(dtype=float, copy=True)
    positive = np.isfinite(raw) & (raw > 0)
    log = np.full_like(raw, np.nan)
    np.log(raw, out=log, where=positive)
    positions = {name: i for i, name in enumerate(assets)}
    q = np.stack(
        [
            log[:, positions[row[0]]] - (log[:, positions[row[1]]] if len(row) == 2 else 0.0)
            for row in legs
        ],
        axis=1,
    )
    close = pd.DataFrame(q, index=x.index, columns=pairs.target.tolist())
    moves = close.diff()
    sd = moves.shift(1).rolling(63, min_periods=42).std(ddof=0).clip(lower=1e-6)
    channels = []
    for lag in DRIVER_LAGS:
        # min_periods==lag requires every adjacent interval in the short path.
        signal = moves.rolling(lag, min_periods=lag).sum() / (sd * math.sqrt(lag))
        channels.append(signal.clip(-12, 12).to_numpy(dtype=float, copy=True))
    return np.stack(channels, axis=2)


def encode_response(drivers, y, pairs):
    """Learn target-specific response coefficients using only released pairs.

    Every window uses one shared pairwise-finite mask for x, y, xx, yy and xy.
    Count shrinkage is a fixed regularizer, not an effective-sample-size claim.
    Returned features omit the response intercept to avoid renaming raw priors.
    """
    import numpy as np
    import pandas as pd

    metadata_legs(pairs)
    signals = np.asarray(drivers, dtype=float)
    if (
        not isinstance(y, pd.DataFrame)
        or y.empty
        or len(y) > STOP
        or not y.columns.is_unique
        or y.columns.tolist() != pairs.target.tolist()
        or not pd.api.types.is_integer_dtype(y.index)
        or y.index.tolist() != list(range(len(y)))
        or signals.shape != (len(y), len(pairs), 2)
        or np.isinf(signals).any()
        or any(
            not pd.api.types.is_numeric_dtype(y[c]) or pd.api.types.is_bool_dtype(y[c]) for c in y
        )
    ):
        raise Stop("Outcome and driver axes/types are invalid.")
    values = y.to_numpy(dtype=float, copy=True)
    values[(~np.isfinite(values)) | (values == -999999)] = np.nan
    horizons = pairs.lag.to_numpy(dtype=int, copy=True)
    # These outcome origins could not be released by the last feature origin.
    for h in (1, 2, 3, 4):
        values[max(0, len(y) - h - 1) :, horizons == h] = np.nan
    encoded = np.full((len(y), len(pairs), 4), np.nan, dtype=float)
    summaries = []
    for k, _lag in enumerate(DRIVER_LAGS):
        for wi, window in enumerate(WINDOWS):
            for h in (1, 2, 3, 4):
                cols = np.flatnonzero(horizons == h)
                if len(cols) == 0:
                    continue
                a = pd.DataFrame(signals[:, cols, k], index=y.index).shift(h + 1)
                b = pd.DataFrame(values[:, cols], index=y.index).shift(h + 1)
                valid = a.notna() & b.notna()
                a = a.where(valid)
                b = b.where(valid)
                n = valid.astype(float).rolling(window, min_periods=1).sum()
                amin = window * 2 // 3
                ma = a.rolling(window, min_periods=amin).mean()
                mb = b.rolling(window, min_periods=amin).mean()
                va = (a * a).rolling(window, min_periods=amin).mean() - ma * ma
                vb = (b * b).rolling(window, min_periods=amin).mean() - mb * mb
                cov = (a * b).rolling(window, min_periods=amin).mean() - ma * mb
                sx = np.sqrt(va.where(va > 1e-10))
                sy = np.sqrt(vb.where(vb > 1e-16))
                rho = (cov / (sx * sy)).clip(-1, 1)
                shrink = n / (n + 63 * h)
                current = pd.DataFrame(signals[:, cols, k], index=y.index)
                feature = (shrink * rho * (current - ma) / sx).clip(-6, 6)
                encoded[:, cols, 2 * k + wi] = feature.to_numpy(dtype=float, copy=True)
                lo = min(252, len(y))
                hi = min(TRAIN_STOP, len(y))
                rvals = rho.iloc[lo:hi].to_numpy(dtype=float, copy=True)
                finite = np.isfinite(rvals)
                paired = n.iloc[lo:hi].to_numpy(dtype=float, copy=True)
                shrink_values = shrink.iloc[lo:hi].to_numpy(dtype=float, copy=True)
                usable = rvals[finite]
                summaries.append(
                    dict(
                        feature=RESPONSE_NAMES[2 * k + wi],
                        horizon=h,
                        window=window,
                        targets=len(cols),
                        training_rows=max(hi - lo, 0),
                        coefficient_cells=int(finite.sum()),
                        mean_paired_count=float(paired[finite].mean()) if finite.any() else None,
                        mean_shrinkage=float(shrink_values[finite].mean())
                        if finite.any()
                        else None,
                        positive_response_fraction=float((usable > 0).mean())
                        if len(usable)
                        else None,
                        rho_q05=float(np.quantile(usable, 0.05)) if len(usable) else None,
                        rho_q95=float(np.quantile(usable, 0.95)) if len(usable) else None,
                    )
                )
    return encoded, summaries


def build_response(x, y, pairs):
    import numpy as np

    if not x.index.equals(y.index):
        raise Stop("Market and outcome rows must align.")
    drivers = build_drivers(x, pairs)
    response, summaries = encode_response(drivers, y, pairs)
    block = np.concatenate([drivers, response], axis=2).astype(np.float32)
    if np.isinf(block).any():
        raise Stop("Response features contain infinity.")
    meta = dict(
        targets=len(pairs),
        templates=6,
        drivers=2,
        response_templates=4,
        coefficient_summaries=summaries,
        source_delay_by_horizon={str(h): h + 1 for h in (1, 2, 3, 4)},
        parameter_training="Rolling supervised moments of released origin-aligned pairs; no forecasting-model optimizer.",
        outcome_mean_added=False,
        coverage_columns_added=0,
        max_feature_label_origin={str(h): max(-1, len(y) - 1 - (h + 1)) for h in (1, 2, 3, 4)},
    )
    return block, DRIVER_NAMES + RESPONSE_NAMES, meta


def variant_names(names, variant):
    if variant not in VARIANTS or names != DRIVER_NAMES + RESPONSE_NAMES:
        raise Stop("Unknown or altered response panel.")
    return (
        DRIVER_NAMES
        if variant == "response_drivers"
        else RESPONSE_NAMES
        if variant == "response_encoded"
        else names
    )


def append_panel(base, block, names, variant):
    import numpy as np

    from commodity_prediction.domain.catalog import Panel

    chosen = variant_names(names, variant)
    if block.shape != (len(base.dates), len(base.targets), len(names)) or set(names) & set(
        base.names
    ):
        raise Stop("Response array does not align with frozen control.")
    panel = Panel(
        np.concatenate([base.values, block[:, :, [names.index(n) for n in chosen]]], axis=2),
        base.dates,
        base.targets,
        base.names + chosen,
        dict(base.source_series),
    )
    panel.validate()
    return panel


def smoke():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260912)
    x = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.015, (180, 5)), axis=0)), columns=list("ABCDE")
    )
    p = pd.DataFrame(
        {
            "target": [f"target_{i}" for i in range(4)],
            "pair": ["A - B", "B - C", "C - D", "D - E"],
            "lag": [1, 2, 3, 4],
        }
    )
    y = pd.DataFrame(rng.normal(0, 0.02, (180, 4)), columns=p.target)
    before = y.copy(deep=True)
    before_x = x.copy(deep=True)
    a, n, _ = build_response(x, y, p)
    b, _, _ = build_response(x.iloc[:151], y.iloc[:151], p)
    np.testing.assert_array_equal(a[:151], b)
    changed = y.copy(deep=True)
    changed.iloc[130:] += 1
    c, _, _ = build_response(x, changed, p)
    for j, h in enumerate(p.lag):
        np.testing.assert_array_equal(a[: 130 + h + 1, j], c[: 130 + h + 1, j])
    future_x = x.copy(deep=True)
    future_x.iloc[160:] *= 10
    d, _, _ = build_response(future_x, y, p)
    np.testing.assert_array_equal(a[:160], d[:160])
    pd.testing.assert_frame_equal(y, before, check_exact=True)
    pd.testing.assert_frame_equal(x, before_x, check_exact=True)
    if not np.isfinite(a[-1, :, 2:]).all():
        raise Stop("Synthetic response coverage failed.")
    return dict(
        status="DELAYED_RESPONSE_SMOKE_PASSED",
        exact_prefix=True,
        horizon_release_boundary_exact=True,
        input_unchanged=True,
        future_market_perturbation_exact=True,
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
    )


def audit_features(block, names, meta, base, pairs):
    import numpy as np

    def signature(a):
        a = np.array(a, dtype=np.float32, copy=True, order="C")
        a[np.isnan(a)] = np.nan
        return hashlib.sha256(a.tobytes()).hexdigest()

    existing = {signature(base.values[252:TRAIN_STOP, :, i]): n for i, n in enumerate(base.names)}
    rows = []
    for i, name in enumerate(names):
        history = block[252:TRAIN_STOP, :, i]
        valid = np.isfinite(history)
        v = history[valid]
        digest = signature(history)
        dup = existing.get(digest)
        if dup is None:
            existing[digest] = name
        rows.append(
            dict(
                name=name,
                kind="driver" if name in DRIVER_NAMES else "learned_response",
                training_finite_fraction=float(valid.mean()),
                targets_80pct_coverage=int((valid.mean(axis=0) >= 0.8).sum()),
                q05=float(np.quantile(v, 0.05)) if len(v) else None,
                q95=float(np.quantile(v, 0.95)) if len(v) else None,
                exact_duplicate_of=dup,
                training_sha256=digest,
            )
        )
    return rows


def fit_one(root, u, name, stage, lineage, base, block, names, y, pairs, fold, config, report):
    import joblib
    import pandas as pd

    from commodity_prediction.domain.attribution.run import fit_stage
    from commodity_prediction.domain.experiment import Experiment
    from commodity_prediction.domain.model import prepare, select

    if name not in VARIANTS or stage.exists():
        raise Stop("Undeclared/existing response stage; no automatic refit.")
    panel = append_panel(base, block, names, name)
    settings = {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
    stats = prepare(panel, y, fold.train_stop, settings)
    experiment = Experiment(name, algorithm="histogram")
    _, audit = select(stats, experiment.candidates(panel.names), settings, False)
    admitted = [n for n in audit["selected_names"] if n.startswith("delayed_response__")]
    numeric = [n for n in admitted if n in RESPONSE_NAMES]
    if not admitted or (name != "response_drivers" and not numeric):
        raise Stop("No proposed usable inputs survived training-only checks; no fit started.")
    if set(audit["rejection_reasons"]) & {"feature_budget", "correlated", "unstable_sign"}:
        raise Stop("Undeclared screening exclusion.")
    if report["fit_attempts_this_call"] >= 3:
        raise Stop("Three-fit limit exhausted.")
    report["fit_attempts_this_call"] += 1
    u.atomic_json(report_path(root), report)
    result = fit_stage(panel, y, pairs, fold, experiment, settings, stats, stage, lineage)
    saved = pd.read_parquet(stage / "predictions.parquet")
    replay = joblib.load(stage / "model.joblib").predict(panel, START, STOP)
    error = u.exact_predictions(saved, replay)
    recorded = {
        **result,
        "new_inputs_admitted": len(admitted),
        "response_inputs_admitted": len(numeric),
    }
    report["results"].append(recorded)
    report["new_training_fits"] += 1
    report["maximum_prediction_replay_error"] = max(
        error, report["maximum_prediction_replay_error"]
    )
    report["checkpoint_hashes"][name] = u.verify_stage(stage, lineage)
    u.atomic_json(report_path(root), report)
    emit(
        "RESPONSE_MODEL_VERIFIED",
        variant=name,
        completed=report["new_training_fits"],
        total=3,
        fold=FOLD,
        official_metric=result["metrics"]["official_metric"],
    )
    del panel, stats, replay
    gc.collect()


def decisions(results, control, metric):
    if len(results) != 3 or {r["variant"] for r in results} != set(VARIANTS):
        raise Stop("Incomplete response results.")
    original = metric(control["daily_rank_correlations"], 180)
    if not math.isclose(original, control["official_metric"], rel_tol=0, abs_tol=1e-12):
        raise Stop("Control score/daily mismatch.")
    scores = {r["variant"]: metric(r["metrics"]["daily_rank_correlations"], 180) for r in results}
    rows = []
    eligible = []
    for r in results:
        name = r["variant"]
        score = scores[name]
        if (
            r.get("fold") != FOLD
            or r.get("train_stop") != TRAIN_STOP
            or r.get("validation_start") != START
            or r.get("validation_stop") != STOP
        ):
            raise Stop("Wrong training/validation period.")
        if not math.isclose(score, r["metrics"]["official_metric"], abs_tol=1e-12, rel_tol=0):
            raise Stop("Score/daily mismatch.")
        chosen = r["selection"]["selected_names"]
        admitted = sum(n.startswith("delayed_response__") for n in chosen)
        response = sum(n in RESPONSE_NAMES for n in chosen)
        intended = variant_names(DRIVER_NAMES + RESPONSE_NAMES, name)
        if (
            admitted != r["new_inputs_admitted"]
            or response != r["response_inputs_admitted"]
            or admitted == 0
        ):
            raise Stop("Admission records disagree.")
        all_admitted = set(intended).issubset(chosen)
        delta = score - original
        increment = score - scores["response_drivers"]
        passed = (
            delta >= GAIN and all_admitted and (name == "response_drivers" or increment >= GAIN)
        )
        rows.append(
            dict(
                variant=name,
                official_metric=score,
                delta_vs_current_market=delta,
                delta_vs_drivers=increment,
                new_inputs_admitted=admitted,
                response_inputs_admitted=response,
                all_intended_inputs_admitted=all_admitted,
                passes_review_gate=bool(passed),
            )
        )
        if passed:
            eligible.append(name)
    return rows, eligible


def worker(root):
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    from commodity_prediction.data import Fold
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.market_path.features import build_panel
    from commodity_prediction.domain.run import load_panel
    from commodity_prediction.runtime import seal_checkpoint
    from commodity_prediction.studies.evaluation import compare_predictions, evaluate

    root = Path(root)
    session, older, u = previous(root)
    lineage, evidence = identity(root, u)
    directory = u.safe_path(root, "artifacts/response_encoding/" + lineage)
    if directory.exists():
        raise Stop("Existing response directory preserved; no blind rerun.")
    directory.mkdir(parents=True)
    began = time.monotonic()
    r = dict(
        project="commodity-prediction",
        status="RUNNING",
        source_commit=SHA,
        lineage=lineage,
        identity=evidence,
        plan=plan(),
        started_utc=utc(),
        new_training_fits=0,
        fit_attempts_this_call=0,
        control_refits=0,
        old_context_model_refits=0,
        maximum_prediction_replay_error=0.0,
        results=[],
        checkpoint_hashes={},
        feature_gate="open",
        promotion_allowed=False,
        fold=FOLD,
        validation_dates=180,
        final_test_evaluations=0,
        aws_api_calls=0,
        github_writes=False,
    )

    def save():
        r["elapsed_seconds"] = round(time.monotonic() - began, 3)
        u.atomic_json(report_path(root), r)

    def expire(*_):
        raise Stop("240-second response deadline reached; completed stages retained.")

    handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        save()
        ready = u.readiness(root)
        u.validate_runtime(root, ready)
        u.validate_source(root)
        prior = read_prior(root, u, True)
        parents = u.checkpoint_snapshot(root, ready)
        r["prior_context_comparisons"] = prior["comparison_rows"]
        r["prior_control_pooled"] = prior["prior_control_pooled"]
        r["historical_fold_controls"] = list(older.CONTROL_SCORES)
        emit("CHECK_RESPONSE_TIMING")
        r["contract_smoke"] = smoke()
        save()
        for name, pin in u.RAW.items():
            if u.digest(u.safe_path(root, "data/raw/" + name)) != pin:
                raise Stop("Raw checksum changed: " + name)
        final = u.read_json(root / "configs/final_evaluation.json")
        if final["evaluated"] or final["final_test_start_date_id"] != 1714:
            raise Stop("Final-test boundary changed.")
        x = pd.read_csv(root / "data/raw/train.csv", nrows=STOP).set_index("date_id")
        y = (
            pd.read_csv(root / "data/raw/train_labels.csv", nrows=STOP)
            .set_index("date_id")
            .replace(-999999, np.nan)
        )
        pairs = pd.read_csv(root / "data/raw/target_pairs.csv")
        if (
            len(x) != STOP
            or x.index.tolist() != list(range(STOP))
            or not x.index.equals(y.index)
            or list(y.columns) != [f"target_{i}" for i in range(424)]
            or pairs.target.tolist() != list(y.columns)
        ):
            raise Stop("Middle-fold input axes changed.")
        config = u.read_json(root / "configs/domain_study.json")
        u.atomic_json(directory / "plan.json", evidence)
        with threadpool_limits(limits=4):
            original = load_panel(root / "artifacts" / FEATURE / "features")
            prefix = Panel(
                original.values[:STOP],
                original.dates[:STOP],
                original.targets,
                original.names,
                dict(original.source_series),
            )
            base, _ = build_panel(prefix, x, pairs, "current_market")
            del prefix, original
            gc.collect()
            cp = root / "artifacts" / PARENT / "fold_1/current_market"
            saved = pd.read_parquet(cp / "predictions.parquet")
            replay = joblib.load(cp / "model.joblib").predict(base, START, STOP)
            u.exact_predictions(saved, replay)
            control = evaluate(y.loc[saved.index], saved, pairs)
            if not session.close(
                control["official_metric"], CONTROL
            ) or saved.index.tolist() != list(range(START, STOP)):
                raise Stop("Saved middle-fold control did not replay at the expected score.")
            r["control"] = control
            emit("SAVED_FOLD_1_CONTROL_REPLAY_PASSED", score=CONTROL, refits=0)
            emit("BUILD_DELAYED_RESPONSE_FEATURES", drivers=2, learned_responses=4)
            block, names, meta = build_response(x, y, pairs)
            prefixes = [TRAIN_STOP, START, STOP - 1]
            for stop in prefixes:
                short, sn, sm = build_response(x.iloc[:stop], y.iloc[:stop], pairs)
                np.testing.assert_array_equal(short, block[:stop])
                if sn != names or sm["source_delay_by_horizon"] != meta["source_delay_by_horizon"]:
                    raise Stop("Feature definition changed with prefix.")
            r["causal_checks"] = dict(
                exact_prefixes=prefixes,
                new_feature_labels_used=True,
                maximum_feature_label_origins=plan()["maximum_feature_label_origins"],
                origin_paired_release_delay="horizon+1",
                parameters_use_only_released_pairs=True,
            )
            r["encoding_metadata"] = meta
            r["feature_audit"] = audit_features(block, names, meta, base, pairs)
            stage = directory / "features"
            stage.mkdir()
            np.savez_compressed(stage / "response.npz", values=block, dates=np.arange(STOP))
            u.atomic_json(
                stage / "inventory.json",
                dict(metadata=meta, targets=pairs.target.tolist(), plan=plan()),
            )
            seal_checkpoint(stage, lineage, ["response.npz", "inventory.json"])
            r["feature_checkpoint_hashes"] = u.verify_stage(stage, lineage)
            save()
            fold = Fold(FOLD, TRAIN_STOP, START, STOP)
            for name in VARIANTS:
                emit(
                    "FIT_RESPONSE_PANEL",
                    variant=name,
                    completed=r["new_training_fits"],
                    total=3,
                    fold=FOLD,
                )
                fit_one(
                    root,
                    u,
                    name,
                    directory / "fold_1" / name,
                    lineage,
                    base,
                    block,
                    names,
                    y,
                    pairs,
                    fold,
                    config,
                    r,
                )
            rows, eligible = decisions(r["results"], control, session.metric)
            daily = {"current_market": np.asarray(control["daily_rank_correlations"])}
            daily.update(
                {
                    v["variant"]: np.asarray(v["metrics"]["daily_rank_correlations"])
                    for v in r["results"]
                }
            )
            r["comparisons"] = compare_predictions(daily, [180], plan()["contrasts"], config)
            r.update(
                comparison_rows=rows,
                candidates_for_review=eligible,
                decision="REVIEW_RESPONSE_ON_OTHER_PERIODS"
                if eligible
                else "STOP_TESTED_RESPONSE_PANELS",
            )
        r["parents_unchanged"] = parents == u.checkpoint_snapshot(root, ready)
        if not r["parents_unchanged"]:
            raise Stop("Frozen parent checkpoint changed.")
        read_prior(root, u, True)
        u.validate_source(root)
        r.update(
            status="RESPONSE_REVIEW_READY",
            source_unchanged=True,
            context_parent_unchanged=True,
            finished_utc=utc(),
            limitations=[
                "Repeatedly inspected middle development fold. No fresh confirmatory evidence or full adaptive-search correction.",
                "Supervised feature coefficients update from released labels; downstream histogram model remains fixed.",
                "The two original price drivers are a separate control. A response-panel gain alone does not establish the benefit of online updating versus frozen coefficients.",
                "The fixed 63*h shrinkage pseudo-count is heuristic; overlapping outcomes are not independent samples.",
                "Window lengths and panels are fixed before this result. No validation-based coefficient/window tuning.",
                "Only active-control and within-family exact duplicates audited; no exhaustive historical numerical deduplication claim.",
                "No source resets, package installs, cloud calls, old control fits or final-test evaluation.",
                "Private feature/model ZIP is not off-disk backup until separately downloaded.",
            ],
        )
        save()
        u.atomic_json(directory / "review.json", r)
        return r
    except BaseException as exc:
        r.update(status="STOPPED", error=type(exc).__name__ + ": " + str(exc), finished_utc=utc())
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def verify_complete(root, r, session, u):
    lineage, evidence = identity(root, u)
    read_prior(root, u)
    if (
        r.get("status") not in TERMINAL
        or r.get("lineage") != lineage
        or r.get("identity") != evidence
        or r.get("new_training_fits") != 3
        or r.get("fit_attempts_this_call") != 3
        or r.get("control_refits") != 0
        or r.get("maximum_prediction_replay_error") != 0
        or r.get("final_test_evaluations") != 0
        or r.get("parents_unchanged") is not True
        or r.get("source_unchanged") is not True
        or r.get("context_parent_unchanged") is not True
        or r.get("causal_checks", {}).get("new_feature_labels_used") is not True
        or r.get("causal_checks", {}).get("maximum_feature_label_origins")
        != plan()["maximum_feature_label_origins"]
        or r.get("contract_smoke", {}).get("status") != "DELAYED_RESPONSE_SMOKE_PASSED"
        or r.get("fold") != FOLD
        or set(r.get("checkpoint_hashes", {})) != set(VARIANTS)
        or not session.close(r.get("control", {}).get("official_metric", float("nan")), CONTROL)
        or r.get("control", {}).get("date_ids") != list(range(START, STOP))
    ):
        raise Stop("Not a complete verified response-encoding study.")
    rows, eligible = decisions(r["results"], r["control"], session.metric)
    decision = "REVIEW_RESPONSE_ON_OTHER_PERIODS" if eligible else "STOP_TESTED_RESPONSE_PANELS"
    if (
        rows != r["comparison_rows"]
        or eligible != r["candidates_for_review"]
        or decision != r["decision"]
    ):
        raise Stop("Stored decision differs from the declared computation.")
    directory = root / "artifacts/response_encoding" / lineage
    u.verify_stage(directory / "features", lineage, r["feature_checkpoint_hashes"])
    for name, pins in r["checkpoint_hashes"].items():
        stage = directory / "fold_1" / name
        u.verify_stage(stage, lineage, pins)
        record = next(v for v in r["results"] if v["variant"] == name)
        if u.read_json(stage / "result.json") != {
            k: v
            for k, v in record.items()
            if k not in ["new_inputs_admitted", "response_inputs_admitted"]
        }:
            raise Stop("Report differs from sealed fitted result.")


def run(root=ROOT):
    root = Path(root)
    session, older, u = previous(root)
    ready = u.readiness(root)
    u.validate_runtime(root, ready)
    u.validate_source(root)
    read_prior(root, u)
    out = u.safe_path(root, "logs/manual_response_encoding")
    out.mkdir(parents=True, exist_ok=True)
    with u.safe_path(out, "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report_path(root).exists():
            r = u.read_json(report_path(root))
            if r.get("status") not in TERMINAL:
                raise Stop(
                    "Interrupted response run exists. Preserve its report; no automatic retry or budget reset."
                )
            verify_complete(root, r, session, u)
            emit("COMPLETED_RESPONSE_REUSED", new_fits=0)
            return {**r, "new_fits_this_notebook_call": 0}
        lineage, _ = identity(root, u)
        if (root / "artifacts/response_encoding" / lineage).exists():
            raise Stop("Orphan response directory preserved; no refit.")
        if shutil.disk_usage(root).free < 1024**3:
            raise Stop("Need 1 GiB free; do not delete old projects blindly.")
        u.atomic_json(
            report_path(root),
            dict(status="STARTING_WORKER", lineage=lineage, new_training_fits=0, started_utc=utc()),
        )
        args = [
            ready["runtime"]["python"],
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--root",
            str(root),
        ]
        began = time.monotonic()
        try:
            session._supervise_command(
                args, root, LIMIT, out / "response_encoding_worker.log", u.environment(root)
            )
        except BaseException as exc:
            r = u.read_json(report_path(root))
            if r.get("status") == "STARTING_WORKER":
                r["new_training_fits"] = None
            r.update(
                status="STOPPED",
                supervisor_error=type(exc).__name__ + ": " + str(exc),
                supervised_seconds=round(time.monotonic() - began, 3),
            )
            u.atomic_json(report_path(root), r)
            raise
        r = u.read_json(report_path(root))
        verify_complete(root, r, session, u)
        r.update(
            supervised_seconds=round(time.monotonic() - began, 3), new_fits_this_notebook_call=3
        )
        u.atomic_json(report_path(root), r)
        return r


def charts(r):
    import numpy as np
    import plotly.graph_objects as go

    figures = []

    def add(f, title, x="", y=""):
        f.update_layout(
            title=title,
            height=480,
            margin=dict(l=80, r=35, t=90, b=100),
            xaxis_title=x,
            yaxis_title=y,
            legend=dict(orientation="h", y=-0.30),
        )
        figures.append(f)

    old = r["prior_context_comparisons"]
    add(
        go.Figure(
            go.Bar(x=[a["variant"] for a in old], y=[a["delta_vs_current_market"] for a in old])
        ),
        "1 · Completed instrument-context result — do not rerun",
        "Previous panel",
        "Middle-fold score change",
    )
    audit = r["feature_audit"]
    add(
        go.Figure(
            go.Bar(x=[a["name"] for a in audit], y=[a["training_finite_fraction"] for a in audit])
        ),
        "2 · Observed training features — missing pairs stay missing",
        "Candidate",
        "Finite training fraction",
    )
    summaries = r["encoding_metadata"]["coefficient_summaries"]
    f = go.Figure()
    for h in (1, 2, 3, 4):
        rows = [a for a in summaries if a["horizon"] == h]
        f.add_bar(
            x=[a["feature"] for a in rows],
            y=[a["mean_paired_count"] for a in rows],
            name=f"Horizon {h}",
        )
    add(
        f,
        "3 · Same-mask historical pairs used by response estimates",
        "Response feature",
        "Mean paired count (training)",
    )
    f = go.Figure()
    for h in (1, 2, 3, 4):
        rows = [a for a in summaries if a["horizon"] == h]
        f.add_scatter(
            x=[a["feature"] for a in rows],
            y=[a["positive_response_fraction"] for a in rows],
            mode="lines+markers",
            name=f"Horizon {h}",
        )
    add(
        f,
        "4 · Positive versus negative historical response — training only",
        "Response feature",
        "Fraction of positive correlations",
    )
    limits = r["causal_checks"]["maximum_feature_label_origins"]
    add(
        go.Figure(
            go.Bar(
                x=[f"Horizon {h}" for h in (1, 2, 3, 4)], y=[limits[str(h)] for h in (1, 2, 3, 4)]
            )
        ),
        "5 · Latest label origin permitted at the last prediction",
        "Target horizon",
        "Maximum released source origin",
    )
    rows = r["comparison_rows"]
    names = [a["variant"] for a in rows]
    add(
        go.Figure(
            go.Bar(
                x=["current_market"] + names,
                y=[r["control"]["official_metric"]] + [a["official_metric"] for a in rows],
            )
        ),
        "6 · Same middle-fold comparison — no pooled-score substitution",
        "Representation",
        "Official metric on 180 dates",
    )
    f = go.Figure()
    f.add_bar(x=names, y=[a["delta_vs_current_market"] for a in rows], name="Against saved control")
    f.add_bar(x=names, y=[a["delta_vs_drivers"] for a in rows], name="Against driver-only panel")
    f.update_layout(barmode="group")
    f.add_hline(y=GAIN, line_dash="dash", annotation_text="+0.002 review threshold")
    add(
        f,
        "7 · Does learned response add value beyond its price drivers?",
        "Panel",
        "Matched official-metric change",
    )
    f = go.Figure()
    for row in r["comparisons"]:
        if row["block_dates"] == 20 and row["reference"] == "current_market":
            lo, hi = row["conditional_95_interval"]
            f.add_scatter(x=[lo, hi], y=[row["variant"]] * 2, mode="lines", showlegend=False)
            f.add_scatter(x=[row["delta"]], y=[row["variant"]], mode="markers", name=row["variant"])
    f.add_vline(x=0, line_dash="dash")
    add(
        f,
        "8 · Conditional 20-date intervals — not whole-search correction",
        "Delta versus original control",
        "Panel",
    )
    f = go.Figure()
    control = np.asarray(r["control"]["daily_rank_correlations"])
    for row in r["results"]:
        f.add_scatter(
            x=list(range(START, STOP)),
            y=np.cumsum(np.asarray(row["metrics"]["daily_rank_correlations"]) - control).tolist(),
            name=row["variant"],
        )
    add(
        f,
        "9 · Temporal concentration — correlation difference, NOT trading profit",
        "Prediction origin",
        "Cumulative daily-correlation difference",
    )
    f = go.Figure()
    f.add_bar(
        x=names,
        y=[a["new_inputs_admitted"] - a["response_inputs_admitted"] for a in rows],
        name="Price drivers admitted",
    )
    f.add_bar(
        x=names, y=[a["response_inputs_admitted"] for a in rows], name="Response encodings admitted"
    )
    f.update_layout(barmode="stack")
    add(
        f,
        "10 · Actual new inputs admitted — not causal importance",
        "Panel",
        "Number of added templates",
    )
    return figures


def private_backup(root, r, u):
    base = root / "artifacts/response_encoding" / r["lineage"]
    paths = [base / "plan.json", base / "review.json"] + [
        base / "features" / n for n in r["feature_checkpoint_hashes"]
    ]
    for name in VARIANTS:
        paths += [base / "fold_1" / name / n for n in r["checkpoint_hashes"][name]]
    pins = {
        str(p.relative_to(root)): u.digest(u.safe_path(root, str(p.relative_to(root))))
        for p in paths
    }
    if sum(p.stat().st_size for p in paths) > 256 * 1024**2:
        raise Stop("Backup exceeds 256 MiB; fitted stages preserved.")
    dest = u.safe_path(output_dir(root), "response_encoding_checkpoints.zip")
    tmp = u.safe_path(dest.parent, dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, str(p.relative_to(root)))
        z.writestr("SHA256SUMS.json", json.dumps(pins, sort_keys=True, indent=2))
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():
            raise Stop("Backup CRC verification failed.")
        for name, h in pins.items():
            if hashlib.sha256(z.read(name)).hexdigest() != h:
                raise Stop("Backup checksum differs.")
    os.replace(tmp, dest)
    return dict(
        path=str(dest),
        sha256=u.digest(dest),
        bytes=dest.stat().st_size,
        files=len(paths),
        private=True,
        contains_label_derived_features_models_predictions=True,
        off_disk_backup_confirmed=False,
        raw_csvs_included=False,
    )


def finish(root, report, figures):
    root = Path(root)
    session, older, u = previous(root)
    verify_complete(root, report, session, u)
    if len(figures) != 10:
        raise Stop("Expected ten Plotly figures.")
    r = dict(report)
    path = output_dir(root) / "response_encoding_dashboard.html"
    r.update(
        dashboard=str(path),
        dashboard_sha256=older.export_dashboard(
            path,
            "Delayed-response feature ablation",
            [(str(f.layout.title.text), f) for f in figures],
        ),
        plotly_figures=10,
    )
    try:
        r["private_checkpoint_bundle"] = private_backup(root, r, u)
    except Exception as exc:
        r["backup_warning"] = type(exc).__name__ + ": " + str(exc)
    r.update(
        status="NOTEBOOK_AND_RESPONSE_READY",
        notebook=str(root / "notebooks" / NOTEBOOK),
        updated_utc=utc(),
    )
    u.atomic_json(report_path(root), r)
    return r


def protocol_markdown():
    return "# Delayed conditional response encoding — notebook 12\n\n## Measured starting point\nNotebook 11 completed three middle-fold fits. Identity scored 0.15334865292560604,\nidentity-aligned shocks 0.1336489611227173, and their union 0.12888769504762737,\nversus the saved current_market score 0.16704165319061334. None passed its gate.\nThose panels stay closed unchanged. These are reported results, not new replays\nperformed while preparing this package. The pooled 535-date control is 0.3097087232124053.\n\n## Hypothesis (not a demonstrated predictive relationship)\nEncode each target's historically estimated RESPONSE to a market signal, rather\nthan add more static identity indicators or raw transforms and require a single\npooled learner to infer every changing response. Learned feature coefficients are\nsupervised. The downstream histogram forecaster and its settings stay unchanged.\n\nTwo drivers use the directed log-price spread, or a single log price for a single\nasset target: 1-row and 5-row changes, scaled by sqrt(lag) times trailing 63-row\nspread-return SD through t-1 (minimum 42 observations; floor 1e-6). Driver values\nare clipped to [-12,12]. No prices are forward-filled, and the short path requires\nall adjacent observations. All target legs must be present in input metadata.\n\nFor target j with horizon h, let d=h+1. At prediction t, pair signal s[u,j] with\nits corresponding outcome y[u,j] only for u<=t-d. For each 63/126-row window,\ncompute all moments over the SAME pairwise-finite observations (minimum 42/84).\nNo origin u is paired with an outcome from a different origin. Missing outcomes\nare never economic zeros. Pair counts are used internally, not added as features.\n\nThe centered standardized response feature is\n\n    n/(n + 63*h) * corr(s,y) * (s[t]-mean(s))/sd(s)\n\nclipped to [-6,6]. Target variance <=1e-16 or signal variance <=1e-10 makes the\nfeature missing. No outcome intercept is added: this does not rename the existing\nreleased-target mean priors. The fixed 63*h pseudo-count conservatively attenuates\nnoisy overlapping-horizon estimates; it is NOT a proven effective-sample formula.\nNo selection of windows or shrinkage strength on validation is permitted.\n\nThe feature states update on already-released earlier validation outcomes. This\nis explicitly prequential supervised feature generation, NOT zero-learning\npreprocessing and NOT refitting the downstream forecasting model on validation.\nThe latest usable feature-label origins at prediction 1528 are 1526,1525,1524,1523\nfor horizons 1,2,3,4. No data at or beyond the final-test boundary are evaluated.\nThe historical parent feature panel is reused under its existing release contract.\n\n## Matched experiment\nThree new models on middle development fold 1 only:\n\n1. response_drivers: two label-free spread drivers added to current_market.\n2. response_encoded: four centered learned responses added to current_market.\n3. response_joint: exact union of the two and four added features.\n\nNo new availability or identity indicators. Model settings, training preprocessing,\nand target standardization remain fixed. Exact duplicates, constants and missingness\nare checked using the fitting interval. The declarations are frozen before results.\n\nTraining ends at 1343 (stop1344); validation origins1349-1528. This period has\nalready informed research. It is exploratory, not fresh confirmation. At most\nthree fits within one240-second worker. Existing models must replay exactly.\nA previously interrupted attempt cannot be silently rerun or have its budget reset.\nCompleted stages are retained. A completed report can be re-opened without fitting.\n\nDrivers earn review with >=0.002 gain over the saved same-period control. The\nencoded/joint panels must gain >=0.002 over BOTH the saved control and drivers.\nAll intended added inputs must be admitted and integrity/replay checks pass.\nPassing only earns review; no automatic other folds or promotion. Five conditional\nblock contrasts at10,20,40 dates do not correct all adaptive research history.\nA future frozen-coefficient comparison would be required to isolate the value of\nsequential updating itself; this experiment tests the combined response encoding.\n\n## Novelty / overlap boundary\nExisting source has raw released target means/risk/ranks and asset-factor return\ncovariances. This recipe pairs TARGET outcomes with their own ORIGIN-aligned\nmarket signals under each horizon release delay. It is not another target-rank\ntransform, asset-to-asset lead-lag graph, or anchor one-hot array. The exact\nnumerical duplicate audit covers the active87-template baseline and this family,\nnot every historical abandoned representation. Positive performance is unproven.\n\n## Primary sources reviewed for rationale\n- Stefan Nagel, Evaporating Liquidity, NBER17653 (2011):\n  https://www.nber.org/papers/w17653\n  Equity reversal returns vary with market conditions. This motivates allowing\n  response changes; it is not evidence this commodity feature predicts well.\n- Lonnie, Mitsui15th-place participant writeup (2026):\n  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th\n  Describes adaptation to newly released labels. We do not reproduce that neural\n  ensemble, its model-retraining schedule, or claim its score for our folds.\n- Repo domain/features.py (pinned d142a4cb57a5c4b2880f9341619e13a735b1cddc):\n  existing per-target release delay h+1 is retained, not weakened.\n\n## Deliverables and preservation\nCanonical source/config are unchanged; new script/config/protocol/notebook are\nlocal additions, not automatically pushed to GitHub. The report and self-contained\nPlotly HTML contain evidence, not raw target tables. The private ZIP includes\nlabel-derived features, models and predictions: keep it private and download for\nan off-disk copy. Stop the SageMaker application, never delete its space. Keep\nolder project folders because the notebook kernel reuses their verified environment.\n"


def notebook_document():
    cells = []

    def md(s):
        cells.append(
            dict(cell_type="markdown", metadata={}, id=f"response12-{len(cells):02}", source=s)
        )

    def code(s):
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                id=f"response12-{len(cells):02}",
                source=s,
                execution_count=None,
                outputs=[],
            )
        )

    md(
        "# Commodity prediction · Delayed response features\n\n"
        "**Milestone12:** three matched, bounded feature ablations on the middle development period. "
        "The downstream learner stays fixed; the new representation learns rolling signal/outcome relationships only after labels are released. "
        "No final test, old-model refits, package installs, or automatic additional periods."
    )
    code(
        "from pathlib import Path\nimport importlib.util\nimport pandas as pd\nROOT=Path('/home/sagemaker-user/projects/commodity-prediction-manual')\n"
        "spec=importlib.util.spec_from_file_location('response12_notebook',ROOT/'scripts/commodity_response_encoding.py')\n"
        "support=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(support)\n"
        "session,older,u=support.previous(ROOT)\nready=u.readiness(ROOT)\nu.validate_runtime(ROOT,ready)\nu.validate_source(ROOT)\nprior=support.read_prior(ROOT,u)\n"
        "print('Source:',support.SHA)\nprint('Verified runtime:',ready['runtime']['python'])\n"
    )
    md(
        "## Prior result: completed, negative, preserved\n\nInstrument identity and its aligned shocks did not improve this period. "
        "Do not rerun notebook11. A new supervised encoding asks a different question; it is not established as useful merely because it is more elaborate."
    )
    code(
        "display(pd.DataFrame(prior['comparison_rows']))\nprint('Prior decision:',prior['decision'])\n"
    )
    md(
        "## Freeze the comparison before results\n\nTwo base price signals versus four delayed response encodings versus their exact union. "
        "No additional coverage/identity columns. Coefficients use paired historical signal/outcome origins and horizon-specific release delays. "
        "This is supervised feature learning, not a new optimized ensemble. All three models use the existing histogram configuration."
    )
    code(
        "display(pd.DataFrame({'panel':support.VARIANTS,'added_numeric_templates':[2,4,6]}))\n"
        "print('Training stop:',support.TRAIN_STOP,'validation:',support.START,'through',support.STOP-1)\n"
        "print('Same-period saved control:',support.CONTROL)\nprint('Maximum three new fits; 240-second supervised worker.')\n"
    )
    md(
        "## Execute once\n\nThe worker verifies source, data, runtime, saved models and release-timing smoke tests before fitting. "
        "Feature coefficients may update from legally released earlier validation outcomes; forecasting-model parameters stay fixed. "
        "A reported error requires diagnosis, not repeated Run All or a budget reset."
    )
    code(
        "report=support.run(ROOT)\nprint('RESULT:',report['status'])\nprint('New fits:',report['new_fits_this_notebook_call'])\n"
        "print('DECISION:',report['decision'])\nfigures=support.charts(report)\n"
    )
    captions = [
        (
            "Preserve the measured negative result",
            "Previous context results use the same middle fold; these bars are historical evidence.",
        ),
        (
            "Measure usable feature coverage",
            "All price legs must be present. Insufficient paired history stays missing, not zero-filled outcomes.",
        ),
        (
            "Estimate from matched pairs",
            "The same finite sample enters every mean, variance and covariance. More observations do not prove more independent information.",
        ),
        (
            "Allow response direction to change",
            "A positive correlation can indicate continuation; a negative one can indicate reversal. These descriptive training fractions are not causal explanations or a new validation score.",
        ),
        (
            "Enforce availability by prediction origin",
            "Horizons1–4 have delays2–5. The latest possible origin is shown, not an assertion that its label is nonmissing.",
        ),
        (
            "Compare the same180dates",
            "Do not compare these scores directly to the pooled535-date result or the competition leaderboard.",
        ),
        (
            "Separate new drivers from learned responses",
            "Encoded/joint panels must beat both the original control and the driver-only fit before further review.",
        ),
        (
            "Keep uncertainty visible",
            "Paired blocks condition on the fitted models and this family. The historical adaptive search is not multiplicity-corrected.",
        ),
        (
            "Check concentration in time",
            "Cumulative correlation differences are NOT investment returns. A few dates can dominate a point estimate.",
        ),
        (
            "Confirm actual admission",
            "Counts establish which features were available to the fitted model; they are not permutation or causal importance.",
        ),
    ]
    for i, (title, text) in enumerate(captions):
        md("## " + title + "\n\n" + text)
        code(f"figures[{i}].show()\n")
    md(
        "## Save, download, stop\n\nA passing gate earns replication, not promotion. Save this notebook, download the report/HTML/private ZIP and stop the space. "
        "Do not publish the private ZIP or delete old projects."
    )
    code(
        "report=support.finish(ROOT,report,figures)\ndisplay(pd.DataFrame(report['comparison_rows']))\n"
        "print('RESULT:',report['status'])\nprint('DECISION:',report['decision'])\nprint('REPORT:',support.report_path(ROOT))\n"
        "print('DASHBOARD:',report['dashboard'])\nprint('PRIVATE BACKUP:',report.get('private_checkpoint_bundle',{}).get('path',report.get('backup_warning')))\n"
        "print('Ctrl+S, download evidence, STOP SPACE. No automatic shutdown.')\n"
    )
    md(
        "## Research sources and limits\n\n"
        "The recipe is a new hypothesis, not a reproduction of a published score.\n\n"
        "- [Nagel: time-varying equity reversal/liquidity returns](https://www.nber.org/papers/w17653)\n"
        "- [Mitsui participant: adaptation with released labels](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th)\n"
        "- Exact formula, missingness, shrinkage, comparison gate and overlap limitations: `docs/manual_response_encoding.md`."
    )
    return dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(
                display_name="Commodity - manual (verified)",
                language="python",
                name="commodity-manual",
            ),
            language_info=dict(name="python", version="3.12"),
            commodity_milestone="delayed-response-fold-1",
        ),
        nbformat=4,
        nbformat_minor=5,
    )


def install(root=ROOT):
    root = Path(root)
    session, older, u = previous(root)
    u.validate_source(root)
    read_prior(root, u)
    emit("INSTALL_RESPONSE_NOTEBOOK", source_commit=SHA, new_training_fits=0)
    nb = notebook_document()
    paths = {
        u.safe_path(root, "scripts/commodity_response_encoding.py"): Path(__file__).read_bytes(),
        u.safe_path(root, "configs/manual_response_encoding.json"): (
            json.dumps(plan(), sort_keys=True, indent=2) + "\n"
        ).encode(),
        u.safe_path(root, "docs/manual_response_encoding.md"): protocol_markdown().encode(),
        u.safe_path(root, "notebooks/" + NOTEBOOK): (json.dumps(nb, indent=1) + "\n").encode(),
    }

    def cell_source(c):
        return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]

    for p, b in paths.items():
        if p.exists():
            if p.suffix == ".ipynb":
                existing = u.read_json(p)
                if [(c["cell_type"], cell_source(c)) for c in existing["cells"]] != [
                    (c["cell_type"], cell_source(c)) for c in nb["cells"]
                ]:
                    raise Stop("Existing notebook edits preserved: " + str(p))
            elif p.read_bytes() != b:
                raise Stop("Existing different file preserved: " + str(p))
    for p, b in paths.items():
        if not p.exists():
            u.write_new(p, b)
    print(
        "RESULT: RESPONSE_NOTEBOOK_READY\nNOTEBOOK: "
        + str(root / "notebooks" / NOTEBOOK)
        + "\nKERNEL: Commodity - manual (verified)",
        flush=True,
    )
    return dict(new_training_fits=0, files=[str(p) for p in paths])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    old_handler = None

    def install_deadline(*_):
        raise Stop("90-second setup deadline reached; existing files preserved.")

    try:
        if args.worker:
            worker(args.root)
        else:
            old_handler = signal.signal(signal.SIGALRM, install_deadline)
            signal.alarm(90)
            install(args.root)
    except BaseException as exc:
        print("RESULT: STOPPED\nERROR: " + type(exc).__name__ + ": " + str(exc), flush=True)
        print(
            "Preserve report/error. Do not repeat unchanged. Stop space manually, never Delete space.",
            flush=True,
        )
        return 2
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
