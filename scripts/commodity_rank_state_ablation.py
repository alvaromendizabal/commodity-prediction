#!/usr/bin/env python3
"""Install notebook 10, then run three causal released-rank feature ablations.

Default invocation installs local files only. Notebook execution is bounded to
240 seconds, replays the saved control, and fits at most three new CPU models.
No downloads, package installations, cloud/Git writes or final-test evaluation.
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
DIAGNOSIS = "2d7bf39364262dea9ba091a6f67899ab8c40fee9f7efd4a43c724a28ffa8807b"
DIAGNOSIS_HELPER = "a8991b0e9b5517cf5c6c19659084c999aec3cfb7addb28252470b738df5ef80b"
DIAGNOSIS_REPORT = "7003517459b1fbb71c1dfe6e131de47ce1d7c36243d324a0c33166f6cb727c48"
TRAIN_STOP, START, STOP, DELAY = 1164, 1169, 1349, 5
LIMIT, GAIN = 240.0, 0.002
CONTROL = 0.40338108742296147
NOTEBOOK = "10_released_rank_ablation.ipynb"
SHORTLIST = [
    "released_rank_state__global_innovation_63",
    "released_rank_state__horizon_innovation_63",
    "released_rank_state__global_latest",
    "released_rank_state__horizon_latest",
    "released_rank_state__global_minus_horizon_latest",
    "released_rank_state__global_mean_21",
]
GLOBAL = [SHORTLIST[0], SHORTLIST[2], SHORTLIST[5]]
RELATIVE = [SHORTLIST[1], SHORTLIST[3], SHORTLIST[4]]
VARIANTS = {
    "rank_global": GLOBAL,
    "rank_horizon_relative": RELATIVE,
    "rank_joint": GLOBAL + RELATIVE,
}
NUMERIC = [
    f"released_rank_state__{g}_{f}"
    for g in ("global", "horizon")
    for f in ("latest", "mean_21", "mean_63", "trend_21_63", "innovation_63")
] + [
    "released_rank_state__global_minus_horizon_latest",
    "released_rank_state__global_minus_horizon_mean_63",
]
COVERAGE = ["released_rank_state__cohort_observed_fraction", "released_rank_state__own_coverage_63"]
TERMINAL = {"RANK_STATE_REVIEW_READY", "NOTEBOOK_AND_RANK_STATE_READY"}


class Stop(RuntimeError):
    """Preserve the evidence; diagnose an interrupted run before another attempt."""


def utc() -> str:
    return datetime.now(UTC).isoformat()


def emit(stage: str, **fields) -> None:
    print(json.dumps({"utc": utc(), "stage": stage, **fields}, allow_nan=False), flush=True)


def previous(root: Path):
    """Import the already-corrected feature builder, never invoke its old study."""
    path = Path(root) / "scripts/commodity_feature_diagnosis.py"
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != DIAGNOSIS_HELPER
    ):
        raise Stop("The corrected notebook-09 helper is missing or changed; preserve it.")
    spec = importlib.util.spec_from_file_location("rank10_verified_diagnosis", path)
    diagnosis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnosis)
    network, session, older, u = diagnosis.previous(Path(root))
    return diagnosis, session, older, u


def output_dir(root):
    return Path(root) / "logs/manual_rank_state"


def report_path(root):
    return output_dir(root) / "commodity_rank_state_report.json"


def plan() -> dict:
    return {
        "study": "released_ordinal_rank_state_matched_first_fold",
        "source_commit": SHA,
        "parent_lineage": PARENT,
        "rank_candidate_parent_lineage": DIAGNOSIS,
        "shortlist_from_training_only": SHORTLIST,
        "variants": VARIANTS,
        "numerical_candidate_counts": {n: len(v) for n, v in VARIANTS.items()},
        "coverage_inputs_added": 0,
        "source_origin_delay": DELAY,
        "training_stop_exclusive": TRAIN_STOP,
        "warmup_dates": 252,
        "validation_start": START,
        "validation_stop_exclusive": STOP,
        "maximum_feature_label_origin": STOP - 1 - DELAY,
        "feature_update_policy": "Frozen forecasting models; feature state updates using same-origin outcomes released at least five rows earlier. Later validation predictions may use already-released earlier validation outcomes; never contemporaneous/future outcomes.",
        "reference_formula": "Average tied ranks mapped to [-1,1] among observed labels, minimum three observations; no zero-filled target labels. EWM spans 21/63, min periods 14/42, adjust=False, ignore_na=False.",
        "selection_policy": "Fixed six-name notebook-09 training shortlist; no validation re-selection and no new coverage columns. Same parent preprocessing and fixed histogram learner.",
        "new_fit_limit": 3,
        "worker_limit_seconds": LIMIT,
        "control_refits": 0,
        "old_network_fits": 0,
        "graph_reconstructions": 0,
        "gate_gain_over_saved_control": GAIN,
        "gate": "At least +0.002 same-fold official metric vs saved current_market, exact replay, fixed shortlist and all selected numerical inputs admitted. Review only; no automatic other folds.",
        "rule_timing": "Declared after the notebook-09 training screen and negative network ablations, before rank-feature validation scores. The development fold has already informed prior research.",
        "contrasts": [(n, "current_market") for n in VARIANTS]
        + [("rank_joint", "rank_global"), ("rank_joint", "rank_horizon_relative")],
        "other_folds": False,
        "feature_gate": "open",
        "promotion_allowed": False,
        "final_test_evaluations": 0,
        "aws_api_calls": 0,
        "github_writes": False,
    }


def validate_diagnosis(report: dict) -> None:
    if (
        report.get("status") != "NOTEBOOK_AND_DIAGNOSIS_READY"
        or report.get("lineage") != DIAGNOSIS
        or report.get("decision") != "STOP_TESTED_NETWORK_VARIANTS"
        or report.get("new_training_fits") != 2
        or report.get("control_refits") != 0
        or report.get("rank_feature_models_fitted") != 0
        or report.get("maximum_prediction_replay_error") != 0.0
        or report.get("parents_unchanged") is not True
        or report.get("final_test_evaluations") != 0
    ):
        raise Stop("The notebook-09 result is not the reviewed completed diagnosis.")
    lab = report["rank_lab"]
    if (
        lab.get("shortlist") != SHORTLIST
        or lab.get("screen_stop_exclusive") != TRAIN_STOP
        or lab.get("screen_start") != 252
        or lab.get("source_origin_delay") != DELAY
        or lab.get("validation_rows_scored") != 0
        or not lab.get("prefix_replay_passed")
    ):
        raise Stop("Rank shortlist or training/release boundary changed.")
    rows = {r["name"]: r for r in lab["rows"]}
    if any(n not in rows or rows[n]["screen_exclusion"] is not None for n in SHORTLIST):
        raise Stop("Unscreened candidate found in the fixed shortlist.")


def read_diagnosis(root: Path, u) -> dict:
    path = u.safe_path(
        root, "logs/manual_feature_diagnosis/commodity_feature_diagnosis_report.json"
    )
    if u.digest(path) != DIAGNOSIS_REPORT:
        raise Stop("Notebook-09 report bytes changed; return the new report before fitting.")
    report = u.read_json(path)
    validate_diagnosis(report)
    stage = u.safe_path(root, "artifacts/feature_diagnosis/" + DIAGNOSIS + "/rank_candidates")
    u.verify_stage(stage, DIAGNOSIS, report["rank_checkpoint_hashes"])
    return report


def identity(root: Path, u) -> tuple[str, dict]:
    # JSON round-trip normalizes tuple contrasts into the on-disk list representation.
    declaration = json.loads(json.dumps(plan()))
    if u.read_json(u.safe_path(root, "configs/manual_rank_state_ablation.json")) != declaration:
        raise Stop("Declared rank experiment was edited; no retrospective threshold changes.")
    evidence = {
        "plan": declaration,
        "helper_sha256": u.digest(Path(__file__)),
        "diagnosis_helper_sha256": DIAGNOSIS_HELPER,
        "diagnosis_report_sha256": DIAGNOSIS_REPORT,
        "domain_config_sha256": u.digest(root / "configs/domain_study.json"),
    }
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(), evidence


def released_features(y, pairs, diagnosis):
    """Extend the verified training-only recipe, without widening its old API.

    This new API permits only the 1,349-row first-fold prefix. The final five
    outcome rows are removed from the feature input before any ranking. Scoring
    receives its own outcome object; no input mutation or future-label filling.
    """
    import numpy as np
    import pandas as pd

    if (
        not isinstance(y, pd.DataFrame)
        or not isinstance(pairs, pd.DataFrame)
        or y.empty
        or len(y) > STOP
        or not y.columns.is_unique
        or not pairs.columns.is_unique
        or not {"target", "lag"}.issubset(pairs.columns)
        or pairs.empty
        or pairs[["target", "lag"]].isna().any().any()
        or pairs.target.duplicated().any()
        or not pairs.lag.isin([1, 2, 3, 4]).all()
        or pairs.target.tolist() != y.columns.tolist()
        or y.index.tolist() != list(range(len(y)))
        or not all(pd.api.types.is_numeric_dtype(y[c]) for c in y)
    ):
        raise Stop("Rank inputs must be a bounded, aligned first-fold prefix with horizons 1-4.")
    clean = y.where(np.isfinite(y) & (y != -999999)).copy(deep=True)
    # These rows cannot be released for any feature origin in this requested prefix.
    clean.iloc[max(0, len(clean) - DELAY) :, :] = np.nan
    known = clean.shift(DELAY)
    global_rank = pd.DataFrame(
        diagnosis.ordinal_row_state(known.to_numpy()), index=y.index, columns=y.columns
    )
    horizon_rank = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for horizon in sorted(pairs.lag.unique()):
        cols = pairs.loc[pairs.lag == horizon, "target"].tolist()
        horizon_rank.loc[:, cols] = diagnosis.ordinal_row_state(known[cols].to_numpy())
    output = {}
    for group, q in [("global", global_rank), ("horizon", horizon_rank)]:
        fast = q.ewm(span=21, min_periods=14, adjust=False, ignore_na=False).mean()
        slow = q.ewm(span=63, min_periods=42, adjust=False, ignore_na=False).mean()
        for name, value in [
            ("latest", q),
            ("mean_21", fast),
            ("mean_63", slow),
            ("trend_21_63", fast - slow),
            ("innovation_63", q - slow),
        ]:
            output[f"released_rank_state__{group}_{name}"] = value.to_numpy()
    output[NUMERIC[10]] = global_rank.to_numpy() - horizon_rank.to_numpy()
    output[NUMERIC[11]] = output[NUMERIC[2]] - output[NUMERIC[7]]
    fraction = known.notna().mean(axis=1).to_numpy(copy=True)
    fraction[:DELAY] = np.nan
    output[COVERAGE[0]] = np.broadcast_to(fraction[:, None], y.shape).copy()
    output[COVERAGE[1]] = (
        global_rank.notna().astype(float).rolling(63, min_periods=42).mean().to_numpy()
    )
    names = NUMERIC + COVERAGE
    block = np.stack([output[n] for n in names], axis=2).astype(np.float32)
    if block.shape != (len(y), len(pairs), 14) or np.isinf(block).any():
        raise Stop("Invalid extended rank-state representation.")
    return block, names


def contract_smoke(diagnosis) -> dict:
    """Small real-library Copy-on-Write + timing tests before opening private labels."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260912)
    y = pd.DataFrame(rng.normal(size=(150, 16)), columns=[f"target_{i}" for i in range(16)])
    y.iloc[50:55, 0] = np.nan
    pairs = pd.DataFrame({"target": y.columns, "lag": np.arange(16) % 4 + 1})
    original = y.copy(deep=True)
    a, names = released_features(y, pairs, diagnosis)
    old, old_names = diagnosis.rank_features(y, pairs)
    if names != old_names:
        raise Stop("Rank formula names changed.")
    np.testing.assert_array_equal(a, old)
    for stop in [70, 115]:
        small, _ = released_features(y.iloc[:stop], pairs, diagnosis)
        np.testing.assert_array_equal(small, a[:stop])
    changed = y.copy(deep=True)
    changed.iloc[80:, :] = -4 * changed.iloc[80:, :]
    future, _ = released_features(changed, pairs, diagnosis)
    np.testing.assert_array_equal(a[:85], future[:85])
    pd.testing.assert_frame_equal(y, original, check_exact=True)
    return {
        "status": "RANK_TIMING_SMOKE_PASSED",
        "prefix_checks": [70, 115],
        "future_perturbation_exact": True,
        "same_training_recipe_exact": True,
        "input_unchanged": True,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }


def verify_candidate_prefix(root, block, names, pairs, report, diagnosis, u, y):
    import numpy as np

    stage = root / "artifacts/feature_diagnosis" / DIAGNOSIS / "rank_candidates"
    u.verify_stage(stage, DIAGNOSIS, report["rank_checkpoint_hashes"])
    inventory = u.read_json(stage / "inventory.json")
    if inventory["names"] != names or inventory["targets"] != pairs.target.tolist():
        raise Stop("Saved candidate axes differ.")
    with np.load(stage / "candidates.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["dates"], np.arange(TRAIN_STOP))
        np.testing.assert_array_equal(archive["values"], block[:TRAIN_STOP])
    # Cross-check against the corrected old builder, without executing notebook 09.
    replay, replay_names = diagnosis.rank_features(y.iloc[:TRAIN_STOP], pairs)
    if replay_names != names:
        raise Stop("Training recipe names differ.")
    np.testing.assert_array_equal(replay, block[:TRAIN_STOP])
    for stop in [START, 1250, STOP]:
        prefix, pn = released_features(y.iloc[:stop], pairs, diagnosis)
        if pn != names:
            raise Stop("Feature prefix metadata differ.")
        np.testing.assert_array_equal(prefix, block[:stop])
    return {
        "saved_training_prefix_exact": True,
        "old_builder_training_prefix_exact": True,
        "origin_prefixes_exact": [START, 1250, STOP],
        "max_source_origin_at_last_prediction": STOP - 1 - DELAY,
        "validation_origin_labels_used_only_after_release": True,
    }


def representation_audit(block, names, diagnosis) -> dict:
    """Only fitting-prefix values inform these diagnostics; no outcome fitting."""
    import numpy as np

    selected = block[252:TRAIN_STOP, :, [names.index(n) for n in SHORTLIST]]
    flat = selected.reshape(-1, len(SHORTLIST)).astype(float)
    correlations = np.eye(len(SHORTLIST))
    stats = []
    for j, name in enumerate(SHORTLIST):
        feature = selected[:, :, j]
        count = np.isfinite(feature).sum(axis=0)
        means = np.divide(
            np.where(np.isfinite(feature), feature, 0).sum(axis=0),
            count,
            out=np.full(feature.shape[1], np.nan),
            where=count > 0,
        )
        observed = feature[np.isfinite(feature)]
        centered = (feature - means[None, :])[np.isfinite(feature)]
        total_var = float(np.var(observed)) if len(observed) else 0.0
        within_fraction = float(np.mean(centered**2) / total_var) if total_var > 1e-15 else None
        stats.append(
            {
                "feature": name,
                "training_finite_fraction": float(np.isfinite(feature).mean()),
                "within_target_variation_fraction": within_fraction,
                "training_sha256": diagnosis.array_hash(feature),
            }
        )
        for k in range(j):
            mask = np.isfinite(flat[:, j]) & np.isfinite(flat[:, k])
            a, b = flat[mask, j], flat[mask, k]
            value = (
                float(np.corrcoef(a, b)[0, 1])
                if len(a) > 2 and min(a.std(), b.std()) > 1e-12
                else np.nan
            )
            correlations[j, k] = correlations[k, j] = value
    return {
        "rows": stats,
        "correlation_names": SHORTLIST,
        "correlation_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in correlations
        ],
        "scope": "Training rows 252:1164 only. Diagnostics do not reselect features. Variance fraction is descriptive, not proof of incremental predictive information.",
    }


def append_panel(base, block, names, chosen):
    import numpy as np

    from commodity_prediction.domain.catalog import Panel

    if (
        block.shape[:2] != base.values.shape[:2]
        or block.shape[2] != len(names)
        or len(set(names)) != len(names)
        or len(set(chosen)) != len(chosen)
        or not set(chosen).issubset(names)
        or set(chosen).intersection(base.names)
        or set(chosen).intersection(COVERAGE)
    ):
        raise Stop("Candidate axes, duplicates or forbidden coverage input.")
    panel = Panel(
        np.concatenate([base.values, block[:, :, [names.index(n) for n in chosen]]], axis=2),
        base.dates,
        base.targets,
        base.names + list(chosen),
        dict(base.source_series),
    )
    panel.validate()
    return panel


def fit_one(root, u, name, stage, lineage, base, block, names, y, pairs, fold, config, report):
    import joblib
    import pandas as pd

    from commodity_prediction.domain.attribution.run import fit_stage
    from commodity_prediction.domain.experiment import Experiment
    from commodity_prediction.domain.model import prepare, select

    if name not in VARIANTS or stage.exists():
        raise Stop("Undeclared variant or existing stage; no automatic refit.")
    panel = append_panel(base, block, names, VARIANTS[name])
    settings = {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
    stats = prepare(panel, y, fold.train_stop, settings)
    experiment = Experiment(name, algorithm="histogram")
    _, audit = select(stats, experiment.candidates(panel.names), settings, False)
    admitted = sorted(set(audit["selected_names"]).intersection(SHORTLIST))
    if set(admitted) != set(VARIANTS[name]) or set(audit["selected_names"]).intersection(COVERAGE):
        raise Stop("The declared numerical features were not all admitted; inspect before fitting.")
    if set(audit["rejection_reasons"]).intersection(
        {"feature_budget", "correlated", "unstable_sign"}
    ):
        raise Stop("Undeclared screening exclusion.")
    if report["fit_attempts_this_call"] >= len(VARIANTS):
        raise Stop("Three-fit limit exhausted.")
    report["fit_attempts_this_call"] += 1
    u.atomic_json(report_path(root), report)
    result = fit_stage(panel, y, pairs, fold, experiment, settings, stats, stage, lineage)
    saved = pd.read_parquet(stage / "predictions.parquet")
    replay = joblib.load(stage / "model.joblib").predict(panel, START, STOP)
    error = u.exact_predictions(saved, replay)
    result = {**result, "rank_numeric_admitted": len(admitted), "rank_numeric_names": admitted}
    report["checkpoint_hashes"][name] = u.verify_stage(stage, lineage)
    report["results"].append(result)
    report["new_training_fits"] += 1
    report["maximum_prediction_replay_error"] = max(
        report["maximum_prediction_replay_error"], error
    )
    u.atomic_json(report_path(root), report)
    emit(
        "RANK_MODEL_VERIFIED",
        variant=name,
        completed=report["new_training_fits"],
        total=3,
        official_metric=result["metrics"]["official_metric"],
    )
    del panel, stats, replay
    gc.collect()


def decision_rows(results, control, metric):
    if len(results) != len(VI := VARIANTS) or {r["variant"] for r in results} != set(VI):
        raise Stop("Incomplete or duplicated rank-model result inventory.")
    baseline = metric(control["daily_rank_correlations"], 180)
    if not math.isclose(baseline, control["official_metric"], abs_tol=1e-12, rel_tol=0):
        raise Stop("Saved baseline metric differs from daily correlations.")
    rows, candidates = [], []
    for name, chosen in VI.items():
        result = next(r for r in results if r["variant"] == name)
        score = metric(result["metrics"]["daily_rank_correlations"], 180)
        if not math.isclose(score, result["metrics"]["official_metric"], abs_tol=1e-12, rel_tol=0):
            raise Stop("Fitted score differs from daily correlations.")
        admitted = set(result["selection"]["selected_names"]).intersection(SHORTLIST)
        if admitted != set(chosen) or result["rank_numeric_admitted"] != len(chosen):
            raise Stop("Numerical admission evidence differs.")
        delta = score - baseline
        gate = delta >= GAIN
        rows.append(
            {
                "variant": name,
                "official_metric": score,
                "delta_vs_current_market": delta,
                "rank_candidates": len(chosen),
                "rank_admitted": len(admitted),
                "coverage_inputs": 0,
                "passes_review_gate": gate,
            }
        )
        if gate:
            candidates.append(name)
    return rows, candidates


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
    diagnosis, session, older, u = previous(root)
    began = time.monotonic()
    lineage, evidence = identity(root, u)
    directory = u.safe_path(root, "artifacts/rank_state_ablation/" + lineage)
    if directory.exists():
        raise Stop("Existing rank study directory preserved; no blind worker rerun.")
    directory.mkdir(parents=True)
    report = {
        "project": "commodity-prediction",
        "status": "RUNNING",
        "source_commit": SHA,
        "lineage": lineage,
        "identity": evidence,
        "plan": plan(),
        "started_utc": utc(),
        "new_training_fits": 0,
        "fit_attempts_this_call": 0,
        "control_refits": 0,
        "old_network_fits": 0,
        "graph_reconstructions": 0,
        "maximum_prediction_replay_error": 0.0,
        "results": [],
        "checkpoint_hashes": {},
        "feature_gate": "open",
        "promotion_allowed": False,
        "final_test_evaluations": 0,
        "validation_dates": 180,
        "aws_api_calls": 0,
        "github_writes": False,
    }

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - began, 3)
        u.atomic_json(report_path(root), report)

    def expire(*_):
        raise Stop("240-second worker deadline; preserve successful stages, no automatic retry.")

    previous_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        save()
        ready = u.readiness(root)
        u.validate_runtime(root, ready)
        u.validate_source(root)
        prior = read_diagnosis(root, u)
        parents = u.checkpoint_snapshot(root, ready)
        report["rank_lab"] = prior["rank_lab"]
        report["prior_network_diagnosis"] = prior["comparison_rows"]
        report["prior_control_pooled"] = prior["prior_control_pooled"]
        emit("TEST_CAUSAL_RANK_CONTRACT")
        report["contract_smoke"] = contract_smoke(diagnosis)
        save()
        for name, expected in u.RAW.items():
            if u.digest(u.safe_path(root, "data/raw/" + name)) != expected:
                raise Stop("Raw file changed: " + name)
        final = u.read_json(root / "configs/final_evaluation.json")
        if final["evaluated"] or final["final_test_start_date_id"] != 1714:
            raise Stop("Final-evaluation boundary changed.")
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
            or pairs.target.tolist() != list(y.columns)
            or list(y.columns) != [f"target_{i}" for i in range(424)]
        ):
            raise Stop("First-fold data schema changed.")
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
            cp = root / "artifacts" / PARENT / "fold_0/current_market"
            saved = pd.read_parquet(cp / "predictions.parquet")
            replay = joblib.load(cp / "model.joblib").predict(base, START, STOP)
            u.exact_predictions(saved, replay)
            control = evaluate(y.loc[saved.index], saved, pairs)
            if not session.close(
                control["official_metric"], CONTROL
            ) or saved.index.tolist() != list(range(START, STOP)):
                raise Stop("Saved first-fold control failed exact reproduction.")
            report["control"] = control
            emit("SAVED_CONTROL_REPLAY_PASSED", score=CONTROL, refits=0)
            emit("EXTEND_RELEASED_RANK_FEATURES", rows=STOP, source_origin_delay=DELAY)
            block, names = released_features(y, pairs, diagnosis)
            report["causal_checks"] = verify_candidate_prefix(
                root, block, names, pairs, prior, diagnosis, u, y
            )
            report["representation_audit"] = representation_audit(block, names, diagnosis)
            stage = directory / "features"
            stage.mkdir()
            np.savez_compressed(stage / "candidates.npz", values=block, dates=np.arange(STOP))
            u.atomic_json(
                stage / "inventory.json",
                {"names": names, "targets": pairs.target.tolist(), "plan": plan()},
            )
            seal_checkpoint(stage, lineage, ["candidates.npz", "inventory.json"])
            report["feature_checkpoint_hashes"] = u.verify_stage(stage, lineage)
            save()
            fold = Fold(0, TRAIN_STOP, START, STOP)
            for name in VARIANTS:
                emit("FIT_RANK_PANEL", variant=name, completed=report["new_training_fits"], total=3)
                fit_one(
                    root,
                    u,
                    name,
                    directory / "fold_0" / name,
                    lineage,
                    base,
                    block,
                    names,
                    y,
                    pairs,
                    fold,
                    config,
                    report,
                )
            rows, candidates = decision_rows(report["results"], control, session.metric)
            daily = {"current_market": np.asarray(control["daily_rank_correlations"])}
            daily.update(
                {
                    r["variant"]: np.asarray(r["metrics"]["daily_rank_correlations"])
                    for r in report["results"]
                }
            )
            report["comparisons"] = compare_predictions(daily, [180], plan()["contrasts"], config)
            report["comparison_rows"] = rows
            report["candidates_for_review"] = candidates
            report["decision"] = (
                "REVIEW_RANK_STATES_ON_OTHER_FOLDS"
                if candidates
                else "STOP_TESTED_RANK_STATE_PANELS"
            )
        report["parents_unchanged"] = u.checkpoint_snapshot(root, ready) == parents
        if not report["parents_unchanged"]:
            raise Stop("Parent checkpoint snapshot changed.")
        read_diagnosis(root, u)
        u.validate_source(root)
        report.update(
            source_unchanged=True,
            rank_parent_unchanged=True,
            status="RANK_STATE_REVIEW_READY",
            finished_utc=utc(),
            limitations=[
                "This is a repeatedly inspected first development fold, not fresh confirmatory or leaderboard evidence.",
                "Five family contrasts use conditional block intervals; no correction for the full adaptive research history.",
                "Latest and innovation features share information; the union is a measured interaction test, not six independent signals.",
                "Previously released validation outcomes update future feature states under the five-row rule; model parameters and the shortlist remain frozen.",
                "The current experiment is a feature addition, not an ensemble, ranking-loss change or replication of a competitor system.",
                "Missing observations change rank cohorts; coverage is diagnosed but coverage columns are not fitted here.",
                "A mean state can partly encode stable target identity. Training variance diagnostics are descriptive, not proof of conditional benefit.",
                "Private checkpoints are on the same disk until the user downloads them; no cloud/Git writes or auto shutdown.",
            ],
        )
        save()
        u.atomic_json(directory / "review.json", report)
        return report
    except BaseException as exc:
        report.update(
            status="STOPPED", error=type(exc).__name__ + ": " + str(exc), finished_utc=utc()
        )
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def verify_complete(root, r, session, u):
    lineage, evidence = identity(root, u)
    read_diagnosis(root, u)
    if (
        r.get("status") not in TERMINAL
        or r.get("lineage") != lineage
        or r.get("identity") != evidence
        or r.get("new_training_fits") != 3
        or r.get("fit_attempts_this_call") != 3
        or r.get("control_refits") != 0
        or r.get("old_network_fits") != 0
        or r.get("graph_reconstructions") != 0
        or r.get("maximum_prediction_replay_error") != 0.0
        or r.get("final_test_evaluations") != 0
        or r.get("parents_unchanged") is not True
        or r.get("rank_parent_unchanged") is not True
        or r.get("source_unchanged") is not True
        or r.get("causal_checks", {}).get("saved_training_prefix_exact") is not True
    ):
        raise Stop("Not a completed, verified rank-state experiment.")
    rows, candidates = decision_rows(r["results"], r["control"], session.metric)
    expected_decision = (
        "REVIEW_RANK_STATES_ON_OTHER_FOLDS" if candidates else "STOP_TESTED_RANK_STATE_PANELS"
    )
    if (
        rows != r["comparison_rows"]
        or candidates != r["candidates_for_review"]
        or r["decision"] != expected_decision
        or set(r["checkpoint_hashes"]) != set(VARIANTS)
    ):
        raise Stop("Stored decision or checkpoint inventory changed.")
    directory = root / "artifacts/rank_state_ablation" / lineage
    u.verify_stage(directory / "features", lineage, r["feature_checkpoint_hashes"])
    for name in VARIANTS:
        stage = directory / "fold_0" / name
        u.verify_stage(stage, lineage, r["checkpoint_hashes"][name])
        sealed = u.read_json(stage / "result.json")
        recorded = next(v for v in r["results"] if v["variant"] == name)
        if sealed != {
            k: v
            for k, v in recorded.items()
            if k not in ["rank_numeric_admitted", "rank_numeric_names"]
        }:
            raise Stop("Saved model result differs from the report.")


def run(root=ROOT):
    root = Path(root)
    diagnosis, session, older, u = previous(root)
    ready = u.readiness(root)
    u.validate_runtime(root, ready)
    u.validate_source(root)
    read_diagnosis(root, u)
    out = u.safe_path(root, "logs/manual_rank_state")
    out.mkdir(parents=True, exist_ok=True)
    with u.safe_path(out, "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report_path(root).exists():
            r = u.read_json(report_path(root))
            if r.get("status") not in TERMINAL:
                raise Stop(
                    "Interrupted rank study exists. Return its report; no automatic retry or budget reset."
                )
            verify_complete(root, r, session, u)
            emit("COMPLETED_RANK_STUDY_REUSED", new_fits=0)
            return {**r, "new_fits_this_notebook_call": 0}
        lineage, _ = identity(root, u)
        if (root / "artifacts/rank_state_ablation" / lineage).exists():
            raise Stop("Orphan rank artifact directory found; preserved without refitting.")
        if shutil.disk_usage(root).free < 1024**3:
            raise Stop("Need 1 GiB free. Do not delete old project folders blindly.")
        u.atomic_json(
            report_path(root),
            {
                "status": "STARTING_WORKER",
                "lineage": lineage,
                "started_utc": utc(),
                "new_training_fits": 0,
            },
        )
        args = [
            ready["runtime"]["python"],
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--root",
            str(root),
        ]
        started = time.monotonic()
        try:
            session._supervise_command(
                args, root, LIMIT, out / "rank_state_worker.log", u.environment(root)
            )
        except BaseException as exc:
            r = u.read_json(report_path(root))
            if r.get("status") == "STARTING_WORKER":
                r["new_training_fits"] = None
            r.update(
                status="STOPPED",
                supervisor_error=type(exc).__name__ + ": " + str(exc),
                supervised_seconds=round(time.monotonic() - started, 3),
            )
            u.atomic_json(report_path(root), r)
            raise
        r = u.read_json(report_path(root))
        verify_complete(root, r, session, u)
        r.update(
            supervised_seconds=round(time.monotonic() - started, 3), new_fits_this_notebook_call=3
        )
        u.atomic_json(report_path(root), r)
        return r


def charts(r):
    import numpy as np
    import plotly.graph_objects as go

    figs = []

    def layout(fig, title, x=None, y=None):
        fig.update_layout(
            title=title,
            height=465,
            margin=dict(l=70, r=35, t=85, b=100),
            xaxis_title=x,
            yaxis_title=y,
            legend=dict(orientation="h", y=-0.3),
        )
        figs.append(fig)

    old = r["prior_network_diagnosis"]
    f = go.Figure(
        go.Bar(x=[v["variant"] for v in old], y=[v["delta_vs_current_market"] for v in old])
    )
    layout(
        f,
        "1 · Completed network diagnosis — do not refit",
        "Prior representation",
        "First-fold metric change",
    )
    shortlist_rows = {v["name"]: v for v in r["rank_lab"]["rows"]}
    f = go.Figure()
    for key, label in [
        ("half_1_mean_ic", "Training half 1"),
        ("half_2_mean_ic", "Training half 2"),
    ]:
        f.add_bar(
            x=[n.removeprefix("released_rank_state__") for n in SHORTLIST],
            y=[shortlist_rows[n][key] for n in SHORTLIST],
            name=label,
        )
    f.update_layout(barmode="group")
    layout(
        f, "2 · Why these six? Training-only associations", "Fixed candidate", "Mean daily Spearman"
    )
    stats = r["representation_audit"]["rows"]
    f = go.Figure(
        go.Bar(
            x=[v["feature"].removeprefix("released_rank_state__") for v in stats],
            y=[v["within_target_variation_fraction"] for v in stats],
        )
    )
    layout(
        f,
        "3 · Time variation versus stable target differences",
        "Candidate",
        "Within-target fraction of variance",
    )
    names = [n.removeprefix("released_rank_state__") for n in SHORTLIST]
    f = go.Figure(
        go.Heatmap(
            z=r["representation_audit"]["correlation_matrix"], x=names, y=names, zmin=-1, zmax=1
        )
    )
    layout(
        f,
        "4 · Candidate overlap in the fitting interval",
        "Training candidate",
        "Training candidate",
    )
    origins = np.arange(START, STOP)
    f = go.Figure()
    f.add_scatter(
        x=origins.tolist(), y=(origins - DELAY).tolist(), name="Newest permitted outcome origin"
    )
    f.add_scatter(
        x=origins.tolist(), y=origins.tolist(), name="Prediction origin", line=dict(dash="dash")
    )
    layout(f, "5 · Common-origin label-release boundary", "Prediction origin", "Outcome origin")
    rows = r["comparison_rows"]
    f = go.Figure(
        go.Bar(
            x=["current_market"] + [v["variant"] for v in rows],
            y=[r["control"]["official_metric"]] + [v["official_metric"] for v in rows],
        )
    )
    layout(
        f, "6 · Same 180-date predictive comparison", "Representation", "Official first-fold metric"
    )
    f = go.Figure(
        go.Bar(x=[v["variant"] for v in rows], y=[v["delta_vs_current_market"] for v in rows])
    )
    f.add_hline(y=GAIN, line_dash="dash", annotation_text="Review threshold +0.002")
    layout(
        f,
        "7 · Incremental value over the saved baseline",
        "Rank feature panel",
        "Matched metric change",
    )
    bounds = [
        v for v in r["comparisons"] if v["block_dates"] == 20 and v["reference"] == "current_market"
    ]
    f = go.Figure()
    for v in bounds:
        lo, hi = v["conditional_95_interval"]
        f.add_scatter(x=[lo, hi], y=[v["variant"]] * 2, mode="lines", showlegend=False)
        f.add_scatter(x=[v["delta"]], y=[v["variant"]], mode="markers", name=v["variant"])
    f.add_vline(x=0, line_dash="dash")
    layout(
        f,
        "8 · Conditional 20-date block intervals — not global search correction",
        "Metric delta",
        "Panel",
    )
    base = np.asarray(r["control"]["daily_rank_correlations"])
    f = go.Figure()
    for v in r["results"]:
        difference = np.asarray(v["metrics"]["daily_rank_correlations"]) - base
        f.add_scatter(
            x=list(range(START, STOP)), y=np.cumsum(difference).tolist(), name=v["variant"]
        )
    layout(
        f,
        "9 · Concentration of rank-correlation change — NOT trading profit",
        "Origin",
        "Cumulative correlation difference",
    )
    f = go.Figure()
    f.add_bar(
        x=[v["variant"] for v in rows],
        y=[v["rank_candidates"] for v in rows],
        name="Declared numeric",
    )
    f.add_bar(
        x=[v["variant"] for v in rows],
        y=[v["rank_admitted"] for v in rows],
        name="Admitted numeric",
    )
    f.update_layout(barmode="group")
    layout(
        f, "10 · Were the new features actually used as model inputs?", "Panel", "Template count"
    )
    return figs


def private_backup(root, r, u):
    base = root / "artifacts/rank_state_ablation" / r["lineage"]
    paths = [base / "plan.json", base / "review.json"]
    paths += [base / "features" / n for n in r["feature_checkpoint_hashes"]]
    for name in VARIANTS:
        paths += [base / "fold_0" / name / n for n in r["checkpoint_hashes"][name]]
    pins = {
        str(p.relative_to(root)): u.digest(u.safe_path(root, str(p.relative_to(root))))
        for p in paths
    }
    if sum(p.stat().st_size for p in paths) > 256 * 1024**2:
        raise Stop("Private archive exceeds its 256-MiB packaging bound; models retained.")
    dest = u.safe_path(output_dir(root), "rank_state_checkpoints.zip")
    temp = u.safe_path(dest.parent, dest.name + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for p in paths:
            archive.write(p, str(p.relative_to(root)))
        archive.writestr("SHA256SUMS.json", json.dumps(pins, sort_keys=True, indent=2))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():
            raise Stop("Private archive CRC check failed.")
        for name, expected in pins.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise Stop("Private archive hash differs.")
    os.replace(temp, dest)
    return {
        "path": str(dest),
        "sha256": u.digest(dest),
        "bytes": dest.stat().st_size,
        "files": len(paths),
        "private": True,
        "contains_label_derived_features_models_predictions": True,
        "off_disk_backup_confirmed": False,
        "raw_csvs_included": False,
    }


def finish(root, report, figures):
    root = Path(root)
    diagnosis, session, older, u = previous(root)
    verify_complete(root, report, session, u)
    if len(figures) != 10:
        raise Stop("Expected ten Plotly figures.")
    result = dict(report)
    dashboard = output_dir(root) / "rank_state_dashboard.html"
    result.update(
        dashboard=str(dashboard),
        dashboard_sha256=older.export_dashboard(
            dashboard,
            "Released-rank feature ablations",
            [(str(f.layout.title.text), f) for f in figures],
        ),
        plotly_figures=10,
    )
    try:
        result["private_checkpoint_bundle"] = private_backup(root, result, u)
    except Exception as exc:
        result["backup_warning"] = type(exc).__name__ + ": " + str(exc)
    result.update(
        status="NOTEBOOK_AND_RANK_STATE_READY",
        notebook=str(root / "notebooks" / NOTEBOOK),
        updated_utc=utc(),
    )
    u.atomic_json(report_path(root), result)
    return result


def protocol_markdown():
    return """# Released-rank feature ablation — notebook 10

## Source-derived state
Notebook 09 completed after the read-only-array correction. The numerical-only
network scores were 0.377545 and 0.399654 versus 0.403381 on the same first fold;
neither met the review gate. Close those tested formulations unchanged. The
pooled current-market development score remains 0.3097087232124053.

The training-only rank laboratory shortlisted six of twelve numerical candidates.
The six names and their group membership are fixed in the adjacent declaration.
No coverage diagnostic is included as a fitted feature. No new shortlist is chosen
using the validation outcomes. Features rejected by this screen are not universally
proven useless; they do not enter this particular small follow-up.

## New, as-yet-unmeasured hypothesis
Recently released cross-sectional ordinal states may add useful information to
raw-return priors. Global latest rank, innovation versus the 63-row exponentially
weighted mean and a 21-row mean form one panel. Within-horizon latest rank,
within-horizon innovation and the global-minus-horizon latest contrast form a
second. A third is their exact union. Panel sizes are 3, 3 and 6.
This is a time-varying representation; it is not the earlier static rank-prior
experiment. It may still partly encode persistent target differences. The
training-only variance chart diagnoses that possibility without changing the fit.

## Availability and leakage contract
For prediction origin t, all targets' outcome cohort comes from origins <=t-5.
Waiting five rows before global or horizon ranking respects h+1 releases for h=1..4
and prevents a short-horizon observation being compared against an unreleased
long-horizon label from the same origin. Ranking uses observed labels, average
ties, minimum three observations, maps to [-1,1], and does not fill missing labels
with zero. EWM spans/minimum histories are 21/14 and 63/42, adjust=False,
ignore_na=False. The existing corrected training builder is never modified.

The new builder extends only to the 1349-row first-fold prefix. The last five
outcome rows are removed from its private feature-input copy before ranking.
The 1164-row constructed prefix must exactly match notebook 09's sealed array
and corrected builder, and full-prefix comparisons are checked at prediction
origins. A synthetic future-label perturbation test runs in the locked AWS runtime.

Later validation predictions may use earlier validation outcomes AFTER release;
this is sequential feature updating, not refitting. The model and shortlist stay
fixed. There is no claim that validation labels are never read: separate labels
also support honest validation scoring. Final origins 1714-1960 are never evaluated.

## Matched fitting and evidence
Use the unchanged current_market control, replayed exactly; zero control refits.
Use the existing histogram model settings, training-only preprocessing, and
existing admitted-feature policy (max_features equals panel width; correlation
cap 1.01). All declared numerical additions must be admitted before fitting.
At most three new fits, first fold only, and a 240-second supervised worker.
Each successful model/prediction/result stage is sealed and replayed independently.
An interrupted run is not retried automatically and never gets a silent time reset.
A completed run is verified/reused without fitting. No prior network models run.

Compare all three additions to current_market plus union-versus-each-group,
using the original 10/20/40-date paired block intervals. A +0.002 matched gain
is only a resource-allocation gate for review of later periods. The first fold
has repeatedly informed research; neither five-contrast intervals nor the gate
correct the full adaptive search history. No model promotion, final evaluation,
ensemble, hyperparameter changes or leaderboard-equivalence claims.

## Research context — distinct from measured project findings
A participant's 26th-place MITSUI writeup reports stable historical target-rank
features and clustering. Its displayed construction includes zero-filled labels;
we deliberately do not adopt that missingness convention. The current time-varying,
common-release-cohort representation is a separate hypothesis, not a reproduction
of that complete solution or a promise of its leaderboard performance.

Primary sources reviewed for this milestone:
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation

## Operations and publication
The helper installs one notebook, one script, this protocol and a fixed declaration.
It makes no AWS/Git calls, installs no packages and changes no old working copies.
Use Commodity - manual (verified). Keep old folders because the kernel environment
is reused from commodity-prediction-current. Save the executed notebook and JSON,
download the private ZIP off disk, and STOP (not DELETE) the space. Model/data ZIPs
are private. The notebook/report contain aggregate research evidence. Generated
source is local, not already published to GitHub. Further research remains open.
"""


def notebook_document():
    cells = []

    def md(text):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": f"rank10-{len(cells):02}",
                "source": text.splitlines(keepends=True),
            }
        )

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "id": f"rank10-{len(cells):02}",
                "execution_count": None,
                "outputs": [],
                "source": text.splitlines(keepends=True),
            }
        )

    md(
        "# Commodity prediction · Released-rank feature ablations\n\n"
        "**Milestone 10:** test the six training-selected ordinal features with the learner fixed.\n\n"
        "Up to **three new CPU fits**. No network reruns, coverage inputs, tuning, other folds or final test.\n"
        "The previous numerical-only network additions did not beat the baseline; they remain closed unchanged.\n"
        "Use **Commodity - manual (verified)**. This notebook is a research artifact, not a leaderboard claim."
    )
    code(
        "from pathlib import Path\nimport importlib.util\nimport pandas as pd\nfrom IPython.display import display\n"
        "ROOT=Path('/home/sagemaker-user/projects/commodity-prediction-manual')\n"
        "spec=importlib.util.spec_from_file_location('rank_state_support',ROOT/'scripts/commodity_rank_state_ablation.py')\n"
        "support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)\n"
        "diagnosis,session,older,u=support.previous(ROOT)\n"
        "ready=u.readiness(ROOT);u.validate_runtime(ROOT,ready);u.validate_source(ROOT)\n"
        "prior=support.read_diagnosis(ROOT,u)\n"
        "print('Pinned source:',support.SHA)\nprint('Previous diagnosis:',prior['decision'])\n"
        "print('New fit cap:',3,'| Worker seconds:',support.LIMIT)\n"
    )
    md(
        "## Fixed hypothesis and matched panels\n\n"
        "Global rank measures position across available targets. Within-horizon rank separates different "
        "prediction horizons. Innovations measure a new position relative to its prior slow state. "
        "The exact union tests whether the two groups complement each other. No selection uses the forthcoming validation scores."
    )
    code(
        "display(pd.DataFrame([{'panel':p,'numeric_count':len(ns),'features':', '.join(n.removeprefix('released_rank_state__') for n in ns),'coverage_inputs':0} for p,ns in support.VARIANTS.items()]))\n"
    )
    md(
        "## Availability is part of the feature definition\n\n"
        "All targets from an outcome origin wait five rows before they enter a rank cohort. "
        "Later validation predictions may use earlier validation outcomes only after this delay. "
        "The fitted model and shortlist remain frozen; the last five labels are excluded from the feature-input copy. "
        "Old training-prefix features must replay exactly. Missing labels are never replaced by zeros."
    )
    code(
        "print('Training prefix stops before:',support.TRAIN_STOP)\n"
        "print('Validation origins:',support.START,'through',support.STOP-1)\n"
        "print('Newest label origin for last prediction:',support.STOP-1-support.DELAY)\n"
        "display(pd.DataFrame(prior['rank_lab']['rows']).query('name in @support.SHORTLIST')[['name','half_1_mean_ic','half_2_mean_ic','training_coverage']])\n"
    )
    md(
        "## Execute once\n\n"
        "This cell launches the only model worker. It verifies saved parents, the locked environment, "
        "synthetic timing tests and exact control/candidate replay before any new fit. Each completed fit "
        "is sealed. Stop and retain the error/report if it fails; no unreviewed retry or budget reset."
    )
    code(
        "report=support.run(ROOT)\nprint('RESULT:',report['status'])\n"
        "print('Fits this notebook call:',report['new_fits_this_notebook_call'])\n"
        "print('Decision:',report['decision'])\nfigures=support.charts(report)\n"
    )
    titles = [
        (
            "What we are not rerunning",
            "Prior negative network results are preserved. Removing indicators improved the masked comparison but did not beat the original baseline.",
        ),
        (
            "Training-only shortlist",
            "Positive and negative stable associations may both be usable by a nonlinear learner. These are screening associations, not validation gains.",
        ),
        (
            "Time-varying versus persistent information",
            "A time-varying formula can still encode stable target identity. This fitting-prefix variance decomposition is descriptive and does not alter model inputs.",
        ),
        (
            "Redundancy is visible",
            "Correlations here use fitting-prefix feature values only. The 3/3/6 plan remains fixed regardless of this diagnostic.",
        ),
        (
            "Sequential release timing",
            "At prediction t, the most recent eligible outcome origin is t-5. This is not permission to fit on validation outcomes.",
        ),
        (
            "Primary same-fold metric",
            "Every model is scored on the same 180 origins. Do not compare this first-fold number directly with the 535-date pooled baseline.",
        ),
        (
            "Added value and the compute gate",
            "The +0.002 threshold only determines whether the family earns review on later periods. No automatic continuation.",
        ),
        (
            "Uncertainty",
            "Conditional block intervals condition on these models. They do not account for the complete adaptive search history or prove a leaderboard improvement.",
        ),
        (
            "Is the change concentrated?",
            "A cumulative rank-correlation difference is neither realized P&L nor a Sharpe ratio of trading returns.",
        ),
        (
            "Features actually admitted",
            "All intended numerical inputs must be admitted before a fit. No coverage-only columns are added to these models.",
        ),
    ]
    for i, (title, text) in enumerate(titles):
        md("## " + title + "\n\n" + text)
        code(f"figures[{i}].show()\n")
    md(
        "## Decision and durable evidence\n\n"
        "Save this notebook with Ctrl+S, download the JSON and private checkpoint ZIP, then stop the space. "
        "A passing first-fold gate requires review before other folds; a negative result closes only the tested panels. "
        "Feature engineering stays open. The dashboard is self-contained HTML; no Chrome/Kaleido is invoked."
    )
    code(
        "report=support.finish(ROOT,report,figures)\n"
        "display(pd.DataFrame(report['comparison_rows']))\n"
        "print('RESULT:',report['status'])\nprint('DECISION:',report['decision'])\n"
        "print('REPORT:',support.report_path(ROOT))\nprint('DASHBOARD:',report['dashboard'])\n"
        "print('PRIVATE BACKUP:',report.get('private_checkpoint_bundle',{}).get('path',report.get('backup_warning')))\n"
        "print('Save this notebook. Download the private backup. Stop space; do NOT Delete space.')\n"
    )
    md(
        "## Sources and interpretation\n\n"
        "The 26th-place participant report motivates examining historical ranks, but uses a different system and missingness policy. "
        "Our delayed time-varying representation is a distinct hypothesis.\n\n"
        "- [Participant writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction)\n"
        "- [pandas array-copy contract](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html)\n"
        "- [Official competition evaluation](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation)\n"
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Commodity - manual (verified)",
                "language": "python",
                "name": "commodity-manual",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "commodity_milestone": "released-rank-first-fold",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def install(root=ROOT):
    root = Path(root)
    diagnosis, session, older, u = previous(root)
    u.validate_source(root)
    read_diagnosis(root, u)
    nb = notebook_document()
    paths = {
        u.safe_path(root, "scripts/commodity_rank_state_ablation.py"): Path(__file__).read_bytes(),
        u.safe_path(root, "configs/manual_rank_state_ablation.json"): (
            json.dumps(plan(), sort_keys=True, indent=2) + "\n"
        ).encode(),
        u.safe_path(root, "docs/manual_rank_state_ablation.md"): protocol_markdown().encode(),
        u.safe_path(root, "notebooks/" + NOTEBOOK): (json.dumps(nb, indent=1) + "\n").encode(),
    }
    for path, body in paths.items():
        if path.exists():
            if path.suffix == ".ipynb":
                existing = u.read_json(path)

                def source(cell):
                    value = cell.get("source", "")
                    return "".join(value) if isinstance(value, list) else value

                if [(c["cell_type"], source(c)) for c in existing["cells"]] != [
                    (c["cell_type"], source(c)) for c in nb["cells"]
                ]:
                    raise Stop("Existing notebook edits preserved: " + str(path))
            elif path.read_bytes() != body:
                raise Stop("Existing different file preserved: " + str(path))
    for path, body in paths.items():
        if not path.exists():
            u.write_new(path, body)
    print(
        "RESULT: RANK_STATE_NOTEBOOK_READY\nNOTEBOOK: "
        + str(root / "notebooks" / NOTEBOOK)
        + "\nKERNEL: Commodity - manual (verified)",
        flush=True,
    )
    return {"new_training_fits": 0, "files": [str(p) for p in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.root)
        else:
            install(args.root)
    except BaseException as exc:
        print("RESULT: STOPPED\nERROR: " + type(exc).__name__ + ": " + str(exc), flush=True)
        print(
            "Keep the report/error. Do not repeat unchanged. Stop space manually; do not delete it.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
