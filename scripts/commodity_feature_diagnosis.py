#!/usr/bin/env python3
"""Install notebook 09: two mask-removal fits and a causal rank-state feature lab.

Default action installs only. Notebook run uses a single 240-second worker, then
exports the display/backup. No AWS/Git API calls, installs, old-study reruns, graph
reconstruction, ensemble, rank-feature model fits, or final-test evaluation.
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
import signal
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-manual")
SHA = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
PARENT = "04544d4b1a1d63487a24e88749d03d3259326c63902f2d0c74214e711147d605"
FEATURE = "52d3650abd20481db1a88fe7793085360bb0b4b684c501518ae9f4cb1c9405ba"
NETWORK = "d13660eed28efef3b938b898f502a2b88359c2d6a774f22c4f16d66f28314ccc"
NETWORK_HELPER = "3b0c9b77b5e44962bd9c0068c7f958a7418f826943f5c025aeff2261d4cdf8d4"
NETWORK_REPORT = "3131f1a96e6a7a9f3ca90ad0de0aa7b6b3f6ee685588af327f8ead9346fa0bb1"
TRAIN_STOP, START, STOP = 1164, 1169, 1349
CONTROL = 0.40338108742296147
LIMIT, GAIN, DELAY = 240.0, 0.002, 5
NOTEBOOK = "09_feature_diagnosis_and_rank_lab.ipynb"
VARIANTS = {"numeric_synchronous": "network_synchronous", "numeric_directed": "network_directed"}
RANK_GROUPS = ("global", "horizon")
RANK_FIELDS = ("latest", "mean_21", "mean_63", "trend_21_63", "innovation_63")
RANK_NUMERIC = [f"released_rank_state__{g}_{f}" for g in RANK_GROUPS for f in RANK_FIELDS] + [
    "released_rank_state__global_minus_horizon_latest",
    "released_rank_state__global_minus_horizon_mean_63",
]
RANK_MASKS = [
    "released_rank_state__cohort_observed_fraction",
    "released_rank_state__own_coverage_63",
]
TERMINAL = {"DIAGNOSIS_REVIEW_READY", "NOTEBOOK_AND_DIAGNOSIS_READY"}


class Stop(RuntimeError):
    """Preserve results; diagnose before any repeated operation."""


def _validate_recovery_evidence(root, u, recovery):
    """Accept one reviewed, zero-fit failure; retain its evidence and elapsed budget."""
    if not isinstance(recovery, dict) or recovery.get("kind") != "pandas_readonly_before_fit":
        raise Stop("Missing reviewed recovery evidence; no automatic retry.")
    old_sha = "8577c61b09c9afd9312666a59ddf9c0f687699198f503c333c2d4f55937262b1"
    source = u.safe_path(root, recovery["original_helper_backup"])
    failed = u.safe_path(root, recovery["failed_report_backup"])
    if u.digest(source) != old_sha or u.digest(failed) != recovery["failed_report_sha256"]:
        raise Stop("Archived failure evidence changed; preserve it and stop.")
    old = u.read_json(failed)
    if (
        old.get("status") != "STOPPED"
        or old.get("error") != "ValueError: assignment destination is read-only"
        or any(
            type(old.get(k)) is not int or old[k] != 0
            for k in [
                "new_training_fits",
                "fit_attempts_this_call",
                "control_refits",
                "graph_reconstructions",
                "rank_feature_models_fitted",
                "final_test_evaluations",
            ]
        )
        or old.get("results") != []
        or old.get("checkpoint_hashes") != {}
        or "rank_checkpoint_hashes" in old
        or "rank_lab" in old
        or old.get("plan") != plan()
        or old.get("source_commit") != SHA
    ):
        raise Stop("Failure is not the reviewed pre-fit read-only-array error.")
    identity_old = old.get("identity", {})
    expected_identity = {
        "plan": plan(),
        "helper_sha256": old_sha,
        "network_helper_sha256": NETWORK_HELPER,
        "network_report_sha256": NETWORK_REPORT,
        "domain_config_sha256": u.digest(root / "configs/domain_study.json"),
    }
    if identity_old != expected_identity:
        raise Stop("Original failure identity differs.")
    old_lineage = hashlib.sha256(json.dumps(identity_old, sort_keys=True).encode()).hexdigest()
    if old.get("lineage") != old_lineage or recovery.get("old_lineage") != old_lineage:
        raise Stop("Original failure lineage differs.")
    directory = u.safe_path(root, "artifacts/feature_diagnosis/" + old_lineage)
    if not directory.is_dir() or sorted(p.name for p in directory.iterdir()) != ["plan.json"]:
        raise Stop("Original attempt contains additional artifacts; do not retry or discard them.")
    if u.read_json(u.safe_path(directory, "plan.json")) != identity_old:
        raise Stop("Original saved plan differs.")
    spent = []
    for k in ["elapsed_seconds", "supervised_seconds"]:
        value = old.get(k)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise Stop("Missing or invalid original runtime accounting.")
        spent.append(float(value))
    used = max(spent)
    if used <= 0 or used >= LIMIT or recovery.get("previous_attempt_seconds") != used:
        raise Stop("Original budget exhausted or altered; no automatic extension.")
    if recovery.get("remaining_seconds") != LIMIT - used:
        raise Stop("Recovery would reset or extend the original budget.")
    if recovery.get("corrected_helper_sha256") != u.digest(Path(__file__)):
        raise Stop("Corrected source does not match the tested recovery.")
    return recovery


def utc():
    return datetime.now(UTC).isoformat()


def emit(stage, **fields):
    print(json.dumps({"utc": utc(), "stage": stage, **fields}, allow_nan=False), flush=True)


def previous(root):
    p = Path(root) / "scripts/commodity_close_network.py"
    if (
        p.is_symlink()
        or not p.is_file()
        or hashlib.sha256(p.read_bytes()).hexdigest() != NETWORK_HELPER
    ):
        raise Stop("Notebook-08 helper is missing or changed; do not overwrite it.")
    spec = importlib.util.spec_from_file_location("verified_network_support", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    session, older, u = mod.previous(Path(root))
    return mod, session, older, u


def output_dir(root):
    return Path(root) / "logs/manual_feature_diagnosis"


def report_path(root):
    return output_dir(root) / "commodity_feature_diagnosis_report.json"


def plan():
    return {
        "study": "network_mask_removal_and_causal_rank_state_lab",
        "source_commit": SHA,
        "parent_lineage": PARENT,
        "network_lineage": NETWORK,
        "max_new_fits": 2,
        "old_control_refits": 0,
        "worker_limit_seconds": LIMIT,
        "training_stop_exclusive": TRAIN_STOP,
        "warmup_dates": 252,
        "validation_start": START,
        "validation_stop_exclusive": STOP,
        "mask_removal_variants": VARIANTS,
        "numerical_network_templates_per_variant": 6,
        "new_availability_inputs_in_fits": 0,
        "fitted_rank_feature_models": 0,
        "rank_numerical_candidates": 12,
        "rank_coverage_candidates": 2,
        "rank_origin_delay": DELAY,
        "rank_reference": "Same origin across targets; h+1 release contract, max h=4.",
        "rank_ema_spans": [21, 63],
        "rank_minimum_history": [14, 42],
        "minimum_cohort_labels": 3,
        "rank_screen_stop_exclusive": TRAIN_STOP,
        "rank_screen_warmup": 252,
        "rank_shortlist_limit": 6,
        "rank_screen": "Training-only daily Spearman associations, two chronological halves; coverage>=0.70, >=40 eligible dates in each half, matching nonzero sign, exact duplicates removed. No validation outcomes used for shortlist.",
        "review_gain_vs_each_reference": GAIN,
        "decision_rule": "Review only if a numeric-only panel gains >=0.002 over BOTH saved current_market and its saved same-numeric-with-masks model; exact replay and all six numeric features admitted. Rank lab is unfit.",
        "rule_timing": "Declared after notebook-08 results. This is a diagnostic removal experiment on a repeatedly inspected development fold, not an independent confirmatory test.",
        "learner_policy": "Frozen histogram settings and original training preprocessing/screening. Existing models/predictions and network cube are reused, never regenerated.",
        "feature_gate": "open",
        "promotion_allowed": False,
        "further_folds": False,
        "final_test_evaluations": 0,
        "aws_api_calls": 0,
        "github_writes": False,
    }


def read_network(root, network, session, u):
    p = u.safe_path(root, "logs/manual_close_network/commodity_close_network_report.json")
    if u.digest(p) != NETWORK_REPORT:
        raise Stop(
            "The inspected notebook-08 report changed; return it instead of rerunning experiments."
        )
    r = u.read_json(p)
    if r.get("decision") != "STOP_THIS_NETWORK_FORMULATION" or r.get("lineage") != NETWORK:
        raise Stop("Unexpected notebook-08 decision or lineage.")
    network.verify_complete(root, r, session, u)
    return r


def identity(root, u):
    p = u.safe_path(root, "configs/manual_feature_diagnosis.json")
    if u.read_json(p) != plan():
        raise Stop("Experiment declaration changed; no retrospective threshold editing.")
    d = {
        "plan": plan(),
        "helper_sha256": u.digest(Path(__file__)),
        "network_helper_sha256": NETWORK_HELPER,
        "network_report_sha256": NETWORK_REPORT,
        "domain_config_sha256": u.digest(root / "configs/domain_study.json"),
    }
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def ordinal_row_state(values, minimum=3):
    """Ranks among observed values only, centered into [-1,1], with average ties.

    Missing outcomes are not replaced by zero. All ties give zero; sparse rows
    with fewer than minimum labels remain unavailable. This is a representation,
    not the official scoring function (which is unchanged).
    """
    import numpy as np
    import pandas as pd

    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or minimum < 2:
        raise Stop("Rank values must be a 2D matrix and minimum>=2.")
    a = np.where(np.isfinite(a) & (a != -999999), a, np.nan)
    f = pd.DataFrame(a)
    count = np.isfinite(a).sum(axis=1)
    ranks = f.rank(axis=1, method="average", na_option="keep").to_numpy()
    denom = np.maximum(count - 1, 1)[:, None]
    z = 2 * (ranks - 1) / denom - 1
    z[count < minimum] = np.nan
    return z


def validate_rank_inputs(y, pairs):
    import numpy as np
    import pandas as pd

    if not isinstance(y, pd.DataFrame) or not isinstance(pairs, pd.DataFrame):
        raise Stop("Rank lab requires DataFrames.")
    if (
        y.empty
        or not y.columns.is_unique
        or not pairs.columns.is_unique
        or not {"target", "lag"}.issubset(pairs.columns)
        or pairs.empty
    ):
        raise Stop("Ambiguous or incomplete rank input axes.")
    if (
        pairs[["target", "lag"]].isna().any().any()
        or pairs.target.duplicated().any()
        or not pairs.lag.isin([1, 2, 3, 4]).all()
        or pairs.target.tolist() != y.columns.tolist()
    ):
        raise Stop("Rank targets or horizon metadata are not aligned.")
    if y.index.tolist() != list(range(len(y))) or len(y) > TRAIN_STOP:
        raise Stop("Rank lab is restricted to the zero-based training prefix, at most 1164 rows.")
    if not all(pd.api.types.is_numeric_dtype(y[c]) for c in y):
        raise Stop("Outcome columns must be numeric.")
    return y.where(np.isfinite(y) & (y != -999999))


def rank_features(y, pairs):
    """14 causal candidates; every source outcome is from origin <=t-5.

    A complete origin cohort is delayed five rows before ANY ranking/smoothing.
    Thus rank denominators cannot access longer-horizon labels too early. EMA
    state may persist across a missing observation; no raw label is forward-filled.
    Strictly training-prefix API avoids accidental validation/final-test screening.
    """
    import numpy as np
    import pandas as pd

    clean = validate_rank_inputs(y, pairs)
    known = clean.shift(DELAY)
    global_rank = pd.DataFrame(
        ordinal_row_state(known.to_numpy()), index=y.index, columns=y.columns
    )
    horizon_rank = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for horizon in sorted(pairs.lag.unique()):
        cols = pairs.loc[pairs.lag == horizon, "target"].tolist()
        horizon_rank.loc[:, cols] = ordinal_row_state(known[cols].to_numpy())
    out = {}
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
            out[f"released_rank_state__{group}_{name}"] = value.to_numpy()
    out[RANK_NUMERIC[10]] = global_rank.to_numpy() - horizon_rank.to_numpy()
    out[RANK_NUMERIC[11]] = (
        out["released_rank_state__global_mean_63"] - out["released_rank_state__horizon_mean_63"]
    )
    cohort_fraction = known.notna().mean(axis=1).to_numpy(copy=True)
    cohort_fraction[:DELAY] = np.nan
    out[RANK_MASKS[0]] = np.broadcast_to(cohort_fraction[:, None], y.shape).copy()
    out[RANK_MASKS[1]] = (
        global_rank.notna().astype(float).rolling(63, min_periods=42).mean().to_numpy()
    )
    names = RANK_NUMERIC + RANK_MASKS
    values = np.stack([out[n] for n in names], axis=2).astype(np.float32)
    if values.shape != (len(y), len(pairs), 14) or np.isinf(values).any():
        raise Stop("Invalid causal rank-state array.")
    return values, names


def daily_association(feature, truth):
    """Training-only cross-sectional Spearman with pairwise missing-value masks."""
    import numpy as np
    import pandas as pd

    f, y = np.asarray(feature, float), np.asarray(truth, float)
    if f.shape != y.shape or f.ndim != 2:
        raise Stop("Association axes do not align.")
    valid = np.isfinite(f) & np.isfinite(y) & (y != -999999)
    a = pd.DataFrame(np.where(valid, f, np.nan)).rank(axis=1).to_numpy()
    b = pd.DataFrame(np.where(valid, y, np.nan)).rank(axis=1).to_numpy()
    count = valid.sum(axis=1)
    aa = np.where(valid, a, 0)
    bb = np.where(valid, b, 0)
    ma = aa.sum(axis=1) / np.maximum(count, 1)
    mb = bb.sum(axis=1) / np.maximum(count, 1)
    da = np.where(valid, a - ma[:, None], 0)
    db = np.where(valid, b - mb[:, None], 0)
    den = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    return np.divide(
        (da * db).sum(axis=1), den, out=np.full(len(f), np.nan), where=(den > 1e-12) & (count >= 3)
    )


def array_hash(a):
    import numpy as np

    v = np.asarray(a, dtype="<f4").copy()
    v[np.isnan(v)] = np.float32(np.nan)
    # Canonicalize signed zeros only; no numerical values or missingness are filled.
    v[v == 0] = 0
    return hashlib.sha256(v.tobytes()).hexdigest()


def rank_screen(block, names, y, base_values=None, base_names=None):
    import numpy as np

    if len(y) != TRAIN_STOP or block.shape != (TRAIN_STOP, y.shape[1], len(names)):
        raise Stop("Training screen expects exactly 1164 aligned rows.")
    lo, hi, middle = 252, TRAIN_STOP, (252 + TRAIN_STOP) // 2
    signatures = {}
    if base_values is not None:
        if base_values.shape[:2] != (TRAIN_STOP, y.shape[1]) or base_names is None:
            raise Stop("Historical feature overlap axes differ.")
        for i, n in enumerate(base_names):
            signatures.setdefault(array_hash(base_values[lo:hi, :, i]), n)
    rows = []
    truth = y.to_numpy(dtype=float)
    clean_y = np.where(np.isfinite(truth) & (truth != -999999), truth, np.nan)
    profile = []
    for j, name in enumerate(names):
        f = block[:, :, j]
        sig = array_hash(f[lo:hi])
        duplicate = signatures.get(sig)
        signatures.setdefault(sig, name)
        daily = daily_association(f[lo:hi], clean_y[lo:hi])
        a, b = daily[: middle - lo], daily[middle - lo :]
        av = a[np.isfinite(a)]
        bv = b[np.isfinite(b)]
        ca, cb = len(av), len(bv)
        ma, mb = (float(av.mean()) if ca else None), (float(bv.mean()) if cb else None)
        coverage = float(np.isfinite(f[lo:hi]).mean())
        reason = (
            "coverage_only_not_ranked"
            if name in RANK_MASKS
            else "exact_duplicate"
            if duplicate
            else "coverage"
            if coverage < 0.70
            else "insufficient_dates"
            if min(ca, cb) < 40
            else "unstable_or_zero_direction"
            if ma is None or mb is None or ma * mb <= 0
            else None
        )
        relevance = min(abs(ma), abs(mb)) if reason is None else 0.0
        finite = f[lo:hi][np.isfinite(f[lo:hi])]
        rows.append(
            {
                "name": name,
                "type": "coverage" if name in RANK_MASKS else "numeric",
                "half_1_mean_ic": ma,
                "half_2_mean_ic": mb,
                "half_1_dates": ca,
                "half_2_dates": cb,
                "training_coverage": coverage,
                "exact_duplicate_of": duplicate,
                "screen_exclusion": reason,
                "training_relevance": relevance,
                "q05": float(np.quantile(finite, 0.05)) if len(finite) else None,
                "q95": float(np.quantile(finite, 0.95)) if len(finite) else None,
                "training_sha256": sig,
            }
        )
        for start in range(lo, hi, 114):
            end = min(start + 114, hi)
            profile.append(
                {
                    "feature": name,
                    "start": start,
                    "stop": end,
                    "finite_fraction": float(np.isfinite(f[start:end]).mean()),
                }
            )
    eligible = sorted(
        [r for r in rows if r["screen_exclusion"] is None],
        key=lambda r: (-r["training_relevance"], r["name"]),
    )
    short = [r["name"] for r in eligible[:6]]
    return {
        "candidate_templates": len(names),
        "numeric_candidates": 12,
        "coverage_candidates": 2,
        "rows": rows,
        "shortlist": short,
        "shortlist_limit": 6,
        "coverage_by_training_block": profile,
        "screen_start": lo,
        "screen_stop_exclusive": hi,
        "split_date": middle,
        "validation_rows_scored": 0,
        "new_training_fits": 0,
        "overlap_scope": "Exact duplicate comparison with active current_market training arrays and within rank family; not exhaustive semantic deduplication or proof of conditional value.",
    }


def cached_network(root, old, u):
    import numpy as np

    stage = root / "artifacts/close_network_ablation" / NETWORK / "features"
    u.verify_stage(stage, NETWORK, old["feature_checkpoint_hashes"])
    inv = u.read_json(stage / "inventory.json")
    with np.load(stage / "candidates.npz", allow_pickle=False) as z:
        a, dates = z["values"], z["dates"]
    if (
        a.shape != (STOP, 424, 16)
        or dates.tolist() != list(range(STOP))
        or len(inv["names"]) != 16
        or len(set(inv["names"])) != 16
        or inv["targets"] != [f"target_{i}" for i in range(424)]
        or np.isinf(a).any()
    ):
        raise Stop("Saved network feature axes or values changed.")
    return a, inv["names"]


def numerical_columns(network, variant, names):
    if variant not in VARIANTS:
        raise Stop("Undeclared diagnostic fit.")
    kind = "synchronous" if variant == "numeric_synchronous" else "directed"
    chosen = list(network.NUMERIC[kind])
    if len(chosen) != 6 or set(chosen) & set(network.MASKS) or not set(chosen).issubset(names):
        raise Stop("Numeric-only panel does not have the exact six recorded numeric inputs.")
    return chosen


def decision_rows(results, old, metric):
    if {r["variant"] for r in results} != set(VARIANTS) or len(results) != 2:
        raise Stop("Expected exactly two diagnostic fits.")
    original = float(old["control"]["official_metric"])
    refs = {r["variant"]: float(r["official_metric"]) for r in old["comparison_rows"]}
    rows, advance = [], []
    for r in results:
        value = metric(r["metrics"]["daily_rank_correlations"], 180)
        if not math.isclose(value, r["metrics"]["official_metric"], abs_tol=1e-12, rel_tol=0):
            raise Stop("Daily-metric reconstruction differs.")
        name = r["variant"]
        masked = VARIANTS[name]
        numeric = r["numeric_admitted"]
        d0, dm = value - original, value - refs[masked]
        good = d0 >= GAIN and dm >= GAIN and numeric == 6
        rows.append(
            {
                "variant": name,
                "official_metric": value,
                "masked_reference": masked,
                "delta_vs_current_market": d0,
                "delta_vs_masked_same_numeric": dm,
                "numeric_admitted": numeric,
                "explicit_network_masks_added": 0,
                "passes_review_gate": good,
            }
        )
        if good:
            advance.append(name)
    return rows, advance


def fit_one(
    root, u, network, stage, lineage, name, base, block, names, y, pairs, fold, config, report
):
    import joblib
    import numpy as np
    import pandas as pd

    from commodity_prediction.domain.attribution.run import fit_stage
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.experiment import Experiment
    from commodity_prediction.domain.model import prepare, select

    chosen = numerical_columns(network, name, names)
    panel = Panel(
        np.concatenate([base.values, block[:, :, [names.index(n) for n in chosen]]], axis=2),
        base.dates,
        base.targets,
        base.names + chosen,
        dict(base.source_series),
    )
    panel.validate()
    if stage.exists():
        raise Stop("Unexpected pre-existing fit; preserve and inspect, never auto-refit.")
    settings = {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
    stats = prepare(panel, y, fold.train_stop, settings)
    _, audit = select(
        stats, Experiment(name, algorithm="histogram").candidates(panel.names), settings, False
    )
    admitted = set(audit["selected_names"]).intersection(chosen)
    if len(admitted) != 6 or set(audit["selected_names"]).intersection(network.MASKS):
        raise Stop("Diagnostic isolation failed before fitting.")
    if set(audit["rejection_reasons"]).intersection(
        {"feature_budget", "correlated", "unstable_sign"}
    ):
        raise Stop("Undeclared screening exclusions.")
    report["fit_attempts_this_call"] += 1
    u.atomic_json(report_path(root), report)
    result = fit_stage(
        panel,
        y,
        pairs,
        fold,
        Experiment(name, algorithm="histogram"),
        settings,
        stats,
        stage,
        lineage,
    )
    saved = pd.read_parquet(stage / "predictions.parquet")
    replay = joblib.load(stage / "model.joblib").predict(panel, START, STOP)
    error = u.exact_predictions(saved, replay)
    result["numeric_admitted"] = len(admitted)
    report["checkpoint_hashes"][name] = u.verify_stage(stage, lineage)
    report["results"].append(result)
    report["new_training_fits"] += 1
    report["maximum_prediction_replay_error"] = max(
        report["maximum_prediction_replay_error"], error
    )
    u.atomic_json(report_path(root), report)
    emit(
        "DIAGNOSTIC_MODEL_VERIFIED",
        variant=name,
        completed=report["new_training_fits"],
        total=2,
        official_metric=result["metrics"]["official_metric"],
    )
    del panel, stats, replay
    gc.collect()
    return saved


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
    network, session, older, u = previous(root)
    starting_receipt = u.read_json(report_path(root)) if report_path(root).exists() else {}
    recovery = starting_receipt.get("recovery")
    if recovery is not None:
        if starting_receipt.get("status") != "STARTING_WORKER":
            raise Stop("Recovery worker was not claimed by its supervisor.")
        _validate_recovery_evidence(root, u, recovery)
    carried = recovery["previous_attempt_seconds"] if recovery else 0.0
    remaining = LIMIT - carried
    started = time.monotonic()
    lineage, evidence = identity(root, u)
    directory = u.safe_path(root, "artifacts/feature_diagnosis/" + lineage)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "commodity-prediction",
        "status": "RUNNING",
        "source_commit": SHA,
        "lineage": lineage,
        "identity": evidence,
        "plan": plan(),
        "started_utc": utc(),
        "feature_gate": "open",
        "promotion_allowed": False,
        "final_test_evaluations": 0,
        "new_training_fits": 0,
        "control_refits": 0,
        "fit_attempts_this_call": 0,
        "maximum_prediction_replay_error": 0.0,
        "results": [],
        "checkpoint_hashes": {},
        "graph_reconstructions": 0,
        "rank_feature_models_fitted": 0,
        "aws_api_calls": 0,
        "github_writes": False,
        "validation_dates": 180,
        "recovery": recovery,
        "previous_attempt_seconds": carried,
        "worker_budget_seconds": remaining,
    }

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        u.atomic_json(report_path(root), report)

    def expire(*_):
        raise Stop("240-second milestone deadline reached; retain completed files.")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        save()
        ready = u.readiness(root)
        u.validate_runtime(root, ready)
        u.validate_source(root)
        old = read_network(root, network, session, u)
        parents = u.checkpoint_snapshot(root, ready)
        report["prior_network_comparisons"] = old["comparison_rows"]
        report["prior_control_pooled"] = old["prior_control_pooled"]
        report["prior_network_report_sha256"] = NETWORK_REPORT
        for name, expected in u.RAW.items():
            if u.digest(u.safe_path(root, "data/raw/" + name)) != expected:
                raise Stop("Raw file changed: " + name)
        final = u.read_json(root / "configs/final_evaluation.json")
        if final["evaluated"] or final["final_test_start_date_id"] != 1714:
            raise Stop("Final gate changed.")
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
                raise Stop("Saved control did not reproduce.")
            report["control"] = control
            emit("SAVED_CONTROL_REPLAY_PASSED", score=CONTROL)
            block, names = cached_network(root, old, u)
            # Replay the TWO masked counterparts using saved weights; zero refits.
            for _name, masked in VARIANTS.items():
                chosen = network.plan()["variants"][masked]
                p = Panel(
                    np.concatenate(
                        [base.values, block[:, :, [names.index(n) for n in chosen]]], axis=2
                    ),
                    base.dates,
                    base.targets,
                    base.names + chosen,
                    dict(base.source_series),
                )
                stage = root / "artifacts/close_network_ablation" / NETWORK / "fold_0" / masked
                truth_saved = pd.read_parquet(stage / "predictions.parquet")
                pred = joblib.load(stage / "model.joblib").predict(p, START, STOP)
                u.exact_predictions(truth_saved, pred)
                observed = evaluate(y.loc[pred.index], pred, pairs)
                prior_result = next(r for r in old["results"] if r["variant"] == masked)
                np.testing.assert_array_equal(
                    observed["daily_rank_correlations"],
                    prior_result["metrics"]["daily_rank_correlations"],
                )
                emit("MASKED_REFERENCE_REPLAY_PASSED", variant=masked)
                del p, pred
            # Independent candidate lab receives the TRAINING PREFIX ONLY.
            emit("ENGINEER_RELEASED_RANK_STATES", training_rows=TRAIN_STOP, new_candidates=14)
            training_y = y.iloc[:TRAIN_STOP].copy()
            ranks, rank_names = rank_features(training_y, pairs)
            for stop in [505, 917]:
                q, qn = rank_features(training_y.iloc[:stop], pairs)
                if qn != rank_names:
                    raise Stop("Rank prefix names differ.")
                np.testing.assert_array_equal(q, ranks[:stop])
            rank_result = rank_screen(
                ranks, rank_names, training_y, base.values[:TRAIN_STOP], base.names
            )
            rank_result.update(
                prefix_replay_passed=True,
                source_origin_delay=DELAY,
                max_source_origin_at_train_end=TRAIN_STOP - 1 - DELAY,
                status="TRAINING_RANK_LAB_READY",
            )
            rank_stage = directory / "rank_candidates"
            rank_stage.mkdir()
            np.savez_compressed(
                rank_stage / "candidates.npz", values=ranks, dates=np.arange(TRAIN_STOP)
            )
            u.atomic_json(
                rank_stage / "inventory.json",
                {"names": rank_names, "targets": pairs.target.tolist(), "plan": plan()},
            )
            seal_checkpoint(rank_stage, lineage, ["candidates.npz", "inventory.json"])
            report["rank_checkpoint_hashes"] = u.verify_stage(rank_stage, lineage)
            report["rank_lab"] = rank_result
            save()
            del ranks, training_y
            gc.collect()
            fold = Fold(0, TRAIN_STOP, START, STOP)
            for name in VARIANTS:
                emit(
                    "FIT_MASK_REMOVAL", variant=name, completed=report["new_training_fits"], total=2
                )
                fit_one(
                    root,
                    u,
                    network,
                    directory / "fold_0" / name,
                    lineage,
                    name,
                    base,
                    block,
                    names,
                    y,
                    pairs,
                    fold,
                    config,
                    report,
                )
            rows, candidates = decision_rows(report["results"], old, session.metric)
            daily = {"current_market": np.asarray(control["daily_rank_correlations"])}
            daily.update(
                {
                    r["variant"]: np.asarray(r["metrics"]["daily_rank_correlations"])
                    for r in report["results"]
                }
            )
            daily.update(
                {
                    r["variant"]: np.asarray(r["metrics"]["daily_rank_correlations"])
                    for r in old["results"]
                    if r["variant"] in VARIANTS.values()
                }
            )
            contrasts = [(n, "current_market") for n in VARIANTS] + list(VARIANTS.items())
            report["comparisons"] = compare_predictions(daily, [180], contrasts, config)
            report["comparison_rows"] = rows
            report["candidates_for_review"] = candidates
            report["decision"] = (
                "REVIEW_NUMERIC_ONLY_ON_OTHER_FOLDS"
                if candidates
                else "STOP_TESTED_NETWORK_VARIANTS"
            )
            report["rank_decision"] = (
                "REVIEW_RANK_CANDIDATES_BEFORE_FITS"
                if rank_result["shortlist"]
                else "NO_RANK_CANDIDATE_PASSED_TRAINING_SCREEN"
            )
        report["parents_unchanged"] = u.checkpoint_snapshot(root, ready) == parents
        if not report["parents_unchanged"]:
            raise Stop("Parent snapshot changed.")
        read_network(root, network, session, u)
        u.validate_source(root)
        report.update(
            source_unchanged=True,
            prior_network_unchanged=True,
            status="DIAGNOSIS_REVIEW_READY",
            finished_utc=utc(),
            limitations=[
                "Mask removal is diagnostic and chosen after seeing notebook-08 results, not an independent confirmatory study.",
                "Same numerical inputs and fixed learner settings, but refitting changes the whole fitted model. A score difference is not a universal causal property of masks.",
                "Dropping explicit mask columns does NOT erase missingness information already present in numerical features/preprocessing.",
                "This fold has repeatedly informed research. Four-contrast intervals do not cover full adaptive-search multiplicity.",
                "Rank-state screening uses training outcomes only; associations are not out-of-sample gains. No rank-state model was fitted.",
                "Five-row common-origin delay is conservative for shorter horizons. Observed cohort composition changes with missing labels.",
                "Only active-panel and within-family exact duplicates are audited; semantic/static-rank overlap remains explicit.",
                "Candidate ZIP contains private label-derived rank features; do not upload it to public GitHub or chat.",
                "No other folds, final-test evaluation, GPU, dependency installation, cloud or Git writes.",
            ],
        )
        save()
        u.atomic_json(directory / "review.json", report)
        return report
    except BaseException as exc:
        report.update(
            status="STOPPED",
            error=type(exc).__name__ + ": " + str(exc),
            finished_utc=utc(),
            error_traceback=__import__("traceback").format_exc(),
        )
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def verify_complete(root, r, network, session, u):
    lineage, evidence = identity(root, u)
    old = read_network(root, network, session, u)
    if (
        r.get("status") not in TERMINAL
        or r.get("lineage") != lineage
        or r.get("identity") != evidence
        or r.get("new_training_fits") != 2
        or r.get("control_refits") != 0
        or r.get("maximum_prediction_replay_error") != 0.0
        or r.get("final_test_evaluations") != 0
        or r.get("parents_unchanged") is not True
        or r.get("prior_network_unchanged") is not True
        or r.get("source_unchanged") is not True
        or r.get("graph_reconstructions") != 0
        or r.get("rank_feature_models_fitted") != 0
    ):
        raise Stop("Not a completed verified diagnostic run.")
    rows, allowed = decision_rows(r["results"], old, session.metric)
    if rows != r["comparison_rows"] or allowed != r["candidates_for_review"]:
        raise Stop("Stored diagnostic scores/decision differ.")
    if r["decision"] != (
        "REVIEW_NUMERIC_ONLY_ON_OTHER_FOLDS" if allowed else "STOP_TESTED_NETWORK_VARIANTS"
    ):
        raise Stop("Stored review decision changed.")
    lab = r["rank_lab"]
    if (
        lab["validation_rows_scored"] != 0
        or lab["new_training_fits"] != 0
        or not lab["prefix_replay_passed"]
        or lab["candidate_templates"] != 14
        or lab["screen_stop_exclusive"] != TRAIN_STOP
        or lab["source_origin_delay"] != DELAY
    ):
        raise Stop("Rank-lab training boundary changed.")
    short = [
        v["name"]
        for v in sorted(
            [v for v in lab["rows"] if v["screen_exclusion"] is None],
            key=lambda v: (-v["training_relevance"], v["name"]),
        )[:6]
    ]
    if lab["shortlist"] != short:
        raise Stop("Rank training shortlist changed.")
    directory = root / "artifacts/feature_diagnosis" / lineage
    if set(r["checkpoint_hashes"]) != set(VARIANTS):
        raise Stop("Incomplete diagnostic model inventory.")
    for n in VARIANTS:
        u.verify_stage(directory / "fold_0" / n, lineage, r["checkpoint_hashes"][n])
    u.verify_stage(directory / "rank_candidates", lineage, r["rank_checkpoint_hashes"])


def run(root=ROOT):
    root = Path(root)
    network, session, older, u = previous(root)
    ready = u.readiness(root)
    u.validate_runtime(root, ready)
    u.validate_source(root)
    read_network(root, network, session, u)
    out = u.safe_path(root, "logs/manual_feature_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    with u.safe_path(out, "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery = None
        if report_path(root).exists():
            r = u.read_json(report_path(root))
            if r.get("status") in TERMINAL:
                verify_complete(root, r, network, session, u)
                emit("COMPLETED_DIAGNOSIS_REUSED", new_fits=0)
                return {**r, "new_fits_this_notebook_call": 0}
            if r.get("status") != "READONLY_RECOVERY_READY":
                raise Stop(
                    "Interrupted diagnostic run exists; return report, no automatic retry/budget reset."
                )
            recovery = _validate_recovery_evidence(root, u, r.get("recovery"))
        lineage, _ = identity(root, u)
        if (root / "artifacts/feature_diagnosis" / lineage).exists():
            raise Stop("Orphan stage found; inspect instead of refitting.")
        if __import__("shutil").disk_usage(root).free < 1024**3:
            raise Stop("Need 1 GiB free; do not delete old files blindly.")
        carried = recovery["previous_attempt_seconds"] if recovery else 0.0
        remaining = LIMIT - carried
        u.atomic_json(
            report_path(root),
            {
                "status": "STARTING_WORKER",
                "lineage": lineage,
                "new_training_fits": 0,
                "feature_gate": "open",
                "started_utc": utc(),
                "recovery": recovery,
            },
        )
        # This state transition consumes the one reviewed recovery. A second failure
        # remains STOPPED and is never automatically rearmed by this helper.
        args = [
            ready["runtime"]["python"],
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--root",
            str(root),
        ]
        emit(
            "REMAINING_BUDGET",
            previous_attempt_seconds=carried,
            worker_limit_seconds=remaining,
            cumulative_limit_seconds=LIMIT,
        )
        started = time.monotonic()
        try:
            session._supervise_command(
                args, root, remaining, out / "feature_diagnosis_worker.log", u.environment(root)
            )
        except BaseException as exc:
            r = u.read_json(report_path(root))
            if r.get("status") == "STARTING_WORKER":
                r["new_training_fits"] = None
            r.update(
                status="STOPPED",
                supervisor_error=type(exc).__name__ + ": " + str(exc),
                supervised_seconds=round(time.monotonic() - started, 3),
                previous_attempt_seconds=carried,
                cumulative_supervised_seconds=round(carried + time.monotonic() - started, 3),
            )
            u.atomic_json(report_path(root), r)
            raise
        r = u.read_json(report_path(root))
        verify_complete(root, r, network, session, u)
        r.update(
            supervised_seconds=round(time.monotonic() - started, 3),
            new_fits_this_notebook_call=2,
            previous_attempt_seconds=carried,
            cumulative_supervised_seconds=round(carried + time.monotonic() - started, 3),
        )
        u.atomic_json(report_path(root), r)
        return r


def charts(r):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    figs = []
    old = pd.DataFrame(
        [{"variant": "current_market", "official_metric": r["control"]["official_metric"]}]
        + r["prior_network_comparisons"]
    )
    figs.append(
        ("08: what failed on the same 180 dates", px.bar(old, x="variant", y="official_metric"))
    )
    scored = pd.DataFrame(
        [{"variant": "current_market", "official_metric": r["control"]["official_metric"]}]
        + r["comparison_rows"]
    )
    figs.append(
        (
            "09: remove masks, retain identical numerical inputs",
            px.bar(scored, x="variant", y="official_metric"),
        )
    )
    changes = [
        {"variant": v["variant"], "reference": label, "delta": v[key]}
        for v in r["comparison_rows"]
        for label, key in [
            ("Saved baseline", "delta_vs_current_market"),
            ("Same numeric + masks", "delta_vs_masked_same_numeric"),
        ]
    ]
    f = px.bar(pd.DataFrame(changes), x="variant", y="delta", color="reference", barmode="group")
    f.add_hline(y=GAIN, line_dash="dash")
    figs.append(("Matched changes · +0.002 is an allocation rule, not significance", f))
    bounds = pd.DataFrame(
        [
            {
                "comparison": v["variant"] + " vs " + v["reference"],
                "delta": v["delta"],
                "plus": max(0, v["conditional_95_interval"][1] - v["delta"]),
                "minus": max(0, v["delta"] - v["conditional_95_interval"][0]),
            }
            for v in r["comparisons"]
            if v["block_dates"] == 20
        ]
    )
    f = px.scatter(bounds, x="delta", y="comparison", error_x="plus", error_x_minus="minus")
    f.add_vline(x=0, line_dash="dash")
    figs.append(("20-date conditional intervals · not search-wide correction", f))
    daily = []
    control = np.asarray(r["control"]["daily_rank_correlations"])
    for v in r["results"]:
        for date, value in zip(
            r["control"]["date_ids"],
            np.cumsum(np.asarray(v["metrics"]["daily_rank_correlations"]) - control),
            strict=True,
        ):
            daily.append(
                {
                    "date_id": date,
                    "variant": v["variant"],
                    "cumulative_correlation_difference": value,
                }
            )
    figs.append(
        (
            "Temporal concentration · NOT investment returns",
            px.line(
                pd.DataFrame(daily),
                x="date_id",
                y="cumulative_correlation_difference",
                color="variant",
            ),
        )
    )
    figs.append(
        (
            "Actual admission in diagnostic models",
            px.bar(pd.DataFrame(r["comparison_rows"]), x="variant", y="numeric_admitted"),
        )
    )
    release = []
    for h in [1, 2, 3, 4]:
        release.extend(
            [
                {"horizon": str(h), "event": "Label contract release", "row_offset": h + 1},
                {"horizon": str(h), "event": "Rank cohort usable", "row_offset": 5},
            ]
        )
    figs.append(
        (
            "New feature timing · common-origin cohort waits five rows",
            px.bar(
                pd.DataFrame(release), x="horizon", y="row_offset", color="event", barmode="group"
            ),
        )
    )
    rank = pd.DataFrame(r["rank_lab"]["rows"])
    figs.append(
        (
            "Rank-state candidate coverage · TRAINING only",
            px.bar(rank, x="training_coverage", y="name", orientation="h"),
        )
    )
    assoc = []
    for v in r["rank_lab"]["rows"]:
        if v["type"] == "numeric":
            for half, key in [
                ("Earlier training", "half_1_mean_ic"),
                ("Later training", "half_2_mean_ic"),
            ]:
                assoc.append({"feature": v["name"], "half": half, "mean_daily_association": v[key]})
    figs.append(
        (
            "Rank associations · TRAINING diagnostics, not new scores",
            px.bar(
                pd.DataFrame(assoc),
                x="mean_daily_association",
                y="feature",
                color="half",
                barmode="group",
                orientation="h",
            ),
        )
    )
    c = pd.DataFrame(r["rank_lab"]["coverage_by_training_block"])
    matrix = c.pivot(index="feature", columns="start", values="finite_fraction")
    figs.append(
        (
            "Does candidate availability change within training?",
            go.Figure(
                go.Heatmap(
                    z=matrix.to_numpy(),
                    x=matrix.columns.tolist(),
                    y=matrix.index.tolist(),
                    zmin=0,
                    zmax=1,
                    colorbar={"title": "Coverage"},
                )
            ),
        )
    )
    for title, f in figs:
        f.update_layout(
            title=title,
            height=470,
            margin={"l": 60, "r": 25, "t": 85, "b": 100},
            legend_title_text="",
        )
    for i in [7, 8, 9]:
        figs[i][1].update_layout(height=640, margin={"l": 360, "r": 25, "t": 85, "b": 75})
    figs[3][1].update_layout(margin={"l": 380, "r": 25, "t": 85, "b": 65})
    return figs


def private_backup(root, r, u):
    d = root / "artifacts/feature_diagnosis" / r["lineage"]
    paths = [d / "plan.json", d / "review.json"] + [
        d / "rank_candidates" / n for n in r["rank_checkpoint_hashes"]
    ]
    for v in VARIANTS:
        paths += [d / "fold_0" / v / n for n in r["checkpoint_hashes"][v]]
    pins = {str(p.relative_to(root)): u.digest(p) for p in paths}
    out = output_dir(root) / "feature_diagnosis_checkpoints.zip"
    temp = u.safe_path(out.parent, out.name + ".tmp")
    if out.is_symlink():
        raise Stop("Unsafe backup target.")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, str(p.relative_to(root)))
        z.writestr("SHA256SUMS.json", json.dumps(pins, sort_keys=True, indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip():
            raise Stop("Archive CRC check failed.")
        for n, h in pins.items():
            if hashlib.sha256(z.read(n)).hexdigest() != h:
                raise Stop("Archive content mismatch.")
    os.replace(temp, out)
    return {
        "path": str(out),
        "sha256": u.digest(out),
        "files": len(paths),
        "bytes": out.stat().st_size,
        "private": True,
        "contains_label_derived_rank_features_models_predictions": True,
        "raw_csvs_included": False,
        "off_disk_backup_confirmed": False,
    }


def finish(root, r, figures):
    root = Path(root)
    network, session, older, u = previous(root)
    verify_complete(root, r, network, session, u)
    if len(figures) != 10:
        raise Stop("Expected ten figures.")
    result = {**r}
    path = output_dir(root) / "feature_diagnosis_dashboard.html"
    result.update(
        dashboard=str(path),
        dashboard_sha256=older.export_dashboard(
            path, "Feature diagnosis and causal rank states", figures
        ),
        plotly_figures=10,
    )
    try:
        result["private_checkpoint_bundle"] = private_backup(root, r, u)
    except Exception as exc:
        result["backup_warning"] = type(exc).__name__ + ": " + str(exc)
    result.update(
        status="NOTEBOOK_AND_DIAGNOSIS_READY",
        notebook=str(root / "notebooks" / NOTEBOOK),
        updated_utc=utc(),
    )
    u.atomic_json(report_path(root), result)
    return result


def protocol_markdown():
    return """# Feature diagnosis and released-rank candidate laboratory

## Measured starting point
Notebook 08: baseline 0.403381; network availability 0.372697; synchronous
0.374725; directed 0.363208; joint 0.363892, on the same first 180 dates. All
numeric inputs were admitted. The network experiment failed its declared gate.
No old fit or graph construction is repeated in this milestone.

## Part A: isolate a representation decision, two fits only
The mask-only model was substantially lower. This motivates a *changed*
experiment rather than expanding the same negative panels. Remove the four
explicit network-availability columns while preserving the numerical cube
byte-for-byte. Fit synchronous-only (6) and directed-only (6), each added to the
same 87-template current_market panel. No union model; no extra mask control.
Replay the old baseline and both already-fitted masked counterparts first.
Keep the original learner, train-only preprocessing, missingness handling and
model settings. Six numerical inputs must be admitted or stop before fitting.
A refitted model changes all its splits: this is a representation ablation, not
proof that a coverage feature is universally harmful or a causal market effect.
Numerical missingness still exposes some availability even without the explicit
mask columns. Do not claim that this removes all missingness information.

Compare each result with the original and its own masked counterpart. +0.002
against both earns review, not promotion or automatic new folds. Threshold is
chosen after observing 08, so the analysis is diagnostic/adaptive. Four declared
contrasts use 10/20/40-date blocks; intervals condition on fitted models, not the
entire history of feature selection. Repeated first-fold use remains a limitation.

## Part B: engineer information in rank space, no fitted model
Predictive ranking and regime adaptation motivate representing a target's
historical ordinal state rather than just raw magnitudes or a static mean rank.
The existing repository has long raw-value released priors; the earlier scalar
rank-prior transfer was negative. This dynamic, common-origin cohort construction
is different but still unproven. No new labels are invented and no economic date,
carry, inventory, weather or external supply-chain mapping is assumed.

Target horizon h is released at origin+h+1. Since h is 1..4, a cross-sectional
cohort from origin s is usable at s+5. First shift the ENTIRE label matrix by 5;
only then rank observed labels. This avoids the common leakage mistake of ranking
a short-horizon label against longer-horizon outcomes not released yet. Shorter
horizons deliberately wait longer here to keep every cross-sectional cohort at
the same origin. Missing values (including sentinel -999999 and infinities) remain
missing, not zero. Average ties are mapped into [-1,1]. Sparse groups with fewer
than 3 labels are missing. Equal-valued groups map to zero, a representational
convention; constant dates are undefined in correlation diagnostics.

Twelve numeric templates: global and within-horizon latest ordinal position,
EMA21, EMA63, EMA21-minus-EMA63, latest-minus-EMA63 (10); global-minus-horizon
latest and EMA63 (2). Two coverage diagnostics: cohort observed fraction and own
rank availability over 63 rows. EMA min histories are 14 and 42 observations.
EMA state persists across missing observations; raw outcomes are not forward
filled. Coverage diagnostics are NEVER automatically included in a fitted model.

Only rows 0..1163 enter this lab. At the last training row, the newest source
origin is 1158. Runtime prefix equality is checked at 505 and 917 rows. Screen
rows252..1163 using pairwise-observed daily feature/target Spearman, split at708.
Require >=70% raw feature coverage, >=40 eligible dates in both halves, consistent
nonzero direction, and no exact duplicate against the active base or candidates.
Retain at most6 by the smaller absolute half-mean association. This is an
exploratory screen, not conditional importance or out-of-sample evidence. A
rejected candidate is not proven useless for nonlinear/interaction models.

## Research sources and limits
- MITSUI 26th-place author writeup (historical mean ranks and clustering):
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
  The author uses a static training mean of daily ranks, with zero filling in the
  example. We do not inherit that missing-value or full-prefix timing policy.
- MITSUI 15th-place author writeup (online adaptation and ranking-heavy training):
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th
  This motivates adapting the information state; it does not prove this feature
  recipe works. We do not reproduce the author's architecture/retraining claims.
- Existing source: src/commodity_prediction/domain/features.py at d142a4cb:
  horizon-delayed raw-value EWMs, market-pair pooling and cross-horizon priors.
- Bennett, Cucuringu and Reinert, https://arxiv.org/abs/2201.08283, original
  motivation for the prior incoming-peer network; no new network estimation here.

## Runtime and records
One 240-second worker covers checks, exact old-model replay, cached-feature load,
rank candidate construction/screening, two fits and matched diagnostics. 15-second
supervisor heartbeats; terminate process group at cap. Preserve independent sealed
model and candidate stages. A complete rerun reads/validates checkpoints with zero
fits. An incomplete run stops for diagnosis, not automatic budget resets/retries.
Self-contained HTML, JSON and notebook output are display artifacts; private ZIP
also contains label-derived candidates, models, and predictions. Download ZIP
privately; do not send it to ChatGPT or public GitHub. No space shutdown occurs.

Source/kernel/raw/parent checks precede model loading. Original tracked source and
08 checkpoints remain unchanged. No package installation, AWS/Git API call, final
evaluation, GPU, ensemble, more folds, or fitted rank model is executed. New files
are local additions; they are NOT already on GitHub. Feature engineering stays
open; highest reproduced pooled development metric remains 0.309709 unless a
comparable complete study changes it. No promise of a historical leaderboard win.
"""


def notebook_document():
    cells = []

    def md(s):
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
        )

    def code(s):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": s.splitlines(keepends=True),
            }
        )

    md(
        "# 09 · Diagnose the representation, then engineer released rank states\n\n"
        "**One bounded milestone:** two mask-removal fits plus a training-only 14-candidate laboratory. "
        "Do not rerun notebook 08. No old model refits, graph reconstruction, rank-feature fits, "
        "package installs, other folds, final test or cloud/Git writes.\n\n"
        "**Run All starts up to two new CPU fits.** Worker cap240 seconds; 15-second heartbeats. "
        "Use **Commodity - manual (verified)**. Stop the SageMaker space after saving."
    )
    code(
        "from pathlib import Path\nimport os, importlib.util\nimport pandas as pd\nfrom IPython.display import display\n"
        "ROOT=Path(os.environ.get('COMMODITY_MANUAL_PROJECT','/home/sagemaker-user/projects/commodity-prediction-manual'))\n"
        "spec=importlib.util.spec_from_file_location('feature_diagnosis',ROOT/'scripts/commodity_feature_diagnosis.py')\n"
        "support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)\n"
        "network,session,older,utilities=support.previous(ROOT)\nready=utilities.readiness(ROOT)\n"
        "utilities.validate_runtime(ROOT,ready)\nprint('Verified source:',ready['source_commit'])\n"
    )
    md(
        "## Why not immediately add another large feature family?\n\n"
        "The numerical graph panels were tested *with* four explicit availability inputs, and "
        "availability-only already underperformed. Two removal fits distinguish that representation "
        "choice from the numerical signals. This is a changed, diagnostic test—not an unchanged retry."
    )
    code(
        "old=support.read_network(ROOT,network,session,utilities)\n"
        "display(pd.DataFrame(old['comparison_rows'])[['variant','official_metric','delta_vs_current_market','numeric_admitted']])\n"
        "print('First-fold reference:',old['control']['official_metric'])\n"
        "print('Pooled reference (different scope):',old['prior_control_pooled'])\n"
        "display(pd.DataFrame([{'new_fit':n,'existing_masked_reference':v,'numeric_inputs':6,'explicit_masks':0} for n,v in support.VARIANTS.items()]))\n"
    )
    md(
        "## New feature laboratory: ordinal state under legal release timing\n\n"
        "For prediction t, a complete origin cohort uses outcomes from t−5 at the latest. "
        "Rank observed targets globally and within horizon; never rank against unreleased labels. "
        "Construct latest positions, 21/63-observation EMA states, their trend and innovations, "
        "plus global/within-horizon contrasts. Keep two coverage diagnostics separate.\n\n"
        "Twelve numerical + two coverage templates. Only training rows0–1163 are provided to this lab. "
        "Training-half agreement shortlists at most6; **no rank-state model is fitted**. "
        "Existing static-rank and raw-return-prior experiments are not rerun. See docs/manual_feature_diagnosis.md."
    )
    code(
        "display(pd.DataFrame([{'candidate':n,'kind':'numeric' if n in support.RANK_NUMERIC else 'coverage diagnostic'} for n in support.RANK_NUMERIC+support.RANK_MASKS]))\n"
        "print('Latest permitted source origin at training end:',1163-support.DELAY)\n"
    )
    md(
        "## Execute once\n\n"
        "This cell launches the bounded worker. Old original/masked predictions must replay exactly. "
        "The cached graph cube is used without re-estimating any link. If the worker stops, retain "
        "the error and report; do not reset the budget or delete completed stages."
    )
    code(
        "report=support.run(ROOT)\nprint('RESULT:',report['status'])\nprint('Diagnostic fits this call:',report['new_fits_this_notebook_call'])\n"
        "print('Network decision:',report['decision'])\nprint('Rank lab:',report['rank_decision'])\n"
        "figures=support.charts(report)\n"
    )
    titles = [
        "Prior matched result",
        "Numeric-only comparison",
        "Did removing explicit masks help?",
        "Uncertainty, not a promise",
        "Time concentration, not profit",
        "Feature admission",
        "Information availability",
        "Training coverage",
        "Training-half associations",
        "Availability stability",
    ]
    for i, title in enumerate(titles):
        md("## " + title)
        code(f"figures[{i}][1].show()\n")
    md(
        "## Candidate and result decisions\n\n"
        "A numeric panel requires +0.002 against both old references for review. This is a budget gate, "
        "not a significance claim. Repeated fold reuse remains adaptive; no automatic further folds. "
        "Rank-screen associations are not validation gains. A shortlist of zero is an informative result, "
        "not permission to alter the cutoff until features pass.\n\n"
        "Source and local additions have separate status: no GitHub commit or AWS synchronization is "
        "performed by this notebook. Preserve original worktrees and private records."
    )
    code(
        "display(pd.DataFrame(report['comparison_rows']))\n"
        "display(pd.DataFrame(report['rank_lab']['rows'])[['name','training_coverage','half_1_mean_ic','half_2_mean_ic','screen_exclusion']])\n"
        "print('Training-only shortlist:',report['rank_lab']['shortlist'])\n"
        "report=support.finish(ROOT,report,figures)\nprint('RESULT:',report['status'])\n"
        "print('REPORT:',support.report_path(ROOT))\nprint('DASHBOARD:',report['dashboard'])\n"
        "print('PRIVATE_BACKUP:',report.get('private_checkpoint_bundle',{}).get('path','not created'))\n"
        "if report.get('backup_warning'): print('BACKUP_WARNING:',report['backup_warning'])\n"
        "print('Save notebook; download JSON/HTML and keep ZIP privately. Then STOP SPACE, not Delete.')\n"
    )
    for i, c in enumerate(cells):
        c["id"] = f"diag09-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Commodity - manual (verified)",
                "language": "python",
                "name": "commodity-manual",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def install(root=ROOT):
    root = Path(root)
    network, session, older, u = previous(root)
    u.validate_source(root)
    read_network(root, network, session, u)
    nb = notebook_document()
    paths = {
        u.safe_path(root, "scripts/commodity_feature_diagnosis.py"): Path(__file__).read_bytes(),
        u.safe_path(root, "configs/manual_feature_diagnosis.json"): (
            json.dumps(plan(), sort_keys=True, indent=2) + "\n"
        ).encode(),
        u.safe_path(root, "docs/manual_feature_diagnosis.md"): protocol_markdown().encode(),
        u.safe_path(root, "notebooks/" + NOTEBOOK): (json.dumps(nb, indent=1) + "\n").encode(),
    }
    # Preflight every destination before publishing any new file.
    for p, body in paths.items():
        if p.exists():
            if p.suffix == ".ipynb":
                existing = u.read_json(p)
                if [(c["cell_type"], c.get("source")) for c in existing["cells"]] != [
                    (c["cell_type"], c["source"]) for c in nb["cells"]
                ]:
                    raise Stop("Existing notebook edits preserved: " + str(p))
            elif p.read_bytes() != body:
                raise Stop("Existing different file preserved: " + str(p))
    for p, body in paths.items():
        if not p.exists():
            u.write_new(p, body)
    print(
        "RESULT: FEATURE_DIAGNOSIS_NOTEBOOK_READY\nNOTEBOOK: "
        + str(root / "notebooks" / NOTEBOOK)
        + "\nKERNEL: Commodity - manual (verified)",
        flush=True,
    )
    return {"new_training_fits": 0, "files": [str(p) for p in paths]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    try:
        if a.worker:
            worker(a.root)
        else:
            install(a.root)
    except BaseException as exc:
        print("RESULT: STOPPED\nERROR: " + type(exc).__name__ + ": " + str(exc), flush=True)
        print(
            "Keep error/report, stop space, and do not repeat unchanged. No automatic shutdown.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
