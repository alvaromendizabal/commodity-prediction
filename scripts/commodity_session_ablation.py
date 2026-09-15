#!/usr/bin/env python3
"""Install notebook 07; run a bounded four-fit session-feature ablation manually.

Default action installs only. Notebook execution uses the already verified Python
kernel and immutable parent code/models. No AWS API, network, dependency install,
Git write, old-study rerun, further folds, or automatic space shutdown.
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
import queue
import signal
import subprocess
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-manual")
SHA = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
PARENT = "04544d4b1a1d63487a24e88749d03d3259326c63902f2d0c74214e711147d605"
FEATURE = "52d3650abd20481db1a88fe7793085360bb0b4b684c501518ae9f4cb1c9405ba"
LAB = "aefcc9201825211d64780701062212ae2b0ab1acf6ed17d84b0dad5821d01b98"
ROUND_HASH = "fc27479ee4821dec7e4f965e1d32c885f7cad0b157b6e1bb3b1cd7ac9822ea73"
VALIDATION_HASH = "7c8a7984f87c7ba8a48472194128a6f0173d58903f25c03f6b10c8dfa72dd14a"
LAB_HASH = "b6185ed15a0db01efa15aa70ae1d04e40ad6a6cdd7efb3a26f5550f8b391f50d"
LAB_PINS = {
    "candidates.npz": "71770350f2735a1285c5f3788fd3577345264cb13930e018ad6d88a148e80929",
    "inventory.json": "af9db32e574ed7759bc21c7c14ff960433f0eadf577ea6ef2b49fce3bf6ebc88",
    "manifest.json": "36757007181941848e57f66c35d2dd9fd3a8b549dd6923764b2a6e798d08cdc5",
}
SHORTLIST = [
    "session_pair__intraday_shock_z_63",
    "session_pair__intraday_shock_z_21",
    "session_pair__intraday_shock_z_126",
    "session_pair__overnight_drift_z_63",
    "session_pair__overnight_drift_z_126",
    "session_pair__intraday_drift_z_63",
    "session_pair__intraday_log_risk_ratio_63",
    "session_pair__overnight_risk_share_21",
    "session_pair__overnight_log_risk_ratio_21",
    "session_pair__intraday_drift_z_21",
    "session_pair__overnight_covariance_adjustment_21",
]
MASKS = ["session_pair__all_legs_applicable", "session_pair__paired_applicable"]
VARIANTS = ("session_support_only", "session_intraday", "session_overnight_risk", "session_joint")
CONTROL = 0.40338108742296147
TRAIN_STOP, VALIDATION_START, VALIDATION_STOP = 1164, 1169, 1349
LIMIT = 240.0
GAIN = 0.002
NOTEBOOK = "07_session_feature_ablation.ipynb"
TERMINAL = {"SESSION_ABLATION_REVIEW_READY", "NOTEBOOK_AND_SESSION_ABLATION_READY"}


class Stop(RuntimeError):
    """Retain all evidence; diagnose before repeating a failed operation."""


def utc() -> str:
    return datetime.now(UTC).isoformat()


def emit(stage: str, **fields) -> None:
    print(json.dumps({"utc": utc(), "stage": stage, **fields}, allow_nan=False), flush=True)


def previous(root: Path):
    p = root / "scripts/commodity_feature_round.py"
    if (
        p.is_symlink()
        or not p.is_file()
        or hashlib.sha256(p.read_bytes()).hexdigest() != ROUND_HASH
    ):
        raise Stop("The verified notebook-05/06 helper is absent or changed. Do not overwrite it.")
    spec = importlib.util.spec_from_file_location("verified_session_feature_parent", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m.support(root)


def output_dir(root: Path) -> Path:
    _, u = previous(root)
    return u.safe_path(root, "logs/manual_session_ablation")


def report_path(root: Path) -> Path:
    return output_dir(root) / "commodity_session_ablation_report.json"


def metric(daily, length: int | None = None) -> float:
    import numpy as np

    a = np.asarray(daily, dtype=float)
    if a.ndim != 1 or len(a) < 2 or (length is not None and len(a) != length):
        raise Stop("Unexpected daily-correlation shape.")
    if not np.isfinite(a).all() or (np.abs(a) > 1 + 1e-12).any() or a.std(ddof=0) <= 1e-12:
        raise Stop("Undefined or nonfinite correlation metric.")
    return float(a.mean() / a.std(ddof=0))


def close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def validate_reports(v: dict, s: dict) -> None:
    if (
        v.get("status") != "NOTEBOOK_AND_VALIDATION_REVIEW_READY"
        or v.get("source_commit") != SHA
        or v.get("decision") != "NO_STABLE_GAIN_IN_THIS_FAMILY"
        or v.get("model_checkpoints") != 9
        or v.get("control_refits") != 0
        or v.get("final_test_evaluations") != 0
    ):
        raise Stop("Notebook-05 completion evidence differs from the reviewed report.")
    if not close(v["control"]["official_metric"], 0.3097087232124053):
        raise Stop("The retained pooled reference score changed.")
    expected = {
        "normalized_price": 0.3057333590409221,
        "volume_confirmation": 0.3069106073242419,
        "normalized_joint": 0.29697041754207887,
    }
    rows = v.get("comparison_rows", [])
    if len(rows) != 3 or {r["variant"] for r in rows} != set(expected):
        raise Stop("Normalization comparison inventory changed.")
    if any(not close(r["official_metric"], expected[r["variant"]]) for r in rows):
        raise Stop("Normalization result changed.")
    if (
        s.get("status") != "NOTEBOOK_AND_SESSION_FEATURES_READY"
        or s.get("source_commit") != SHA
        or s.get("lineage") != LAB
        or s.get("candidate_templates") != 41
        or s.get("training_only_shortlist") != SHORTLIST
        or s.get("checkpoint_hashes") != LAB_PINS
        or s.get("screen_start_date") != 252
        or s.get("screen_stop_exclusive") != TRAIN_STOP
        or s.get("validation_rows_scored") != 0
        or s.get("new_training_fits") != 0
        or s.get("final_test_evaluations") != 0
        or not s.get("prefix_replay_passed")
        or s.get("supported_targets") != 89
        or s.get("supported_paired_targets") != 87
    ):
        raise Stop("Notebook-06 training-only candidate contract changed.")
    lookup = {r["name"]: r for r in s["rows"]}
    if len(lookup) != 41 or len(s["rows"]) != 41:
        raise Stop("Candidate names are missing or duplicated.")
    eligible = [r for r in s["rows"] if r["screen_exclusion"] is None and r["stable_direction"]]
    ranked = [
        r["name"]
        for r in sorted(eligible, key=lambda r: (-r["training_relevance"], r["name"]))[:12]
    ]
    if ranked != SHORTLIST:
        raise Stop("Training-only shortlist cannot be reconstructed.")


def read_reviews(root: Path, u) -> tuple[dict, dict]:
    vp = u.safe_path(root, "logs/manual_validation/commodity_validation_report.json")
    sp = u.safe_path(root, "logs/manual_session_features/commodity_session_features_report.json")
    if u.digest(vp) != VALIDATION_HASH or u.digest(sp) != LAB_HASH:
        raise Stop(
            "An inspected report changed. Return the changed report; do not rerun notebooks 05/06."
        )
    v, s = u.read_json(vp), u.read_json(sp)
    validate_reports(v, s)
    return v, s


def plan() -> dict:
    intraday = [n for n in SHORTLIST if n.startswith("session_pair__intraday_")]
    overnight = [n for n in SHORTLIST if n not in intraday]
    assert len(intraday) == 6 and len(overnight) == 5
    return {
        "study": "session_pair_training_shortlist_first_fold",
        "source_commit": SHA,
        "parent_lineage": PARENT,
        "candidate_lineage": LAB,
        "variants": {
            "session_support_only": MASKS,
            "session_intraday": MASKS + intraday,
            "session_overnight_risk": MASKS + overnight,
            "session_joint": MASKS + SHORTLIST,
        },
        "training_stop_exclusive": TRAIN_STOP,
        "validation_start": VALIDATION_START,
        "validation_stop_exclusive": VALIDATION_STOP,
        "warmup_dates": 252,
        "training_selected_numeric_candidates": 11,
        "new_support_indicators": 2,
        "max_new_fits": 4,
        "worker_cumulative_limit_seconds": LIMIT,
        "continue_threshold_vs_each_control": GAIN,
        "further_folds_in_this_helper": False,
        "feature_gate": "open",
        "promotion_allowed": False,
        "final_test_evaluations": 0,
        "model_policy": "Frozen histogram settings and preprocessing; replay current_market, never refit it.",
        "decision_rule": "For a numeric group to earn review, gain >=0.002 over BOTH saved current_market and fitted support-only controls, exact replays and nonzero admitted numeric candidates.",
        "rule_timing": "Declared after reviewing training-only session screen and negative normalization validation; no session fitted validation result observed.",
        "screening": "Fixed 11-name training shortlist, same model screening; exact duplicate audit against 87 current-market templates and within candidates. Not an exhaustive historical catalog overlap audit.",
    }


def identity(root: Path, u) -> tuple[str, dict]:
    installed = u.safe_path(root, "configs/manual_session_ablation.json")
    if installed.exists() and u.read_json(installed) != plan():
        raise Stop(
            "Installed session-ablation declaration changed; do not edit the gate after results."
        )
    obj = {
        "plan": plan(),
        "helper_sha256": u.digest(Path(__file__)),
        "candidate_helper_sha256": ROUND_HASH,
        "validation_report_sha256": VALIDATION_HASH,
        "session_report_sha256": LAB_HASH,
        "candidate_checkpoint_hashes": LAB_PINS,
        "domain_config_sha256": u.digest(root / "configs/domain_study.json"),
    }
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(), obj


def vector_signature(v) -> str:
    import numpy as np

    a = np.ascontiguousarray(v, dtype=np.float32).copy()
    a[np.isnan(a)] = np.float32(np.nan)
    a[a == 0] = np.float32(0)  # treat negative zero as the same observed value
    return hashlib.sha256(a.tobytes()).hexdigest()


def training_duplicate_audit(base_values, base_names: list[str], block, names: list[str]) -> list:
    if base_values.shape[:2] != block.shape[:2] or base_values.shape[0] < TRAIN_STOP:
        raise Stop("Duplicate audit axes changed.")
    reference = {
        vector_signature(base_values[252:TRAIN_STOP, :, i]): n for i, n in enumerate(base_names)
    }
    seen, rows = {}, []
    for n in MASKS + SHORTLIST:
        if n not in names:
            raise Stop("Planned candidate is absent: " + n)
        h = vector_signature(block[252:TRAIN_STOP, :, names.index(n)])
        duplicate = reference.get(h, seen.get(h))
        rows.append(
            {
                "name": n,
                "exact_duplicate_of": duplicate,
                "scope": "training rows 252:1164, all 424 aligned target positions",
                "sha256": h,
            }
        )
        seen.setdefault(h, n)
    return rows


def select_columns(block, names: list[str], variant: str):
    import numpy as np

    panels = plan()["variants"]
    if variant not in panels or len(names) != len(set(names)):
        raise Stop("Unknown panel or ambiguous candidate metadata.")
    selected = panels[variant]
    if set(selected) - set(names):
        raise Stop("A predeclared candidate is missing.")
    value = block[:, :, [names.index(n) for n in selected]]
    if np.isinf(value).any():
        raise Stop("An infinite candidate reached the fitted panel.")
    return value, list(selected)


def subset_diagnostic(y, pred, mask, label: str, old) -> dict:
    import numpy as np

    if not np.any(mask):
        return {"group": label, "targets": 0, "eligible_dates": 0, "official_metric": None}
    daily = old.daily_candidate_ic(pred.to_numpy()[:, mask], y.to_numpy()[:, mask])
    use = daily[np.isfinite(daily)]
    return {
        "group": label,
        "targets": int(np.sum(mask)),
        "eligible_dates": len(use),
        "metric_scope": "Subset diagnostic with at least three usable targets per date; not the full 424-target metric",
        "official_metric": metric(use) if len(use) > 1 and use.std(ddof=0) > 1e-12 else None,
    }


def comparison_rows(results: list, control: dict) -> tuple[list, list]:
    if len(results) != 4 or {r["variant"] for r in results} != set(VARIANTS):
        raise Stop("Expected exactly the four planned fits.")
    if not close(metric(control["daily_rank_correlations"], 180), control["official_metric"]):
        raise Stop("Reference metric cannot be reproduced.")
    scores = {r["variant"]: metric(r["metrics"]["daily_rank_correlations"], 180) for r in results}
    rows, eligible = [], []
    for r in results:
        n, s = r["variant"], r["selection"]
        if (r["fold"], r["train_stop"], r["validation_start"], r["validation_stop"]) != (
            0,
            1164,
            1169,
            1349,
        ):
            raise Stop("Unexpected fitting or evaluation dates.")
        if r["metrics"]["date_ids"] != list(range(1169, 1349)) or not close(
            scores[n], r["metrics"]["official_metric"]
        ):
            raise Stop("Result metric/date alignment mismatch.")
        if s["candidate_templates"] != s["retained_templates"] + s["rejected_templates"]:
            raise Stop("Feature accounting does not balance.")
        if s["retained_templates"] != len(s["selected_names"]):
            raise Stop("Selected feature count disagrees with names.")
        numeric = [v for v in s["selected_names"] if v in SHORTLIST]
        if n != "session_support_only" and not numeric:
            raise Stop(
                "No numeric session feature admitted; do not interpret this as a feature test."
            )
        if n == "session_support_only" and (
            numeric or not set(MASKS).intersection(s["selected_names"])
        ):
            raise Stop("Invalid mask-only control.")
        delta = scores[n] - control["official_metric"]
        delta_mask = scores[n] - scores["session_support_only"]
        passes = n != "session_support_only" and delta >= GAIN and delta_mask >= GAIN
        if passes:
            eligible.append(n)
        rows.append(
            {
                "variant": n,
                "official_metric": scores[n],
                "delta_vs_current_market": delta,
                "delta_vs_support_only": delta_mask,
                "numeric_candidates": len(plan()["variants"][n]) - 2,
                "numeric_admitted": len(numeric),
                "support_indicators_admitted": len(set(MASKS).intersection(s["selected_names"])),
                "candidate_templates": s["candidate_templates"],
                "retained_templates": s["retained_templates"],
                "rejected_templates": s["rejected_templates"],
                "passes_review_gate": passes,
            }
        )
    return rows, eligible


def check_cached_feature_prefix(root: Path, u, block, metadata, pairs) -> None:
    import numpy as np

    stage = u.safe_path(root, "artifacts/session_pair_candidates/" + LAB)
    u.verify_stage(stage, LAB, LAB_PINS)
    inv = u.read_json(stage / "inventory.json")
    if (
        inv["features"] != metadata
        or inv["targets"] != pairs.target.tolist()
        or inv["shortlist"] != SHORTLIST
    ):
        raise Stop("Candidate checkpoint metadata changed.")
    with np.load(stage / "candidates.npz", allow_pickle=False) as a:
        if a["values"].shape != (TRAIN_STOP, 424, 41) or a["values"].dtype != np.float32:
            raise Stop("Unexpected candidate checkpoint shape or dtype.")
        np.testing.assert_array_equal(a["dates"], np.arange(TRAIN_STOP))
        np.testing.assert_array_equal(block[:TRAIN_STOP], a["values"])


def fit_one(
    root,
    u,
    directory,
    lineage,
    variant,
    base,
    block,
    names,
    y,
    pairs,
    fold,
    config,
    started,
    report,
):
    """Fit one new child, or verify/replay a sealed child; never replace an unsealed stage."""
    import joblib
    import numpy as np
    import pandas as pd

    from commodity_prediction.domain.attribution.run import fit_stage
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.experiment import Experiment
    from commodity_prediction.domain.model import prepare, select

    value, selected_names = select_columns(block, names, variant)
    panel = Panel(
        np.concatenate([base.values, value], axis=2),
        base.dates,
        base.targets,
        base.names + selected_names,
        dict(base.source_series),
    )
    panel.validate()
    settings = {**config, "max_features": len(panel.names), "max_abs_correlation": 1.01}
    stage = u.safe_path(directory, "fold_0/" + variant)
    if stage.exists() and not (stage / "manifest.json").exists():
        raise Stop("Unsealed stage preserved; inspect it before resume: " + str(stage))
    stats = None
    reused = stage.exists()
    if reused:
        u.verify_stage(stage, lineage)
    else:
        emit(
            "training_only_screen",
            variant=variant,
            completed=len(report["results"]),
            total=4,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        stats = prepare(panel, y, fold.train_stop, settings)
        _, audit = select(
            stats,
            Experiment(variant, algorithm="histogram").candidates(panel.names),
            settings,
            False,
        )
        need = set(MASKS) if variant == "session_support_only" else set(SHORTLIST)
        if not need.intersection(audit["selected_names"]):
            raise Stop("No planned added information survives training screening: " + variant)
        if set(audit["rejection_reasons"]).intersection(
            {"correlated", "feature_budget", "unstable_sign"}
        ):
            raise Stop("Unreviewed feature screening exclusion.")
        report["fit_attempts_this_call"] += 1
        u.atomic_json(report_path(root), report)
    result = fit_stage(
        panel,
        y,
        pairs,
        fold,
        Experiment(variant, algorithm="histogram"),
        settings,
        stats,
        stage,
        lineage,
    )
    saved = pd.read_parquet(stage / "predictions.parquet")
    replay = joblib.load(stage / "model.joblib").predict(
        panel, fold.validation_start, fold.validation_stop
    )
    error = u.exact_predictions(saved, replay)
    hashes = u.verify_stage(stage, lineage)
    if set(hashes) != {"manifest.json", "model.joblib", "predictions.parquet", "result.json"}:
        raise Stop("Unexpected files in fitted checkpoint.")
    report["checkpoint_hashes"][variant] = hashes
    report["results"].append(result)
    report["new_training_fits"] += int(not reused)
    report["cached_fits_reused"] += int(reused)
    report["maximum_prediction_replay_error"] = max(
        report["maximum_prediction_replay_error"], error
    )
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    u.atomic_json(report_path(root), report)
    emit(
        "checkpoint_verified",
        variant=variant,
        completed=len(report["results"]),
        total=4,
        new_fits=report["new_training_fits"],
        reused=reused,
        official_metric=result["metrics"]["official_metric"],
    )
    del stats, panel, replay
    gc.collect()
    return saved


def worker(root: Path) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    from commodity_prediction.data import Fold
    from commodity_prediction.domain.catalog import Panel
    from commodity_prediction.domain.market_path.features import build_panel
    from commodity_prediction.domain.run import load_panel
    from commodity_prediction.studies.evaluation import compare_predictions, evaluate

    old, u = previous(root)
    started = time.monotonic()
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    lineage, evidence = identity(root, u)
    directory = u.safe_path(root, "artifacts/session_pair_ablation/" + lineage)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "commodity-prediction",
        "status": "RUNNING",
        "source_commit": SHA,
        "lineage": lineage,
        "parent_lineage": PARENT,
        "candidate_lineage": LAB,
        "plan": plan(),
        "identity": evidence,
        "feature_gate": "open",
        "promotion_allowed": False,
        "final_test_evaluations": 0,
        "control_refits": 0,
        "new_training_fits": 0,
        "fit_attempts_this_call": 0,
        "cached_fits_reused": 0,
        "maximum_prediction_replay_error": 0.0,
        "results": [],
        "checkpoint_hashes": {},
        "started_utc": utc(),
        "aws_api_calls": 0,
        "github_writes": False,
        "validation_dates": 180,
        "study_limit_seconds": LIMIT,
        "subgroup_diagnostics": [],
    }
    parents = None

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        u.atomic_json(report_path(root), report)

    def expiry(*_):
        raise Stop("240-second session-study worker deadline reached; valid checkpoints retained.")

    signal.signal(signal.SIGALRM, expiry)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        save()
        ready = u.readiness(root)
        u.validate_runtime(root, ready)
        u.validate_source(root)
        v, s = read_reviews(root, u)
        report["prior_normalization_comparisons"] = v["comparison_rows"]
        report["prior_control_pooled"] = v["control"]["official_metric"]
        report["training_screen"] = [r for r in s["rows"] if r["name"] in SHORTLIST]
        report["supported_targets"], report["unsupported_targets"] = 89, 335
        parents = u.checkpoint_snapshot(root, ready)
        for rel, h in u.RAW.items():
            if u.digest(u.safe_path(root, "data/raw/" + rel)) != h:
                raise Stop("Raw file checksum changed.")
        final = u.read_json(root / "configs/final_evaluation.json")
        if final["evaluated"] or final["final_test_start_date_id"] != 1714:
            raise Stop("Final-test boundary changed.")
        emit(
            "load_first_fold_only",
            input_rows=VALIDATION_STOP,
            latest_scored_origin=VALIDATION_STOP - 1,
        )
        x = pd.read_csv(root / "data/raw/train.csv", nrows=VALIDATION_STOP).set_index("date_id")
        y = (
            pd.read_csv(root / "data/raw/train_labels.csv", nrows=VALIDATION_STOP)
            .set_index("date_id")
            .replace(-999999, np.nan)
        )
        pairs = pd.read_csv(root / "data/raw/target_pairs.csv")
        if (
            len(x) != VALIDATION_STOP
            or x.index.tolist() != list(range(VALIDATION_STOP))
            or not x.index.equals(y.index)
            or pairs.target.tolist() != list(y.columns)
            or list(y.columns) != [f"target_{i}" for i in range(424)]
        ):
            raise Stop("First-fold input schema/order changed.")
        config = u.read_json(root / "configs/domain_study.json")
        if config["warmup_dates"] != 252 or config["threads"] != 4:
            raise Stop("Model preprocessing configuration changed.")
        u.atomic_json(directory / "plan.json", evidence)
        with threadpool_limits(limits=4):
            original = load_panel(root / "artifacts" / FEATURE / "features")
            if original.dates[:VALIDATION_STOP] != list(range(VALIDATION_STOP)):
                raise Stop("Frozen parent dates changed.")
            # Read the immutable cached feature panel, but construct/replay only the first-fold prefix.
            prefix = Panel(
                original.values[:VALIDATION_STOP],
                original.dates[:VALIDATION_STOP],
                original.targets,
                original.names,
                dict(original.source_series),
            )
            base, _ = build_panel(prefix, x, pairs, "current_market")
            del original, prefix
            gc.collect()
            cp = root / "artifacts" / PARENT / "fold_0/current_market"
            saved = pd.read_parquet(cp / "predictions.parquet")
            replay = joblib.load(cp / "model.joblib").predict(
                base, VALIDATION_START, VALIDATION_STOP
            )
            u.exact_predictions(saved, replay)
            control = evaluate(y.loc[saved.index], saved, pairs)
            if saved.index.tolist() != list(range(1169, 1349)) or not close(
                control["official_metric"], CONTROL
            ):
                raise Stop(
                    "Frozen first-fold reference failed exact replay or metric reproduction."
                )
            report["control"] = control
            emit("SAVED_CONTROL_REPLAY_PASSED", score=CONTROL, new_control_fits=0)
            block, metadata, applicable, paired = old.session_features(x, pairs)
            check_cached_feature_prefix(root, u, block, metadata, pairs)
            if applicable.sum() != 89 or (applicable & paired).sum() != 87:
                raise Stop("Session applicability changed.")
            names = [r["name"] for r in metadata]
            report["candidate_prefix_exact"] = True
            report["duplicate_audit"] = training_duplicate_audit(
                base.values, base.names, block, names
            )
            fold = Fold(0, TRAIN_STOP, VALIDATION_START, VALIDATION_STOP)
            for mask, label in [
                (applicable, "complete_bar_targets"),
                (~applicable, "other_targets"),
            ]:
                report["subgroup_diagnostics"].append(
                    {
                        "variant": "current_market",
                        **subset_diagnostic(y.loc[saved.index], saved, mask, label, old),
                    }
                )
            for variant in VARIANTS:
                pred = fit_one(
                    root,
                    u,
                    directory,
                    lineage,
                    variant,
                    base,
                    block,
                    names,
                    y,
                    pairs,
                    fold,
                    config,
                    started,
                    report,
                )
                for mask, label in [
                    (applicable, "complete_bar_targets"),
                    (~applicable, "other_targets"),
                ]:
                    report["subgroup_diagnostics"].append(
                        {
                            "variant": variant,
                            **subset_diagnostic(y.loc[pred.index], pred, mask, label, old),
                        }
                    )
                del pred
            rows, eligible = comparison_rows(report["results"], control)
            daily = {
                r["variant"]: np.asarray(r["metrics"]["daily_rank_correlations"])
                for r in report["results"]
            }
            daily["current_market"] = np.asarray(control["daily_rank_correlations"])
            contrasts = [(n, "current_market") for n in VARIANTS] + [
                (n, "session_support_only") for n in VARIANTS[1:]
            ]
            report["comparisons"] = compare_predictions(daily, [180], contrasts, config)
            report["comparison_rows"], report["candidates_for_review"] = rows, eligible
            report["decision"] = (
                "REVIEW_BEFORE_OTHER_FOLDS" if eligible else "STOP_FITTED_SESSION_FORMULATION"
            )
            report["uncertainty_scope"] = (
                "Seven first-fold contrasts, conditional on these fits; no correction for earlier adaptive research or training shortlist selection."
            )
            report["limitations"] = [
                "Previously inspected first development fold; not new holdout or leaderboard evidence.",
                "Only 89 of 424 targets have complete-leg OHLC definitions; diagnostics expose coverage and pooled-model effects on other targets.",
                "Training association is not causal importance. No feature/learner combination is universally declared best.",
                "Bars follow supplied row alignment, not independently synchronized international sessions.",
                "No additional folds, external fundamental data, hyperparameter changes, or ensemble are executed here.",
                "Private ZIP remains on the same EBS disk until the user downloads it elsewhere.",
            ]
        report["parents_unchanged"] = u.checkpoint_snapshot(root, ready) == parents
        if not report["parents_unchanged"]:
            raise Stop("Frozen dependency bytes changed during experiment.")
        u.verify_stage(root / "artifacts/session_pair_candidates" / LAB, LAB, LAB_PINS)
        u.validate_source(root)
        report["source_unchanged"] = True
        report["status"] = "SESSION_ABLATION_REVIEW_READY"
        report["finished_utc"] = utc()
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


def supervise(root: Path, seconds: float, u) -> None:
    cmd = [
        u.readiness(root)["runtime"]["python"],
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--root",
        str(root),
    ]
    log = output_dir(root) / "session_ablation_worker.log"
    _supervise_command(cmd, root, seconds, log, u.environment(root))


def _supervise_command(cmd, root, seconds, log, env=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise Stop("No worker time remains.")
    process = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    q = queue.Queue()

    def stream():
        for line in process.stdout:
            q.put(line)
        q.put(None)

    threading.Thread(target=stream, daemon=True).start()
    start = time.monotonic()
    next_beat = start + 15
    ended = False
    try:
        with log.open("a") as f:
            while not ended or process.poll() is None:
                now = time.monotonic()
                if now - start >= seconds:
                    raise Stop(
                        "Session study supervisor deadline reached; do not reset its budget."
                    )
                try:
                    line = q.get(timeout=min(0.25, max(0.01, seconds - (now - start))))
                    if line is None:
                        ended = True
                    else:
                        print(line, end="", flush=True)
                        f.write(line)
                        f.flush()
                except queue.Empty:
                    pass
                if now >= next_beat:
                    emit(
                        "heartbeat",
                        worker_elapsed_seconds=round(now - start, 1),
                        limit_seconds=seconds,
                    )
                    next_beat = now + 15
            if process.wait() != 0:
                raise Stop("Worker stopped. Read its saved report/log rather than rerunning.")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        raise
    finally:
        process.stdout.close()


def verify_complete(root, report, u):
    lineage, _ = identity(root, u)
    if (
        report.get("status") not in TERMINAL
        or report.get("lineage") != lineage
        or report.get("source_commit") != SHA
        or report.get("control_refits") != 0
        or report.get("maximum_prediction_replay_error") != 0.0
        or report.get("final_test_evaluations") != 0
        or report.get("parents_unchanged") is not True
        or report.get("source_unchanged") is not True
        or not isinstance(report.get("new_training_fits"), int)
        or not 0 <= report["new_training_fits"] <= 4
    ):
        raise Stop("Not a completed verified session study.")
    rows, eligible = comparison_rows(report["results"], report["control"])
    if rows != report["comparison_rows"] or eligible != report["candidates_for_review"]:
        raise Stop("Stored decision cannot be reconstructed.")
    directory = root / "artifacts/session_pair_ablation" / lineage
    for n in VARIANTS:
        u.verify_stage(directory / "fold_0" / n, lineage, report["checkpoint_hashes"][n])
    if len(report["checkpoint_hashes"]) != 4:
        raise Stop("Incomplete sealed model inventory.")


def run(root: Path = ROOT) -> dict:
    root = Path(root)
    old, u = previous(root)
    ready = u.readiness(root)
    u.validate_runtime(root, ready)
    u.validate_source(root)
    read_reviews(root, u)
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    with u.safe_path(out, "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report_path(root).exists():
            report = u.read_json(report_path(root))
            if report.get("status") not in TERMINAL:
                raise Stop(
                    "Prior interrupted/failed session study exists. Return its report; no automatic retraining."
                )
            verify_complete(root, report, u)
            emit("COMPLETE_SESSION_STUDY_REUSED", new_fits=0)
            return {**report, "new_fits_this_notebook_call": 0}
        lineage, _ = identity(root, u)
        directory = root / "artifacts/session_pair_ablation" / lineage
        if directory.exists():
            raise Stop(
                "Saved session artifacts already exist without a completion receipt. Recover them; do not refit."
            )
        if __import__("shutil").disk_usage(root).free < 1024**3:
            raise Stop("Less than 1 GiB free. Do not delete old checkpoints automatically.")
        u.atomic_json(
            report_path(root),
            {
                "project": "commodity-prediction",
                "status": "STARTING_WORKER",
                "source_commit": SHA,
                "lineage": lineage,
                "new_training_fits": 0,
                "feature_gate": "open",
                "final_test_evaluations": 0,
                "started_utc": utc(),
            },
        )
        started = time.monotonic()
        try:
            supervise(root, LIMIT, u)
        except BaseException as exc:
            report = u.read_json(report_path(root))
            if report.get("status") == "STARTING_WORKER":
                report["new_training_fits"] = None
            report.update(
                status="STOPPED",
                supervisor_error=type(exc).__name__ + ": " + str(exc),
                supervised_seconds=round(time.monotonic() - started, 3),
            )
            u.atomic_json(report_path(root), report)
            raise
        report = u.read_json(report_path(root))
        verify_complete(root, report, u)
        report["supervised_seconds"] = round(time.monotonic() - started, 3)
        report["new_fits_this_notebook_call"] = report["new_training_fits"]
        u.atomic_json(report_path(root), report)
        return report


def charts(report: dict) -> list:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    figs = []
    last = [{"variant": "current_market", "score": report["prior_control_pooled"]}] + [
        {"variant": r["variant"], "score": r["official_metric"]}
        for r in report["prior_normalization_comparisons"]
    ]
    figs.append(
        (
            "Completed normalization study: no retained gain · 535 dates",
            px.bar(pd.DataFrame(last), x="variant", y="score", text_auto=".6f"),
        )
    )
    train = [
        {
            "feature": r["name"].removeprefix("session_pair__"),
            "half": "early training",
            "IC": r["train_half_1_mean_ic"],
        }
        for r in report["training_screen"]
    ]
    train += [
        {
            "feature": r["name"].removeprefix("session_pair__"),
            "half": "later training",
            "IC": r["train_half_2_mean_ic"],
        }
        for r in report["training_screen"]
    ]
    figs.append(
        (
            "Why these 11: training-half association, not validation gains",
            px.bar(pd.DataFrame(train), y="feature", x="IC", color="half", barmode="group"),
        )
    )
    figs.append(
        (
            "Representation coverage: a narrow mechanism, not all markets",
            px.bar(
                pd.DataFrame(
                    {"target_group": ["Complete bars", "Other targets"], "targets": [89, 335]}
                ),
                x="target_group",
                y="targets",
                text="targets",
            ),
        )
    )
    rows = report["comparison_rows"]
    scores = [{"variant": "current_market", "score": report["control"]["official_metric"]}] + [
        {"variant": r["variant"], "score": r["official_metric"]} for r in rows
    ]
    figs.append(
        (
            "Same 180 first-fold dates · official metric",
            px.bar(pd.DataFrame(scores), x="variant", y="score", text_auto=".6f"),
        )
    )
    deltas = [
        {"variant": r["variant"], "reference": ref, "delta": r[key]}
        for r in rows
        for ref, key in [
            ("Saved baseline", "delta_vs_current_market"),
            ("Support-only", "delta_vs_support_only"),
        ]
    ]
    f = px.bar(pd.DataFrame(deltas), x="variant", y="delta", color="reference", barmode="group")
    f.add_hline(y=GAIN, line_dash="dash")
    figs.append(("Does content beat both controls? Dashed line = review threshold", f))
    f = go.Figure()
    for r in report["comparisons"]:
        if r["block_dates"] != 20:
            continue
        label = r["variant"] + " − " + r["reference"]
        lo, hi = r["conditional_95_interval"]
        f.add_trace(go.Scatter(x=[lo, hi], y=[label, label], mode="lines", showlegend=False))
        f.add_trace(go.Scatter(x=[r["delta"]], y=[label], mode="markers", showlegend=False))
    f.add_vline(x=0, line_dash="dot")
    f.update_xaxes(title="Metric change; paired 20-date conditional 95% interval")
    figs.append(("First-fold uncertainty: not correction for adaptive research", f))
    f = go.Figure()
    control = np.asarray(report["control"]["daily_rank_correlations"])
    for r in report["results"]:
        d = np.asarray(r["metrics"]["daily_rank_correlations"])
        f.add_trace(
            go.Scatter(
                x=r["metrics"]["date_ids"],
                y=np.cumsum(d - control),
                mode="lines",
                name=r["variant"],
            )
        )
    f.update_yaxes(title="Cumulative daily-correlation difference (not P&L)")
    figs.append(("Is improvement concentrated in a few dates?", f))
    admission = [
        {"variant": r["variant"], "type": typ, "count": r[key]}
        for r in rows
        for typ, key in [
            ("Numeric admitted", "numeric_admitted"),
            ("Support flags admitted", "support_indicators_admitted"),
        ]
    ]
    figs.append(
        (
            "Features actually admitted by training-only screening",
            px.bar(pd.DataFrame(admission), x="variant", y="count", color="type", barmode="group"),
        )
    )
    figs.append(
        (
            "Subset diagnostics · compare variants within a subset only",
            px.bar(
                pd.DataFrame(report["subgroup_diagnostics"]),
                x="group",
                y="official_metric",
                color="variant",
                barmode="group",
            ),
        )
    )
    for title, f in figs:
        f.update_layout(
            title=title, height=450, margin=dict(l=65, r=30, t=90, b=100), legend_title_text=""
        )
    figs[1][1].update_layout(height=620)
    figs[5][1].update_layout(height=580, margin=dict(l=300, r=30, t=80, b=60))
    return figs


def private_backup(root, report, u) -> dict:
    verify_complete(root, report, u)
    base = root / "artifacts/session_pair_ablation" / report["lineage"]
    paths = [base / "plan.json", base / "review.json"]
    for n in VARIANTS:
        paths += [base / "fold_0" / n / f for f in report["checkpoint_hashes"][n]]
    paths += [report_path(root)]
    pins = {str(p.relative_to(root)): u.digest(p) for p in paths}
    dest = output_dir(root) / "session_ablation_checkpoints.zip"
    temp = u.safe_path(dest.parent, dest.name + ".tmp")
    if dest.is_symlink():
        raise Stop("Unsafe backup destination.")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in paths:
            z.write(p, str(p.relative_to(root)))
        z.writestr("SHA256SUMS.json", json.dumps(pins, sort_keys=True, indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip():
            raise Stop("ZIP CRC verification failed.")
        for name, h in pins.items():
            if hashlib.sha256(z.read(name)).hexdigest() != h:
                raise Stop("ZIP readback mismatch.")
    os.replace(temp, dest)
    return {
        "path": str(dest),
        "sha256": u.digest(dest),
        "bytes": dest.stat().st_size,
        "files": len(paths),
        "private": True,
        "raw_data_included": False,
        "off_disk_backup_confirmed": False,
    }


def finish(root, report, figures):
    root = Path(root)
    old, u = previous(root)
    verify_complete(root, report, u)
    if len(figures) != 9:
        raise Stop("Expected nine notebook figures.")
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    result = {**report}
    path = out / "session_ablation_dashboard.html"
    result.update(
        dashboard=str(path),
        dashboard_sha256=old.export_dashboard(path, "Session feature ablation", figures),
        plotly_figures=9,
    )
    # Rendering/backups cannot call the fitting worker. A failure leaves the fit receipt reusable.
    try:
        result["private_checkpoint_bundle"] = private_backup(root, report, u)
    except Exception as exc:
        result["backup_warning"] = type(exc).__name__ + ": " + str(exc)
    result.update(
        status="NOTEBOOK_AND_SESSION_ABLATION_READY",
        updated_utc=utc(),
        notebook=str(root / "notebooks" / NOTEBOOK),
    )
    u.atomic_json(report_path(root), result)
    return result


def notebook_document() -> dict:
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
                "source": s.splitlines(keepends=True),
                "execution_count": None,
                "outputs": [],
            }
        )

    md(
        "# 07 · Do session features improve predictions, or only describe coverage?\n\n"
        "**One bounded feature experiment: at most four new CPU fits on the first fold.** "
        "Keep the saved `current_market` reference. The completed normalization family did not improve the pooled metric; "
        "do not append its rejected features or rerun notebooks 04–06. The 11 candidates here were shortlisted using training data only.\n\n"
        "**Run All starts fitting in the execution cell.** Worker cap: **240 seconds**, including checks/replay. "
        "No other folds, parameter tuning, package install, external data download, ensemble or final-test evaluation. "
        "Use **Commodity - manual (verified)**. Keep old directories containing its Python environment."
    )
    code(
        "from pathlib import Path\nimport os, importlib.util\nimport pandas as pd\nfrom IPython.display import display\n"
        "ROOT=Path(os.environ.get('COMMODITY_MANUAL_PROJECT','/home/sagemaker-user/projects/commodity-prediction-manual'))\n"
        "spec=importlib.util.spec_from_file_location('session_ablation',ROOT/'scripts/commodity_session_ablation.py')\n"
        "support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)\n"
        "parent,previous=support.previous(ROOT)\nready=previous.readiness(ROOT)\nprevious.validate_runtime(ROOT,ready)\n"
        "print('Source:',ready['source_commit'])\nprint('No training in this cell.')\n"
    )
    md(
        "## Hypothesis and information timing\n\n"
        "The signed overnight or intraday move of a target pair is formed **before** standardization. "
        "Reference means, risks and covariance use rows strictly before the current row. The training screen selected "
        "six intraday candidates and five overnight/risk candidates, of which one is overnight risk share. "
        "Only 89/424 targets have complete-leg bar definitions. Add a **support-only control** before attributing gains to numeric features.\n\n"
        "The source family, windows (21/63/126), clipping rules, shortlist and model settings remain frozen. "
        "The old 1164-row candidate checkpoint must exactly match the recomputed prefix before a new fit. "
        "An exact duplicate audit compares candidates with the current 87-template base; historical families outside that base "
        "are not claimed exhaustively deduplicated. Training-only model screening records all admission/rejection counts."
    )
    code(
        "validation,screen=support.read_reviews(ROOT,previous)\n"
        "display(pd.DataFrame(validation['comparison_rows'])[['variant','official_metric','matched_delta','positive_folds']])\n"
        "display(pd.DataFrame([{'panel':k,'support_flags':2,'numeric_features':len(v)-2} for k,v in support.plan()['variants'].items()]))\n"
        "print('Train: rows 0–1163; fitting warmup 252; validation: 1169–1348. No later labels are read.')\n"
    )
    md(
        "## Preregistered first-fold decision\n\n"
        "For an information-bearing panel to earn **review**, its official-metric gain must be at least **0.002** "
        "over **both** the saved `current_market` model and the support-only model, with nonzero numeric admission and exact replays. "
        "This rule was declared before session-model validation, but after earlier studies used this development fold. "
        "It is a cost-control screen, not unbiased significance or model promotion. "
        "Seven matched contrasts use 10/20/40-date blocks; intervals condition on these fitted models and do not correct prior adaptive research."
    )
    md(
        "## Execute one bounded experiment\n\n"
        "This cell may train four models. It replays, but does not refit, the old baseline. Each completed child is sealed separately. "
        "Do not start a parallel terminal run. If a previous run stopped, return its report rather than retry unchanged. "
        "Completed results reopen without refitting; missing dashboard output can be regenerated independently."
    )
    code(
        "report=support.run(ROOT)\nprint('RESULT:',report['status'])\n"
        "print('New fits this call:',report.get('new_fits_this_notebook_call',report['new_training_fits']))\n"
        "print('DECISION:',report['decision'])\ndisplay(pd.DataFrame(report['comparison_rows']))\nfigures=support.charts(report)\n"
    )
    titles = [
        "Prior family conclusion",
        "Training shortlist rationale",
        "Structural coverage",
        "First-fold performance",
        "Matched additions versus two controls",
        "Uncertainty",
        "Temporal concentration",
        "Actual feature admission",
        "Subset effects and limitations",
    ]
    for i, title in enumerate(titles):
        md("## " + title)
        code(f"figures[{i}][1].show(renderer='plotly_mimetype')\n")
    md(
        "## Interpretation and next research gate\n\n"
        "A good first-fold result must survive later periods; a failed panel should not be tuned unchanged. "
        "A support-only gain is not proof that price-session content helps. A change on the other 335 targets can occur "
        "because the pooled model is refitted; it is not evidence that those targets acquired OHLC data. "
        "No pooled three-fold or leaderboard score is claimed here.\n\n"
        "**Coverage agenda:** subsequent separate research should test close-only, cross-asset or legally released-target "
        "representations for markets without complete bars, after checking overlap with existing return/risk/relationship features. "
        "Carry, inventory and macro-vintage hypotheses require actual dated data; anonymous row IDs are not a calendar.\n\n"
        "**Primary research context:** Blanc, Chicheportiche & Bouchaud (2013), "
        "[overnight/intraday volatility decomposition](https://arxiv.org/abs/1309.5806), supports separating sessions "
        "in a stock-volatility problem—not guaranteed commodity-return alpha. Gorton, Hayashi & Rouwenhorst, "
        "[commodity futures fundamentals](https://www.nber.org/papers/w13249), motivates later inventory/basis work "
        "only when appropriate instruments and publication timing are available."
    )
    md(
        "## Save and stop\n\n"
        "Ctrl+S saves notebook outputs. Download the JSON report and this executed notebook. "
        "Keep the private checkpoint ZIP on your computer; do not attach models or predictions to chat or public GitHub. "
        "Download the self-contained HTML to view charts after stopping AWS. A local ZIP alone is not an off-disk backup. "
        "**Stop space in Studio—not Delete space. No automatic shutdown occurs.**"
    )
    code(
        "completed=support.finish(ROOT,report,figures)\nprint('RESULT:',completed['status'])\n"
        "print('DECISION:',completed['decision'])\nprint('REPORT:',support.report_path(ROOT))\n"
        "print('DASHBOARD:',completed['dashboard'])\n"
        "print('PRIVATE BACKUP:',completed.get('private_checkpoint_bundle',{}).get('path',completed.get('backup_warning','Not created')))\n"
        "print('Stop here. No other folds or model promotion. Save notebook, download evidence, then Stop space.')\n"
    )
    for i, c in enumerate(cells):
        c["id"] = f"session-ablation-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Commodity - manual (verified)",
                "language": "python",
                "name": "commodity-manual",
            },
            "language_info": {"name": "python", "version": "3.12.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def protocol_markdown() -> str:
    return """# Session-pair first-fold ablation protocol

## Completed evidence, not a new result

The submitted normalization report records 535 development dates. Current-market
control: 0.3097087232124053; normalized price: 0.3057333590409221;
volume confirmation: 0.3069106073242419; union: 0.29697041754207887.
All three pooled deltas are negative. The six continuation fits reused the three
first-fold fits and did not refit controls. Preserve this negative experiment.

Notebook 06 constructed 41 templates. Thirty-nine are numerical and two are
support indicators. Eleven numerical candidates had directionally consistent
training-half association; the other 28 are not proved useless, merely not
advanced by this screen. The selected names are frozen in the JSON declaration.
Only 89 of 424 target definitions have complete-leg OHLC: 87 pairs and 2 singles.
A narrow session representation cannot substitute for studying the other markets.

## Four predefined fits

1. Saved current_market + two complete-bar support indicators: new mask control.
2. Same base + indicators + six shortlisted intraday features.
3. Same base + indicators + five overnight/risk features (including risk share).
4. Same base + indicators + the exact eleven-feature union.

The original current_market model is replayed exactly, never refitted. Masks are
identical across new panels. Covariance-aware session features are computed for
the signed pair before normalization; all rolling reference moments stop at t-1.
Invalid observed values stay missing; unsupported targets use explicit structural
zeros and indicators. Supplied-row alignment does not establish synchronized
international closing times. Do not add rejected normalization features here.

## Information contract and budget

Training prefix: 0 through 1163. Model warmup: 252. Validation origins: 1169 through
1348. Label h is released at t+h+1; the existing five-date purge is unchanged.
Raw CSV numerical reads stop at 1348, with no later-fold or final labels loaded.
File hashing reads raw bytes for integrity, not outcome values for selection.
The existing cached feature panel is immutable; its first-fold prefix is used.
The 1164-row candidate prefix must replay exactly against the saved notebook-06
checkpoint. Current-market controls and parent hashes must verify before fitting.

Keep the project histogram settings fixed. Audit exact duplicates against the
current base and within candidates; report all training screening and numeric
admission. Any candidate selection is training-only. The previous screen covered
new features and normalization features, not the entire historical feature catalog.

One externally supervised worker has 240 seconds, including preflight, replay,
feature construction and the four fits. Fifteen-second progress heartbeats.
Independent sealed checkpoints survive later failure. Failed/incomplete runs stop
for diagnosis; completed results reopen without training. No automatic next folds,
GPU, dependency changes, GitHub/AWS writes, final model promotion or submission.
Notebook display, HTML export and ZIP backup are separate from the fitting worker.

## Review gate, not significance or promotion

A numerical panel earns review only if it beats BOTH the saved current_market and
the support-only control by at least 0.002. All exact replays and integrity gates
must pass. The test evaluates an already inspected development fold; do not call
it a fresh holdout. Seven paired contrasts use 10/20/40-date block resampling.
Conditional and simultaneous bounds are within-family; neither corrects all
prior adaptive experiments or screening. Record every negative panel.

If the mask-only panel wins, do not attribute that to numeric session information.
Inspect supported and other-target diagnostic effects separately: a pooled model
can change predictions for unsupported targets after refitting. The subgroup
metric uses at least three usable targets/date and is not the global metric.

## Domain basis and remaining research

Blanc, Chicheportiche & Bouchaud (2013), overnight/intraday volatility feedback:
https://arxiv.org/abs/1309.5806
This supports session decomposition for stock volatility; commodity return
predictability is an unproven transfer hypothesis, not an implication.

Gorton, Hayashi & Rouwenhorst, commodity futures fundamentals:
https://www.nber.org/papers/w13249
Inventories and basis inform futures risk-premium research. They are not
interchangeable with anonymous target row IDs or necessarily short-horizon alpha.
Actual instrument, timestamp and licensing contracts precede external-data joins.

After this test, consider close-only pair innovation, relative market structure
and horizon-correct released-target context for the other markets, after checking
existing return, volatility, graph and macro-relative coverage. Keep feature versus
learner/objective comparisons distinct. Research is not exhausted and a historical
leaderboard score on other dates is not comparable to this first-fold result.

## Artifacts and manual operations

Notebook: notebooks/07_session_feature_ablation.ipynb
Declaration: configs/manual_session_ablation.json
Code: scripts/commodity_session_ablation.py
Report/dashboard/ZIP: logs/manual_session_ablation/
New checkpoints: artifacts/session_pair_ablation/<child-lineage>/fold_0/
Save the executed notebook. Keep private models/predictions out of public GitHub.
Download the ZIP for an off-disk copy, then Stop space (never Delete space).
No space shutdown or remote GitHub write is performed by this helper.
"""


def install(root: Path = ROOT) -> dict:
    root = Path(root)
    old, u = previous(root)
    u.validate_source(root)
    read_reviews(root, u)
    body = Path(__file__).read_bytes()
    nb = (json.dumps(notebook_document(), indent=1) + "\n").encode()
    destinations = {
        u.safe_path(root, "docs/manual_session_ablation.md"): protocol_markdown().encode(),
        u.safe_path(root, "scripts/commodity_session_ablation.py"): body,
        u.safe_path(root, "notebooks/" + NOTEBOOK): nb,
        u.safe_path(root, "configs/manual_session_ablation.json"): (
            json.dumps(plan(), indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    # Check every conflict BEFORE writing anything; preserve executed notebooks.
    for p, b in destinations.items():
        if p.exists():
            if p.suffix == ".ipynb":
                existing = u.read_json(p)
                expected = notebook_document()
                if [(c["cell_type"], c.get("source")) for c in existing["cells"]] != [
                    (c["cell_type"], c["source"]) for c in expected["cells"]
                ]:
                    raise Stop("An existing different notebook was preserved: " + str(p))
            elif p.read_bytes() != b:
                raise Stop("Existing different file was preserved: " + str(p))
    for p, b in destinations.items():
        if not p.exists():
            u.write_new(p, b)
    emit("SESSION_ABLATION_NOTEBOOK_READY", new_fits=0)
    print(
        "RESULT: SESSION_ABLATION_NOTEBOOK_READY\nNOTEBOOK: "
        + str(root / "notebooks" / NOTEBOOK)
        + "\nKERNEL: Commodity - manual (verified)",
        flush=True,
    )
    return {"files": [str(p) for p in destinations], "new_fits": 0}


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
            "No automatic shutdown. Save the error/report, then Stop space. Do not repeat unchanged.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
