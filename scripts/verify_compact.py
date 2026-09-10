"""Check complete public compact evidence without restricted training records."""

import json
from pathlib import Path

import numpy as np

from commodity_prediction.runtime import digest, fingerprint


def verify(root: Path, parent: dict) -> dict | None:
    path = root / "reports/compact_study.json"
    if not path.exists():
        return None  # Tested source branch may precede its first AWS execution.
    report = json.loads(path.read_text())
    evidence = json.loads((root / "reports/compact_lineage.json").read_text())
    assert fingerprint(evidence) == report["lineage"]
    assert evidence["config"] == json.loads((root / "configs/compact_study.json").read_text())
    assert report["parent_lineage"] == parent["lineage"]
    assert evidence["parent_lineage"] == parent["lineage"]
    for name, expected in evidence["files"].items():
        assert digest(root / name) == expected, f"Stale compact source: {name}"
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 36
    assert len(evidence["experiments"]) * 3 == len(report["results"]) == 36
    assert report["checkpoint_count"] == 37
    assert report["maximum_prediction_replay_error"] == 0
    assert report["new_distinct_templates"] == 42
    assert report["new_template_target_assignments"] == 42 * 424
    assert report["reused_product_templates"] == 12
    assert report["feature_gate"] == "open" and not report["holdout_evaluated"]
    assert report["validation_dates"] == 535
    assert report["joint_comparison_count"] == 204 + len(evidence["comparisons"]) == 233
    assert len(report["joint_comparisons"]) == 233 * 3
    for result in report["results"]:
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
        audit = result["selection"]
        assert (
            audit["candidate_templates"]
            == audit["retained_templates"] + audit["rejected_templates"]
        )
        if result["variant"].startswith("admitted_"):
            assert set(audit["rejection_reasons"]) <= {"constant", "missingness", "duplicate"}
        daily = np.asarray(result["metrics"]["daily_rank_correlations"])
        assert abs(daily.mean() / daily.std(ddof=0) - result["metrics"]["official_metric"]) < 1e-12
    for summary in report["summaries"].values():
        daily = np.asarray(summary["daily_rank_correlations"])
        assert len(daily) == 535
        assert abs(daily.mean() / daily.std(ddof=0) - summary["official_metric"]) < 1e-12
    for name in ["historical_mean", "compact_control", "all_control"]:
        assert report["summaries"][name] == parent["summaries"][name]
    assert report["summaries"]["joint_control"] == parent["summaries"]["compact_risk_freshness"]
    return report
