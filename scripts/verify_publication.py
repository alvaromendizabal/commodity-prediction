"""Verify public evidence without requiring restricted competition records in CI."""

import json
from pathlib import Path

import nbformat
import numpy as np

from commodity_prediction.runtime import digest, fingerprint


def main(*, require_aws_evidence: bool = True) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "reports/lineage.json").read_text())
    report = json.loads((root / "reports/research.json").read_text())
    if fingerprint(evidence) != report["lineage"]:
        raise ValueError("Inconsistent published research lineage")
    for name, expected in evidence["files"].items():
        if name.startswith("src/") and digest(root / name) != expected:
            raise ValueError(f"Published results have stale source: {name}")
    assert report["feature_gate"] == "open" and not report["holdout_evaluated"]
    assert report["experiments_completed"] == 30
    study = json.loads((root / "reports/feature_study.json").read_text())
    study_evidence = json.loads((root / "reports/feature_study_lineage.json").read_text())
    assert fingerprint(study_evidence) == study["lineage"]
    assert study["parent_lineage"] == report["lineage"]
    assert study_evidence["config"] == json.loads((root / "configs/feature_study.json").read_text())
    for name, expected in study_evidence["files"].items():
        assert digest(root / name) == expected, f"Stale study source: {name}"
    assert study["experiments_completed"] == len(study_evidence["experiments"]) * 3 == 69
    assert study["validation_dates"] == 535 and study["terminal_embargo_dates"] == 5
    assert (
        study["candidate_count"] == study["reused_candidate_count"] + study["new_candidate_count"]
    )
    assert study["candidate_count"] == sum(study["family_counts"].values())
    assert study["feature_gate"] == "open" and not study["holdout_evaluated"]
    final_evaluation = json.loads((root / "configs/final_evaluation.json").read_text())
    assert not final_evaluation["evaluated"]
    assert final_evaluation["final_test_start_date_id"] > 1708 + 5
    assert final_evaluation["final_test_dates"] == 247
    assert study["publication_review"]["final_evaluation_config_sha256"] == digest(
        root / "configs/final_evaluation.json"
    )
    assert "247 untouched" in study["limitations"][0]
    cloud = json.loads((root / "reports/aws_execution.json").read_text())
    domain = json.loads((root / "reports/domain_study.json").read_text())
    domain_evidence = json.loads((root / "reports/domain_lineage.json").read_text())
    assert fingerprint(domain_evidence) == domain["lineage"]
    assert domain["parent_lineage"] == study["lineage"]
    assert domain_evidence["config"] == json.loads((root / "configs/domain_study.json").read_text())
    assert domain_evidence["final_evaluation_sha256"] == digest(
        root / "configs/final_evaluation.json"
    )
    for name, expected in domain_evidence["files"].items():
        assert digest(root / name) == expected, f"Stale domain source: {name}"
    assert domain["new_fitted_models"] == len(domain_evidence["experiments"]) * 9 == 279
    assert domain["outer_evaluations"] == (len(domain_evidence["experiments"]) + 3) * 3 == 102
    assert domain["validation_dates"] == 535 and domain["untouched_final_test_dates"] == 247
    assert domain["feature_gate"] == "open" and not domain["holdout_evaluated"]
    assert domain["templates"] == sum(domain["inventory"]["family_templates"].values()) == 380
    assert domain["inventory"]["target_template_assignments"] == domain["templates"] * 424
    assert len(domain["comparisons"]) == domain["declared_comparison_count"] * 3
    for result in domain["results"]:
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
        counts = result["selection"]
        if counts is not None:
            assert (
                counts["candidate_templates"]
                == counts["retained_templates"] + counts["rejected_templates"]
            )
            assert counts["retained_templates"] <= 64
        for key in ["metrics", "raw_metrics"]:
            if key in result:
                daily = np.asarray(result[key]["daily_rank_correlations"])
                assert (
                    abs(daily.mean() / daily.std(ddof=0) - result[key]["official_metric"]) < 1e-12
                )
    for summary in domain["summaries"].values():
        daily = np.asarray(summary["daily_rank_correlations"])
        assert (
            len(daily) == 535
            and abs(daily.mean() / daily.std(ddof=0) - summary["official_metric"]) < 1e-12
        )
    attribution = json.loads((root / "reports/tree_attribution.json").read_text())
    attr_evidence = json.loads((root / "reports/tree_attribution_lineage.json").read_text())
    assert fingerprint(attr_evidence) == attribution["lineage"]
    assert attribution["parent_lineage"] == domain["lineage"]
    assert attr_evidence["config"] == json.loads(
        (root / "configs/tree_attribution.json").read_text()
    )
    for name, expected in attr_evidence["files"].items():
        assert digest(root / name) == expected, f"Stale tree-attribution source: {name}"
    assert attribution["new_fitted_models"] == len(attr_evidence["experiments"]) * 3 == 78
    assert attribution["joint_comparison_count"] == 177
    assert len(attribution["joint_comparisons"]) == 177 * 3
    assert attribution["feature_gate"] == "open" and not attribution["holdout_evaluated"]
    for result in attribution["results"]:
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
        daily = np.asarray(result["metrics"]["daily_rank_correlations"])
        assert abs(daily.mean() / daily.std(ddof=0) - result["metrics"]["official_metric"]) < 1e-12
        audit = result["selection"]
        assert (
            audit["candidate_templates"]
            == audit["retained_templates"] + audit["rejected_templates"]
        )
    for name, parent_name in [
        ("all_control", "raw__tree_all"),
        ("reference_control", "raw__tree_reference"),
        ("historical_mean", "historical_mean"),
    ]:
        assert (
            attribution["summaries"][name]["official_metric"]
            == domain["summaries"][parent_name]["official_metric"]
        )
    assert cloud["lineage"] == attribution["lineage"]
    assert cloud["model_checkpoints_replayed"] == 435
    assert cloud["domain_model_checkpoints_replayed"] == 288
    assert cloud["attribution_model_checkpoints_replayed"] == 78
    assert cloud["checkpoint_count"] == 480
    assert cloud["maximum_prediction_replay_error"] <= 1e-12
    robustness = json.loads((root / "reports/domain_robustness.json").read_text())
    robust_evidence = json.loads((root / "reports/domain_robustness_lineage.json").read_text())
    assert fingerprint(robust_evidence) == robustness["lineage"]
    assert robustness["parent_lineage"] == attribution["lineage"]
    assert robust_evidence["config"] == json.loads(
        (root / "configs/domain_robustness.json").read_text()
    )
    for name, expected in robust_evidence["files"].items():
        assert digest(root / name) == expected, f"Stale robustness source: {name}"
    fitting = robust_evidence["fitting_evidence"]
    assert fingerprint(fitting) == robustness["fitting_lineage"]
    for name, expected in fitting["files"].items():
        assert digest(root / name) == expected, f"Stale robustness fitting source: {name}"
    assert robustness["new_fitted_models"] == robustness["model_checkpoints_replayed"] == 36
    assert robustness["checkpoint_count"] == 37
    assert robustness["maximum_prediction_replay_error"] <= 1e-12
    assert robustness["joint_comparison_count"] == 204
    assert len(robustness["joint_comparisons"]) == 204 * 3
    assert robustness["new_distinct_interaction_templates"] == 68
    assert robustness["feature_gate"] == "open" and not robustness["holdout_evaluated"]
    assert robustness["validation_dates"] == 535
    if require_aws_evidence:
        latest = json.loads((root / "reports/aws_feature_research.json").read_text())
        assert latest["status"] == "completed"
        assert latest["lineage"] == robustness["lineage"]
        assert latest["stage_manifests_verified"] == 517
        assert latest["model_replays_total"] == 471
        assert latest["preserved_model_replays"] == cloud["model_checkpoints_replayed"]
        assert latest["latest_model_replays"] == robustness["model_checkpoints_replayed"]
        assert latest["maximum_prediction_replay_error"] <= 1e-12
        assert latest["preserved_replay_report_sha256"] == digest(
            root / "reports/aws_execution.json"
        )
        assert latest["latest_report_sha256"] == digest(root / "reports/domain_robustness.json")
        assert latest["notebooks_executed"] == 3
        assert latest["plotly_static_figure_pairs"] == 25
        assert latest["feature_gate"] == "open" and not latest["holdout_evaluated"]
    for result in robustness["results"]:
        counts = result["selection"]
        assert (
            counts["candidate_templates"]
            == counts["retained_templates"] + counts["rejected_templates"]
        )
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
    for result in robustness["summaries"].values():
        daily = np.asarray(result["daily_rank_correlations"])
        assert len(daily) == 535
        assert abs(daily.mean() / daily.std(ddof=0) - result["official_metric"]) < 1e-12
    for result in study["results"]:
        values = np.asarray(result["daily_rank_correlations"])
        assert abs(values.mean() / values.std(ddof=0) - result["official_metric"]) < 1e-12
        counts = result["screening"]
        if counts:
            assert (
                counts["candidate_output_assignments"]
                == counts["retained_output_assignments"] + counts["rejected_output_assignments"]
            )
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    figures = 0
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["study_lineage"] == study["lineage"], path.name
        assert notebook.metadata["domain_lineage"] == domain["lineage"], path.name
        assert notebook.metadata["attribution_lineage"] == attribution["lineage"], path.name
        assert notebook.metadata["robustness_lineage"] == robustness["lineage"], path.name
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, path.name
                assert not any(o.output_type == "error" for o in cell.outputs), path.name
                figures += sum(
                    "application/vnd.plotly.v1+json" in o.get("data", {})
                    and "image/png" in o.get("data", {})
                    for o in cell.outputs
                )
    assert figures >= 25, "Interactive figures need static GitHub fallbacks"
    print(
        f"Verified source/result lineage, {len(paths)} executed notebooks, and {figures} Plotly/static figure pairs"
    )


if __name__ == "__main__":
    main()
