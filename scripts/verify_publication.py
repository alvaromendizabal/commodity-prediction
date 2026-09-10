"""Verify public evidence without requiring restricted competition records in CI."""

import json
from pathlib import Path

import nbformat
import numpy as np

from commodity_prediction.runtime import digest, fingerprint


def main() -> None:
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
    cloud = json.loads((root / "reports/aws_execution.json").read_text())
    assert cloud["lineage"] == study["lineage"]
    assert cloud["model_checkpoints_replayed"] == 69
    assert cloud["maximum_prediction_replay_error"] <= 1e-12
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
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, path.name
                assert not any(o.output_type == "error" for o in cell.outputs), path.name
                figures += sum(
                    "application/vnd.plotly.v1+json" in o.get("data", {})
                    and "image/png" in o.get("data", {})
                    for o in cell.outputs
                )
    assert figures >= 11, "Interactive figures need static GitHub fallbacks"
    print(
        f"Verified source/result lineage, {len(paths)} executed notebooks, and {figures} Plotly/static figure pairs"
    )


if __name__ == "__main__":
    main()
