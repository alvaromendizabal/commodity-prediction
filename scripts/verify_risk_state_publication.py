"""Verify public risk-state source, metrics, screening and executed notebook lineage."""

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np

from commodity_prediction.runtime import digest, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Validate the draft research checkpoint; executed notebook publication remains a separate gate.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/risk_state_study.json").read_text())
    evidence = json.loads((root / "reports/risk_state_lineage.json").read_text())
    assert report["lineage"] == fingerprint(evidence)
    assert evidence["config"] == json.loads((root / "configs/risk_state_study.json").read_text())
    for name, expected in evidence["files"].items():
        assert digest(root / name) == expected, name
    parent = json.loads((root / "reports/compact_study.json").read_text())
    assert report["parent_lineage"] == parent["lineage"]
    assert report["feature_gate"] == "open" and not report["holdout_evaluated"]
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 24
    assert report["checkpoint_count"] == 25
    assert report["new_distinct_templates"] == 34
    assert report["new_template_target_assignments"] == 34 * 424
    assert report["maximum_prediction_replay_error"] <= 1e-12
    assert report["joint_comparison_count"] == 255
    assert (
        len(report["results"]) == len({(r["variant"], r["fold"]) for r in report["results"]}) == 24
    )
    for result in report["results"]:
        counts = result["selection"]
        assert (
            counts["candidate_templates"]
            == counts["retained_templates"] + counts["rejected_templates"]
        )
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
    for summary in report["summaries"].values():
        daily = np.asarray(summary["daily_rank_correlations"])
        assert len(daily) == 535 and np.isfinite(daily).all()
        assert abs(daily.mean() / daily.std(ddof=0) - summary["official_metric"]) < 1e-12
    if args.aggregate_only:
        print(
            "Verified 24 risk-state fits and 255 matched comparisons; new notebook execution is a separate pending publication gate"
        )
        return
    figures = 0
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        assert notebook.metadata["risk_state_lineage"] == report["lineage"], path.name
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, path.name
                assert not any(o.output_type == "error" for o in cell.outputs), path.name
                figures += sum(
                    "application/vnd.plotly.v1+json" in o.get("data", {})
                    and "image/png" in o.get("data", {})
                    for o in cell.outputs
                )
    assert figures >= 31
    print(f"Verified 24 risk-state fits, 255 matched comparisons and {figures} Plotly/static pairs")


if __name__ == "__main__":
    main()
