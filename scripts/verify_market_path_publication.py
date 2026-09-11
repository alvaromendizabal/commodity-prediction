"""Verify public market-path source, aggregate results, and canonical notebook evidence."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np

EXPECTED_LINEAGE = "04544d4b1a1d63487a24e88749d03d3259326c63902f2d0c74214e711147d605"
EXPECTED_SCORES = {
    "admitted_tail": 0.3053356475284676,
    "current_market": 0.3097087232124053,
    "price_path": 0.30030837044600683,
    "activity_path": 0.30791556736804493,
    "joint_path": 0.29925174187198667,
}
EXPECTED_COUNTS = {
    "current_market": (87, 82, 5),
    "price_path": (105, 102, 3),
    "activity_path": (87, 86, 1),
    "joint_path": (123, 118, 5),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    publication = json.loads((root / "reports/market_path_publication.json").read_text())
    execution = json.loads((root / "reports/market_path_execution.json").read_text())
    probe = json.loads((root / "reports/market_path_probe_execution.json").read_text())
    workspace = json.loads((root / "reports/workspace_readiness.json").read_text())
    config = json.loads((root / "configs/market_path_study.json").read_text())
    final = json.loads((root / "configs/final_evaluation.json").read_text())

    assert publication["status"] == execution["status"] == "completed"
    assert publication["lineage"] == execution["lineage"] == probe["lineage"] == EXPECTED_LINEAGE
    assert publication["feature_gate"] == execution["feature_gate"] == "open"
    assert not publication["holdout_evaluated"] and not execution["holdout_evaluated"]
    assert not final["evaluated"] and final["final_test_start_date_id"] == 1714
    assert publication["validation_dates"] == execution["validation_dates"] == 535
    assert config["max_new_fits"] == 12 and config["probe_fits"] == 4
    assert config["max_run_seconds"] == 300 and config["frozen_control"] == "admitted_tail"

    assert publication["new_fitted_models"] == execution["full_study"]["new_fitted_models"] == 12
    assert (
        publication["fits_this_invocation"] == execution["full_study"]["fits_this_invocation"] == 8
    )
    assert publication["probe_fits_reused"] == execution["full_study"]["probe_fits_reused"] == 4
    assert publication["model_checkpoints_replayed"] == 12
    assert execution["full_study"]["checkpoint_count"] == 13
    assert publication["maximum_prediction_replay_error"] == 0.0
    assert execution["full_study"]["maximum_prediction_replay_error"] == 0.0
    assert probe["fits_this_invocation"] == 4 and probe["maximum_prediction_replay_error"] == 0.0
    assert probe["continue_allowed"] is True

    for name, expected in EXPECTED_SCORES.items():
        summary = publication["scores"][name]
        assert np.isclose(summary["official_metric"], expected, rtol=0, atol=1e-15)
        assert len(summary["fold_scores"]) == 3
        assert all(np.isfinite(summary["fold_scores"]))

    counts = execution["selection_counts_by_fold"]
    for name, expected in EXPECTED_COUNTS.items():
        for fold in range(3):
            row = counts[name][f"fold_{fold}"]
            assert (row["candidate"], row["retained"], row["rejected"]) == expected
            assert row["candidate"] == row["retained"] + row["rejected"]
            assert row["reasons"] == ({"duplicate": expected[2]} if expected[2] else {})

    comparison = publication["predeclared_20_date_comparisons"]["current_market_vs_admitted_tail"]
    expected_delta = EXPECTED_SCORES["current_market"] - EXPECTED_SCORES["admitted_tail"]
    assert np.isclose(comparison["delta"], expected_delta, rtol=0, atol=1e-15)
    lo, hi = comparison["conditional_95_interval"]
    slo, shi = comparison["simultaneous_95_interval"]
    assert lo < 0 < hi and slo < 0 < shi

    assert workspace["status"] == "WORKSPACE_READY"
    assert workspace["checkpoint_manifests_verified"] == 589
    assert workspace["raw_files_verified"] == 8
    assert workspace["new_training_fits"] == 0
    assert execution["new_historical_fits"] == 0
    assert execution["notebook_executions"] == 0
    assert execution["final_test_evaluations"] == 0
    assert execution["compute_cleanup"]["persistent_space_preserved"] is True

    readme = (root / "README.md").read_text()
    research = (root / "docs/market-path-research.md").read_text()
    assert "0.309709" in readme and "market-path" in readme.lower()
    assert "0.309709" in research and "Measured results" in research

    notebook = nbformat.read(root / "notebooks/02_feature_research.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata["market_path_lineage"] == EXPECTED_LINEAGE
    evidence_cells = [c for c in notebook.cells if c.metadata.get("market_path_evidence") is True]
    assert len(evidence_cells) == 1 and "0.309709" in evidence_cells[0].source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)

    print(
        "Verified bounded market-path evidence: 12 fits, 13 sealed stages, "
        "exact replay, 535 development dates, and canonical notebook publication"
    )


if __name__ == "__main__":
    main()
