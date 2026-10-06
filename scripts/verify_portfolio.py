#!/usr/bin/env python3
"""Check public aggregate consistency, not private prediction-level performance."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object) -> float:
    require(type(value) in (float, int), "metric must be numeric")
    number = float(value)  # type: ignore[arg-type]
    require(math.isfinite(number), "metric must be finite")
    return number


def verify(root: Path = ROOT) -> dict[str, int]:
    summary = json.loads((root / "reports/portfolio_summary.json").read_text())
    ledger = json.loads((root / "reports/final_research_ledger.json").read_text())
    notebook = json.loads((root / "notebooks/28_final_research_closeout.ipynb").read_text())
    require(summary["project_status"] == "complete", "project must be complete")
    require(summary["research_cycle_status"] == "completed_and_frozen", "record must be frozen")
    require(
        summary["target_contract"] == {"targets": 424, "forecast_horizons": 4},
        "target contract changed",
    )
    for key in (
        "latest_implementation_published",
        "latest_private_implementation_published",
        "existing_public_history_removed",
        "raw_competition_data_published",
        "production_deployment_claimed",
    ):
        require(summary[key] is False, f"publication boundary changed: {key}")

    comparisons = (
        ("temporal_replication", "candidate", "incumbent", 355),
        ("final_historical_assessment", "incumbent", "historical_mean_control", 247),
    )
    for name, candidate_key, control_key, dates in comparisons:
        row = summary[name]
        delta = finite(row[candidate_key]) - finite(row[control_key])
        require(math.isclose(delta, finite(row["delta"]), rel_tol=0, abs_tol=1e-12), name)
        require(row["dates"] == dates and row["targets"] == 424, f"population changed: {name}")
        low, high = map(finite, row["conditional_paired_95_interval"])
        require(low < 0 < high, f"recorded inconclusive interval changed: {name}")
    require(
        summary["temporal_replication"]["decision"] == "positive_estimate_not_promoted",
        "temporal decision changed",
    )
    final = summary["final_historical_assessment"]
    require(final["evaluated"] is True and final["excluded_dates"] == 0, "assessment coverage")
    require(
        final["decision"] == "recorded_without_post_assessment_reselection",
        "assessment decision changed",
    )

    replay = summary["later_research"]["post_competition_exact_replay"]
    require(replay["scored_test_days"] == 134, "replay coverage changed")
    require(replay["terminal_segment_days"] == 73, "terminal coverage changed")
    require(replay["training_selection_only"] is True, "selection must remain training-only")
    require(replay["test_labels_used_for_selection"] is False, "test-label selection is invalid")

    families = ledger["families"]
    require(ledger["status"] == "complete", "ledger must be complete")
    names = {row["family"] for row in families}
    require(len(families) >= 20 and len(names) == len(families), "missing or duplicate families")
    required = {
        "mixed_horizon_target_routed_trees",
        "regularized_covariance",
        "feature_token_transformer",
        "exact_delayed_label_replay",
    }
    require(required <= names, "required research families missing")
    allowed_outcomes = {
        "retained_reference",
        "rejected",
        "promoted_component",
        "closeout_evaluation",
    }
    require(all(row["outcome"] in allowed_outcomes for row in families), "unknown ledger outcome")

    require(notebook["nbformat"] == 4, "notebook format changed")
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    require(
        {"closeout-title", "research-scale", "validation-evolution", "final-takeaway"} <= cell_ids,
        "required notebook sections missing",
    )
    code_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip()
    ]
    require(len(code_cells) == 5, "closeout code-cell count changed")
    for cell in code_cells:
        count = cell.get("execution_count")
        require(type(count) is int and count > 0, "closeout cell is not executed")
        require(bool(cell.get("outputs")), "closeout cell has no saved output")
        require(
            all(output.get("output_type") != "error" for output in cell["outputs"]),
            "closeout notebook contains an error",
        )
    sources = "\n".join("".join(cell["source"]) for cell in code_cells)
    require("portfolio_summary.json" in sources, "summary input missing from notebook")
    require("final_research_ledger.json" in sources, "ledger input missing from notebook")
    return {
        "targets": 424,
        "forecast_horizons": 4,
        "research_families": len(families),
        "final_replay_days": replay["scored_test_days"],
        "terminal_segment_days": replay["terminal_segment_days"],
        "saved_executed_cells": len(code_cells),
    }


def main() -> None:
    checks = verify()
    print("PORTFOLIO_VERIFY=PASS")
    print("scope=public aggregate consistency; private model scores are not reproduced")
    for key, value in checks.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
