#!/usr/bin/env python3
"""Verify the public commodity-forecasting closeout artifacts.

This intentionally uses only the Python standard library so reviewers can validate
public evidence before installing the full ML environment.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "portfolio_summary.json"
LEDGER = ROOT / "reports" / "final_research_ledger.json"
NOTEBOOK = ROOT / "notebooks" / "28_final_research_closeout.ipynb"


def load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def near(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    summary = load(SUMMARY)
    ledger = load(LEDGER)
    notebook = load(NOTEBOOK)

    assert summary["project_status"] == "complete"
    assert summary["research_cycle_status"] == "completed_and_frozen"
    assert summary["target_contract"] == {"targets": 424, "forecast_horizons": 4}

    tr = summary["temporal_replication"]
    assert near(tr["candidate"] - tr["incumbent"], tr["delta"])
    assert tr["dates"] == 355 and tr["targets"] == 424

    hist = summary["final_historical_assessment"]
    assert near(hist["incumbent"] - hist["historical_mean_control"], hist["delta"])
    assert hist["dates"] == 247 and hist["targets"] == 424

    replay = summary["later_research"]["post_competition_exact_replay"]
    assert replay["scored_test_days"] == 134
    assert replay["terminal_segment_days"] == 73
    assert replay["training_selection_only"] is True
    assert replay["test_labels_used_for_selection"] is False

    families = ledger["families"]
    assert ledger["status"] == "complete"
    assert len(families) >= 20
    names = {row["family"] for row in families}
    required = {
        "mixed_horizon_target_routed_trees",
        "regularized_covariance",
        "feature_token_transformer",
        "exact_delayed_label_replay",
    }
    assert required <= names

    assert notebook["nbformat"] == 4
    cell_ids = {cell.get("id") for cell in notebook.get("cells", [])}
    assert {"closeout-title", "research-scale", "validation-evolution", "final-takeaway"} <= cell_ids

    print("PORTFOLIO_VERIFY=PASS")
    print(f"targets={summary['target_contract']['targets']}")
    print(f"forecast_horizons={summary['target_contract']['forecast_horizons']}")
    print(f"research_families={len(families)}")
    print(f"final_replay_days={replay['scored_test_days']}")
    print(f"terminal_segment_days={replay['terminal_segment_days']}")


if __name__ == "__main__":
    main()
