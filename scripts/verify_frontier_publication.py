"""Verify the employer-facing frontier publication without private competition data."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads((root / "reports/frontier_research_ledger.json").read_text())
    summary = json.loads((root / "reports/portfolio_summary.json").read_text())
    hashes = json.loads((root / "research/frontier/package_hashes.json").read_text())

    assert ledger["retained_panel"]["score"] == 0.2893649296119325
    assert ledger["official_kaggle_submission"]["status"] == "not_yet_scored"
    assert ledger["experiments"][-1]["status"] == "active_engineering_resume"
    assert summary["project_status"] == "portfolio_complete_frontier_active"
    assert summary["official_leaderboard_result"] is False
    assert hashes["active_full15_repaired_package_sha256"] == (
        "17e0733b797864d1e43d7ecec1428a3311a25ce45926df4a7605e33a64677201"
    )
    assert hashes["transformer_completed_source_package_sha256"] == (
        "cc284682cad84148abcf33dc2f66314265810d99c78b0e2273be8a7350856ece"
    )

    notebook = nbformat.read(root / "notebooks/28_frontier_research_ledger.ipynb", as_version=4)
    nbformat.validate(notebook)
    figures = 0
    static = 0
    last_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert isinstance(cell.execution_count, int)
        assert cell.execution_count > last_count
        last_count = cell.execution_count
        assert not any(o.output_type == "error" for o in cell.outputs)
        for output in cell.outputs:
            data = output.get("data", {})
            figures += int("application/vnd.plotly.v1+json" in data)
            static += int("image/svg+xml" in data)
    assert figures >= 3
    assert static >= 3

    required = [
        root / "src/commodity_prediction/competitive/feature_forecast.py",
        root / "src/commodity_prediction/competitive/third_place_reproduction.py",
        root / "tests/test_competitive_feature_forecast.py",
        root / "tests/test_third_place_reproduction.py",
        root / "docs/ROUND_20_PROTOCOL.md",
        root / "docs/ROUND_21A_PROTOCOL.md",
    ]
    assert all(path.is_file() for path in required)

    print(
        f"Verified frontier ledger, integrated competitive source, "
        f"{figures} Plotly figures and {static} SVG fallbacks"
    )


if __name__ == "__main__":
    main()
