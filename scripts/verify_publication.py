"""Verify public evidence without requiring restricted competition records in CI."""

import json
from pathlib import Path

import nbformat

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
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    figures = 0
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, path.name
                assert not any(o.output_type == "error" for o in cell.outputs), path.name
                figures += sum(
                    "application/vnd.plotly.v1+json" in o.get("data", {})
                    and "image/png" in o.get("data", {})
                    for o in cell.outputs
                )
    assert figures >= 8, "Interactive figures need static GitHub fallbacks"
    print(
        f"Verified source/result lineage, {len(paths)} executed notebooks, and {figures} Plotly/static figure pairs"
    )


if __name__ == "__main__":
    main()
