"""Validate source lineage, exact matched metrics and the executed context chapter."""

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np

from commodity_prediction.runtime import digest, fingerprint


def verify(root: Path, aggregate_only: bool = False) -> None:
    report = json.loads((root / "reports/released_context_study.json").read_text())
    evidence = json.loads((root / "reports/released_context_lineage.json").read_text())
    parent = json.loads((root / "reports/risk_state_study.json").read_text())
    assert report["lineage"] == fingerprint(evidence)
    assert evidence["config"] == json.loads(
        (root / "configs/released_context_study.json").read_text()
    )
    assert report["parent_lineage"] == parent["lineage"]
    for name, expected in evidence["files"].items():
        assert digest(root / name) == expected, name
    assert report["feature_gate"] == "open" and not report["holdout_evaluated"]
    assert report["new_fitted_models"] == report["model_checkpoints_replayed"] == 9
    assert report["checkpoint_count"] == 10 and report["candidate_templates_added"] == 24
    assert report["added_template_target_assignments"] == 24 * 424
    assert report["maximum_prediction_replay_error"] <= 1e-12
    assert report["joint_comparison_count"] == parent["joint_comparison_count"] + 8 == 263
    assert (
        len({(r["variant"], r["fold"]) for r in report["results"]}) == len(report["results"]) == 9
    )
    for result in report["results"]:
        counts = result["selection"]
        assert (
            counts["candidate_templates"]
            == counts["retained_templates"] + counts["rejected_templates"]
        )
        assert result["train_stop"] - 1 + 5 < result["validation_start"]
        assert result["validation_stop"] - 1 + 5 < 1709
    for name, summary in report["summaries"].items():
        daily = np.asarray(summary["daily_rank_correlations"])
        assert len(daily) == 535 and np.isfinite(daily).all()
        assert abs(daily.mean() / daily.std(ddof=0) - summary["official_metric"]) < 1e-12
        if name in parent["summaries"]:
            assert summary == parent["summaries"][name]
    for contrast in report["comparisons"]:
        expected = (
            report["summaries"][contrast["variant"]]["official_metric"]
            - report["summaries"][contrast["reference"]]["official_metric"]
        )
        assert abs(contrast["delta"] - expected) < 1e-12
    if aggregate_only:
        print("Verified 9 context fits, 24 candidate templates and 263 matched comparisons")
        return
    nb = nbformat.read(root / "notebooks/02_feature_research.ipynb", as_version=4)
    assert nb.metadata["released_context_lineage"] == report["lineage"]
    chapter = [c for c in nb.cells if "context = checked_released_context(root)" in c.source]
    assert len(chapter) == 1
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(o.output_type == "error" for o in cell.outputs)
    for name in [
        "released_context_scores",
        "released_context_folds",
        "released_context_uncertainty",
    ]:
        assert (root / "reports/figures" / (name + ".png")).stat().st_size > 1000
    figures = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) and "image/png" in o.get("data", {})
        for p in (root / "notebooks").glob("*.ipynb")
        for c in nbformat.read(p, as_version=4).cells
        if c.cell_type == "code"
        for o in c.outputs
    )
    assert figures >= 34
    print(
        f"Verified executed context chapter, 9 exact model replays and {figures} Plotly/static pairs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    verify(Path(__file__).resolve().parents[1], parser.parse_args().aggregate_only)


if __name__ == "__main__":
    main()
