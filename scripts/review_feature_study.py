"""Publish a traceable boundary correction without changing completed model lineage."""

import json
from pathlib import Path

from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def review(root: Path) -> None:
    current = json.loads((root / "reports/feature_study.json").read_text())
    stage = root / "artifacts" / current["lineage"]
    if not verify_checkpoint(stage, current["lineage"]):
        raise ValueError("Missing verified study summary")
    report = json.loads((stage / "summary.json").read_text())
    restriction = root / "configs/final_evaluation.json"
    final = json.loads(restriction.read_text())
    if final["evaluated"] or final["final_test_dates"] != 247:
        raise ValueError("Unexpected final-test restriction")
    report["limitations"][0] = (
        "All scores are historical development estimates. The original reservation has "
        "a permanent five-origin boundary buffer (1709–1713), leaving 247 untouched "
        "final-test origins (1714–1960). No final-test evaluation has occurred."
    )
    report["publication_review"] = {
        "source_summary_sha256": digest(stage / "summary.json"),
        "final_evaluation_config_sha256": digest(restriction),
        "correction": "Supersedes the frozen summary's legacy claim of 252 untouched dates; numerical results are unchanged.",
    }
    atomic_json(root / "reports/feature_study.json", report)


if __name__ == "__main__":
    review(Path(__file__).resolve().parents[1])
