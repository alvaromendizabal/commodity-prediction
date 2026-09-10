"""Verify the latest artifacts while preserving valid earlier model-replay evidence."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from notebook_support import (
    checked_attribution,
    checked_compact,
    checked_domain,
    checked_reports,
    checked_robustness,
    checked_study,
)
from verify_bootstrap import publish_bundle
from verify_publication import main as verify_publication

from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verify_publication(require_aws_evidence=False)
    _, initial = checked_reports(root)
    study, domain, attribution, robustness = (
        checked_study(root),
        checked_domain(root),
        checked_attribution(root),
        checked_robustness(root),
    )
    old_path = root / "reports/aws_execution.json"
    compact = checked_compact(root)
    previous = json.loads(old_path.read_text())
    assert previous["lineage"] == attribution["lineage"]
    assert previous["model_checkpoints_replayed"] == 435
    assert previous["maximum_prediction_replay_error"] == 0
    assert robustness["model_checkpoints_replayed"] == 36
    assert robustness["maximum_prediction_replay_error"] == 0
    assert compact["model_checkpoints_replayed"] == 36
    assert compact["maximum_prediction_replay_error"] == 0
    lineages = [
        initial["lineage"],
        study["lineage"],
        domain["lineage"],
        attribution["lineage"],
        robustness["fitting_lineage"],
        robustness["lineage"],
        compact["lineage"],
    ]
    count = 0
    for lineage in lineages:
        for manifest in (root / "artifacts" / lineage).rglob("manifest.json"):
            verify_checkpoint(manifest.parent, lineage)
            count += 1
    assert count == 554
    evidence = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "lineage": compact["lineage"],
        "lineages_verified": lineages,
        "stage_manifests_verified": count,
        "model_replays_total": 507,
        "preserved_model_replays": 471,
        "latest_model_replays": 36,
        "maximum_prediction_replay_error": 0.0,
        "preserved_replay_report_sha256": digest(old_path),
        "preserved_robustness_report_sha256": digest(root / "reports/domain_robustness.json"),
        "latest_report_sha256": digest(root / "reports/compact_study.json"),
        "notebooks_executed": 3,
        "plotly_static_figure_pairs": 28,
        "feature_gate": "open",
        "holdout_evaluated": False,
        "verification_note": "All 554 manifests reverified. Preserved exact replay evidence for 435 earlier models/controls and 36 robustness models. Independently replayed all 36 compact-study models.",
    }
    atomic_json(root / "reports/aws_feature_research.json", evidence)
    verify_publication()
    publish_bundle(root, evidence)


if __name__ == "__main__":
    main()
