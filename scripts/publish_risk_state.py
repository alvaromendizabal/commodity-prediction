"""Bounded notebook-only publication of already fitted risk-state checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from feature_publication import supervise
from restore_checkpoint import restore
from threadpoolctl import threadpool_limits

from commodity_prediction.cloud import client
from commodity_prediction.data import load_data
from commodity_prediction.domain.risk_state.features import build_panel, experiment_plan
from commodity_prediction.domain.risk_state.run import run_study, study_lineage
from commodity_prediction.domain.run import load_panel
from commodity_prediction.runtime import RunLog, atomic_json, digest, verify_checkpoint


def restore_and_replay(root: Path) -> None:
    """Require the complete sealed run, then independently replay; never fit."""
    settings = json.loads((root / "configs/bootstrap.json").read_text())[
        "risk_state_study_snapshot"
    ]
    s3, aws = client(root)
    bundle = root / "data/risk-state-checkpoint.tar.gz"
    if not bundle.exists() or digest(bundle) != settings["sha256"]:
        s3.download_file(aws["bucket"], settings["key"], str(bundle))
    restore(root, bundle, settings["sha256"])
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    manifests = sorted(directory.rglob("manifest.json"))
    if len(manifests) != 25 or not verify_checkpoint(directory / "summary", lineage):
        raise ValueError("Publication requires all 25 sealed stages; training is prohibited")
    for manifest in manifests:
        if not verify_checkpoint(manifest.parent, lineage):
            raise ValueError("Incomplete sealed risk-state run")
    report = run_study(root)  # Complete-run branch verifies and returns without fitting.
    original = load_panel(root / "artifacts" / evidence["feature_lineage"] / "features")
    x, y, pairs = load_data(root)
    del x, y  # No label-based evaluation or feature selection in publication.
    log = RunLog(root / "logs/risk_state_cloud_replay.jsonl")
    maximum = 0.0
    completed = 0
    with threadpool_limits(limits=4), log.stage("independent_saved_model_replay"):
        for variant in experiment_plan():
            panel = build_panel(original, pairs, variant)
            for result in (r for r in report["results"] if r["variant"] == variant.name):
                stage = directory / f"fold_{result['fold']}" / variant.name
                saved = pd.read_parquet(stage / "predictions.parquet")
                actual = joblib.load(stage / "model.joblib").predict(
                    panel, result["validation_start"], result["validation_stop"]
                )
                # Parquet preserves axis labels that the estimator does not set.
                # The ordered date/target values must still match exactly.
                pd.testing.assert_index_equal(saved.index, actual.index, check_names=False)
                pd.testing.assert_index_equal(saved.columns, actual.columns, check_names=False)
                error = float(np.max(np.abs(saved.to_numpy() - actual.to_numpy())))
                if not np.isfinite(error) or error > 1e-12:
                    raise ValueError("Cloud model replay differs")
                maximum = max(maximum, error)
                completed += 1
                log.event("model_replayed", completed=completed, total=24, error=error)
    all_manifests = list((root / "artifacts").glob("*/**/manifest.json"))
    for manifest in all_manifests:
        expected = manifest.relative_to(root / "artifacts").parts[0]
        if not verify_checkpoint(manifest.parent, expected):
            raise ValueError("Incomplete historical checkpoint")
    if completed != 24 or len(all_manifests) != 579:
        raise ValueError("Unexpected project checkpoint inventory")
    atomic_json(
        root / "reports/risk_state_cloud_preflight.json",
        {
            "verified_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": settings["sha256"],
            "archive_key": settings["key"],
            "lineage": lineage,
            "independent_cloud_model_replays": completed,
            "maximum_prediction_replay_error": maximum,
            "stage_manifests_verified": len(all_manifests),
            "new_training_fits": 0,
            "holdout_evaluated": False,
        },
    )


def verify_and_publish(root: Path) -> None:
    for script in ["verify_publication.py", "verify_risk_state_publication.py"]:
        subprocess.run([sys.executable, str(root / "scripts" / script)], check=True, cwd=root)
    evidence = json.loads((root / "reports/risk_state_cloud_preflight.json").read_text())
    notebooks = [
        nbformat.read(p, as_version=4) for p in sorted((root / "notebooks").glob("*.ipynb"))
    ]
    figures = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) and "image/png" in o.get("data", {})
        for nb in notebooks
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.outputs
    )
    evidence.update(
        status="completed",
        verified_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        notebooks_executed=len(notebooks),
        plotly_static_figure_pairs=figures,
        feature_gate="open",
        publication_budget_seconds=900,
    )
    atomic_json(root / "reports/aws_risk_state_publication.json", evidence)
    bundle = root / "artifacts/risk-state-notebooks.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for directory in ["notebooks", "reports", "logs", "configs"]:
            archive.add(root / directory, arcname=directory)
    s3, aws = client(root)
    key = "bootstrap/risk-state-notebooks.tar.gz"
    s3.upload_file(
        str(bundle),
        aws["bucket"],
        key,
        ExtraArgs={"Metadata": {"sha256": digest(bundle)}, "ServerSideEncryption": "AES256"},
    )
    head = s3.head_object(Bucket=aws["bucket"], Key=key)
    if head["ContentLength"] != bundle.stat().st_size or head["Metadata"]["sha256"] != digest(
        bundle
    ):
        raise ValueError("Notebook publication archive integrity differs")
    evidence.update(publication_archive_key=key, publication_archive_sha256=digest(bundle))
    s3.put_object(
        Bucket=aws["bucket"],
        Key="bootstrap/risk-state-verification.json",
        Body=json.dumps(evidence).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    print(json.dumps(evidence), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["restore", "publish"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage == "restore":
        restore_and_replay(root)
        return 0
    if args.stage == "publish":
        verify_and_publish(root)
        return 0
    s3, aws = client(root)

    def publish(state: dict) -> None:
        log = root / "logs/risk-state-publication.log"
        if log.exists():
            s3.upload_file(str(log), aws["bucket"], "bootstrap/risk-state-publication.log")
        s3.put_object(
            Bucket=aws["bucket"],
            Key="bootstrap/risk-state-publication-status.json",
            Body=json.dumps(state).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    python = sys.executable
    commands = [
        ("restore_and_replay", [python, "scripts/publish_risk_state.py", "--stage", "restore"]),
        ("rendering_dependencies", ["bash", "scripts/prepare_rendering.sh"]),
        ("notebook_generation", [python, "scripts/make_notebooks.py"]),
        ("notebook_execution", [python, "scripts/execute_notebooks.py"]),
        (
            "verification_and_publication",
            [python, "scripts/publish_risk_state.py", "--stage", "publish"],
        ),
    ]
    return supervise(
        root,
        commands,
        publish,
        max_seconds=900,
        status_path="logs/risk-state-publication-status.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
