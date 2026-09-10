"""Replay completed context models and publish one fresh-kernel notebook; never fit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from feature_publication import supervise
from nbclient import NotebookClient
from released_context_notebook import cells
from restore_checkpoint import restore
from threadpoolctl import threadpool_limits
from traitlets.config import Config
from verify_released_context_publication import verify

from commodity_prediction.cloud import client
from commodity_prediction.data import load_data
from commodity_prediction.domain.released_context.features import build_panel, experiment_plan
from commodity_prediction.domain.released_context.run import run_study, study_lineage
from commodity_prediction.domain.run import load_panel
from commodity_prediction.runtime import RunLog, atomic_json, digest, verify_checkpoint


def replay(root: Path) -> None:
    settings = json.loads((root / "configs/bootstrap.json").read_text())[
        "released_context_study_snapshot"
    ]
    s3, aws = client(root)
    bundle = root / "data/released-context-checkpoint.tar.gz"
    if not bundle.exists() or digest(bundle) != settings["sha256"]:
        s3.download_file(aws["bucket"], settings["key"], str(bundle))
    restore(root, bundle, settings["sha256"])
    lineage, evidence = study_lineage(root)
    directory = root / "artifacts" / lineage
    if (
        not verify_checkpoint(directory / "summary", lineage)
        or len(list(directory.rglob("manifest.json"))) != 10
    ):
        raise ValueError("Publication requires all completed stages; training is prohibited")
    report = run_study(root)
    original = load_panel(root / "artifacts" / evidence["feature_lineage"] / "features")
    x, y, pairs = load_data(root)
    del x
    y = y.loc[original.dates]
    count, maximum = 0, 0.0
    log = RunLog(root / "logs/released_context_cloud_replay.jsonl", 30)
    with threadpool_limits(limits=4), log.stage("independent_context_model_replay"):
        for variant in experiment_plan():
            panel = build_panel(original, y, pairs, variant)
            for result in (r for r in report["results"] if r["variant"] == variant.name):
                stage = directory / f"fold_{result['fold']}" / variant.name
                saved = pd.read_parquet(stage / "predictions.parquet")
                actual = joblib.load(stage / "model.joblib").predict(
                    panel, result["validation_start"], result["validation_stop"]
                )
                pd.testing.assert_index_equal(saved.index, actual.index, check_names=False)
                pd.testing.assert_index_equal(saved.columns, actual.columns, check_names=False)
                error = float(np.max(np.abs(saved.to_numpy() - actual.to_numpy())))
                if not np.isfinite(error) or error > 1e-12:
                    raise ValueError("Independent prediction replay differs")
                count += 1
                maximum = max(maximum, error)
                log.event("model_replayed", completed=count, total=9, maximum_error=maximum)
    manifests = sorted((root / "artifacts").rglob("manifest.json"))
    for manifest in manifests:
        expected = manifest.relative_to(root / "artifacts").parts[0]
        if not verify_checkpoint(manifest.parent, expected):
            raise ValueError("Incomplete historical checkpoint")
    if len(manifests) != 589 or count != 9:
        raise ValueError("Unexpected project checkpoint inventory")
    atomic_json(
        root / "reports/released_context_cloud_preflight.json",
        {
            "lineage": lineage,
            "verified_utc": datetime.now(UTC).isoformat(),
            "independent_cloud_model_replays": count,
            "maximum_prediction_replay_error": maximum,
            "stage_manifests_verified": len(manifests),
            "new_training_fits": 0,
            "holdout_evaluated": False,
            "archive_sha256": settings["sha256"],
        },
    )


def execute(root: Path) -> None:
    report = json.loads((root / "reports/released_context_study.json").read_text())
    path = root / "notebooks/02_feature_research.ipynb"
    nb = nbformat.read(path, as_version=4)
    if nb.metadata.get("released_context_lineage") == report["lineage"]:
        verify(root)
        return
    nb.cells = [c for c in nb.cells if not c.metadata.get("released_context")]
    for kind, source in cells():
        cell = (
            nbformat.v4.new_markdown_cell(source)
            if kind == "md"
            else nbformat.v4.new_code_cell(source)
        )
        cell.metadata["released_context"] = True
        nb.cells.append(cell)
    os.environ["COMMODITY_ROOT"] = str(root)
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="commodity",
        config=Config({"KernelManager": {"transport": "ipc"}}),
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nb.metadata["released_context_lineage"] = report["lineage"]
    temporary = path.with_suffix(".tmp")
    nbformat.write(nb, temporary)
    temporary.replace(path)


def publish(root: Path) -> None:
    verify(root)
    for checker in ["verify_publication.py", "verify_risk_state_publication.py"]:
        subprocess.run([sys.executable, str(root / "scripts" / checker)], cwd=root, check=True)
    s3, aws = client(root)
    files = {}
    for relative in [
        "notebooks/02_feature_research.ipynb",
        "reports/figures/released_context_scores.png",
        "reports/figures/released_context_folds.png",
        "reports/figures/released_context_uncertainty.png",
    ]:
        path = root / relative
        key = "bootstrap/released-context-publication/" + relative
        sha = digest(path)
        s3.upload_file(
            str(path),
            aws["bucket"],
            key,
            ExtraArgs={"Metadata": {"sha256": sha}, "ServerSideEncryption": "AES256"},
        )
        head = s3.head_object(Bucket=aws["bucket"], Key=key)
        if head["ContentLength"] != path.stat().st_size or head["Metadata"]["sha256"] != sha:
            raise ValueError("Publication upload integrity differs")
        files[relative] = {"key": key, "sha256": sha, "bytes": path.stat().st_size}
    receipt = json.loads((root / "reports/released_context_cloud_preflight.json").read_text())
    receipt.update(
        status="completed",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        notebooks_reexecuted=1,
        project_notebooks_executed=3,
        plotly_static_figure_pairs=34,
        files=files,
        verified_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(root / "reports/aws_released_context_publication.json", receipt)
    s3.put_object(
        Bucket=aws["bucket"],
        Key="bootstrap/released-context-verification.json",
        Body=json.dumps(receipt).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["replay", "execute", "publish"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage:
        {"replay": replay, "execute": execute, "publish": publish}[args.stage](root)
        return 0
    s3, aws = client(root)

    def status(state: dict) -> None:
        log = root / "logs/released-context-publication.log"
        if log.exists():
            s3.upload_file(str(log), aws["bucket"], "bootstrap/released-context-publication.log")
        s3.put_object(
            Bucket=aws["bucket"],
            Key="bootstrap/released-context-publication-status.json",
            Body=json.dumps(state).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    python = sys.executable
    return supervise(
        root,
        [
            ("restore_and_replay", [python, __file__, "--stage", "replay"]),
            ("rendering_dependencies", ["bash", "scripts/prepare_rendering.sh"]),
            ("research_notebook_execution", [python, __file__, "--stage", "execute"]),
            ("verify_and_publish", [python, __file__, "--stage", "publish"]),
        ],
        status,
        max_seconds=360,
        status_path="logs/released-context-publication-status.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
