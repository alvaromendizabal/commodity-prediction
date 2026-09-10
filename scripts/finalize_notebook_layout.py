"""Re-execute one caption correction, with a four-minute budget and no fitting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from feature_publication import supervise
from nbclient import NotebookClient
from traitlets.config import Config

from commodity_prediction.cloud import client
from commodity_prediction.runtime import atomic_json, digest


def render(root: Path) -> None:
    path = root / "notebooks/02_feature_research.ipynb"
    nb = nbformat.read(path, as_version=4)
    old = 'title="Risk-state attribution with simultaneous 95% intervals"'
    new = (
        'title={"text":"Risk-state attribution with simultaneous 95% intervals",'
        '"xanchor":"left","xref":"container"}'
    )
    changed = 0
    for cell in nb.cells:
        if old in cell.source:
            cell.source = cell.source.replace(old, new)
            changed += 1
    if changed != 1:
        raise ValueError("Expected exactly one caption correction")
    os.environ["COMMODITY_ROOT"] = str(root)
    cfg = Config()
    cfg.KernelManager.transport = "ipc"
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="commodity",
        config=cfg,
        resources={"metadata": {"path": str(root)}},
    ).execute()
    temporary = path.with_suffix(".tmp")
    nbformat.write(nb, temporary)
    temporary.replace(path)
    for checker in ["verify_publication.py", "verify_risk_state_publication.py"]:
        subprocess.run([sys.executable, str(root / "scripts" / checker)], cwd=root, check=True)
    s3, settings = client(root)
    files = {}
    for relative in [
        "notebooks/02_feature_research.ipynb",
        "reports/figures/risk_state_uncertainty.png",
    ]:
        artifact = root / relative
        key = "bootstrap/risk-state-layout/" + relative
        sha = digest(artifact)
        s3.upload_file(
            str(artifact),
            settings["bucket"],
            key,
            ExtraArgs={"Metadata": {"sha256": sha}, "ServerSideEncryption": "AES256"},
        )
        head = s3.head_object(Bucket=settings["bucket"], Key=key)
        if head["ContentLength"] != artifact.stat().st_size or head["Metadata"]["sha256"] != sha:
            raise ValueError("Publication object integrity differs")
        files[relative] = {"key": key, "sha256": sha, "bytes": artifact.stat().st_size}
    receipt = {
        "status": "completed",
        "verified_utc": datetime.now(UTC).isoformat(),
        "reason": "Anchor the long uncertainty title to the left of its container",
        "script_sha256": digest(Path(__file__)),
        "new_training_fits": 0,
        "holdout_evaluated": False,
        "notebooks_reexecuted": 1,
        "files": files,
    }
    atomic_json(root / "reports/notebook_layout_publication.json", receipt)
    s3.put_object(
        Bucket=settings["bucket"],
        Key="bootstrap/risk-state-layout-verification.json",
        Body=json.dumps(receipt).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.render:
        render(root)
        return 0
    s3, settings = client(root)

    def publish(state: dict) -> None:
        log = root / "logs/risk-state-layout.log"
        if log.exists():
            s3.upload_file(str(log), settings["bucket"], "bootstrap/risk-state-layout.log")
        s3.put_object(
            Bucket=settings["bucket"],
            Key="bootstrap/risk-state-layout-status.json",
            Body=json.dumps(state).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    return supervise(
        root,
        [
            ("rendering_dependencies", ["bash", "scripts/prepare_rendering.sh"]),
            ("caption_correction", [sys.executable, __file__, "--render"]),
        ],
        publish,
        max_seconds=240,
        status_path="logs/risk-state-layout-status.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
