"""Publish concrete AWS bootstrap evidence after every required stage succeeds."""

import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from notebook_support import checked_reports

from commodity_prediction.cloud import client
from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    audit, report = checked_reports(root)
    run = root / "artifacts" / report["lineage"]
    stages = []
    for manifest in sorted(run.rglob("manifest.json")):
        verify_checkpoint(manifest.parent, report["lineage"])
        stages.append(str(manifest.parent.relative_to(root)))
    notebooks = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                if cell.execution_count is None or any(
                    o.output_type == "error" for o in cell.outputs
                ):
                    raise ValueError(f"Unexecuted or failed notebook: {path.name}")
        notebooks.append(path.name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    evidence = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "source_commit": commit,
        "lineage": report["lineage"],
        "checkpoint_count": len(stages),
        "notebooks": notebooks,
        "candidate_count": report["candidate_count"],
        "fold_experiments": report["experiments_completed"],
        "development_dates": audit["development_dates"],
        "feature_gate": "open",
        "holdout_evaluated": False,
    }
    atomic_json(root / "reports/aws_execution.json", evidence)
    bundle = root / "artifacts/verified-notebooks.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for directory in ["notebooks", "reports", "logs"]:
            archive.add(root / directory, arcname=directory)
    s3, aws = client(root)
    s3.upload_file(
        str(bundle),
        aws["bucket"],
        "bootstrap/verified-notebooks.tar.gz",
        ExtraArgs={"Metadata": {"sha256": digest(bundle)}, "ServerSideEncryption": "AES256"},
    )
    s3.put_object(
        Bucket=aws["bucket"],
        Key="bootstrap/verification.json",
        Body=json.dumps(evidence).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
