"""Pin the verified private domain checkpoint for reproducible AWS startup."""

import json
import tarfile
from pathlib import Path

from notebook_support import checked_domain

from commodity_prediction.cloud import client
from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = checked_domain(root)
    directory = root / "artifacts" / report["lineage"]
    for manifest in directory.rglob("manifest.json"):
        verify_checkpoint(manifest.parent, report["lineage"])
    bundle = root / "artifacts/domain-study-checkpoint.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(directory, arcname=str(directory.relative_to(root)))
        for name in ["domain_study.json", "domain_lineage.json"]:
            archive.add(root / "reports" / name, arcname="reports/" + name)
    s3, aws = client(root)
    key = "bootstrap/domain-study-checkpoint.tar.gz"
    sha = digest(bundle)
    s3.upload_file(
        str(bundle),
        aws["bucket"],
        key,
        ExtraArgs={"Metadata": {"sha256": sha}, "ServerSideEncryption": "AES256"},
    )
    head = s3.head_object(Bucket=aws["bucket"], Key=key)
    if (
        head["ContentLength"] != bundle.stat().st_size
        or head.get("Metadata", {}).get("sha256") != sha
    ):
        raise ValueError("Domain snapshot failed S3 verification")
    settings = json.loads((root / "configs/bootstrap.json").read_text())
    settings["domain_study_snapshot"] = {"key": key, "sha256": sha, "lineage": report["lineage"]}
    atomic_json(root / "configs/bootstrap.json", settings)
    print(
        json.dumps({"snapshot_key": key, "sha256": sha, "bytes": head["ContentLength"]}), flush=True
    )


if __name__ == "__main__":
    main()
