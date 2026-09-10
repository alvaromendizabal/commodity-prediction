"""Pin verified private domain snapshots without rebuilding completed archives."""

import json
import tarfile
from pathlib import Path

from notebook_support import checked_attribution, checked_domain

from commodity_prediction.cloud import client
from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def pin(root: Path, report: dict, stem: str, filenames: list[str], setting: str) -> None:
    directory = root / "artifacts" / report["lineage"]
    for manifest in directory.rglob("manifest.json"):
        verify_checkpoint(manifest.parent, report["lineage"])
    settings = json.loads((root / "configs/bootstrap.json").read_text())
    s3, aws = client(root)
    key = f"bootstrap/{stem}-checkpoint.tar.gz"
    existing = settings.get(setting)
    if existing and existing["lineage"] == report["lineage"]:
        head = s3.head_object(Bucket=aws["bucket"], Key=existing["key"])
        if head.get("Metadata", {}).get("sha256") != existing["sha256"]:
            raise ValueError("Pinned private snapshot changed unexpectedly")
        print(
            json.dumps({"snapshot_reused": existing["key"], "bytes": head["ContentLength"]}),
            flush=True,
        )
        return
    bundle = root / "artifacts" / f"{stem}-checkpoint.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(directory, arcname=str(directory.relative_to(root)))
        for name in filenames:
            archive.add(root / "reports" / name, arcname="reports/" + name)
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
    settings[setting] = {"key": key, "sha256": sha, "lineage": report["lineage"]}
    atomic_json(root / "configs/bootstrap.json", settings)
    print(
        json.dumps({"snapshot_key": key, "sha256": sha, "bytes": head["ContentLength"]}), flush=True
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pin(
        root,
        checked_domain(root),
        "domain-study",
        ["domain_study.json", "domain_lineage.json"],
        "domain_study_snapshot",
    )
    pin(
        root,
        checked_attribution(root),
        "tree-attribution",
        ["tree_attribution.json", "tree_attribution_lineage.json"],
        "tree_attribution_snapshot",
    )


if __name__ == "__main__":
    main()
