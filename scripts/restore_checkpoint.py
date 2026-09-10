"""Restore a checksum-pinned private bootstrap snapshot without replacing source."""

import json
import tarfile
from pathlib import Path

from commodity_prediction.cloud import client
from commodity_prediction.runtime import digest, verify_checkpoint


def restore(root: Path, bundle: Path, expected: str) -> None:
    if digest(bundle) != expected:
        raise ValueError("Bootstrap checkpoint archive has changed")
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] not in {"artifacts", "reports"}:
                raise ValueError("Bootstrap snapshot may only contain artifacts and reports")
            if not (root / member.name).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe bootstrap member")
            if member.issym() or member.islnk():
                raise ValueError("Bootstrap links are prohibited")
        archive.extractall(root, filter="data")
    for filename in [
        "research.json",
        "feature_study.json",
        "domain_study.json",
        "tree_attribution.json",
    ]:
        report_path = root / "reports" / filename
        if report_path.exists():
            report = json.loads(report_path.read_text())
            run = root / "artifacts" / report["lineage"]
            for manifest in run.rglob("manifest.json"):
                verify_checkpoint(manifest.parent, report["lineage"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/bootstrap.json").read_text())
    s3, aws = client(root)
    bundle = root / "data/research-checkpoint.tar.gz"
    bundle.parent.mkdir(exist_ok=True)
    if not bundle.exists() or digest(bundle) != settings["snapshot_sha256"]:
        s3.download_file(aws["bucket"], settings["snapshot_key"], str(bundle))
    restore(root, bundle, settings["snapshot_sha256"])
    if "feature_study_snapshot" in settings:
        study = settings["feature_study_snapshot"]
        bundle = root / "data/feature-study-checkpoint.tar.gz"
        if not bundle.exists() or digest(bundle) != study["sha256"]:
            s3.download_file(aws["bucket"], study["key"], str(bundle))
        restore(root, bundle, study["sha256"])
    for name, filename in [
        ("domain_study_snapshot", "domain-study-checkpoint.tar.gz"),
        ("tree_attribution_snapshot", "tree-attribution-checkpoint.tar.gz"),
    ]:
        if name in settings:
            study = settings[name]
            bundle = root / "data" / filename
            if not bundle.exists() or digest(bundle) != study["sha256"]:
                s3.download_file(aws["bucket"], study["key"], str(bundle))
            restore(root, bundle, study["sha256"])
    print("Verified bootstrap snapshot and all stage manifests", flush=True)


if __name__ == "__main__":
    main()
