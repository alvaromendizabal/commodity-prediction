"""Run risk-state research with private per-stage archives via expiring S3 URLs."""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

from commodity_prediction.domain.risk_state.run import run_study, study_lineage
from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


class ArchivePublisher:
    """Persist sealed stages using only the headers authorized by each signed URL."""

    def __init__(self, root: Path, lineage: str, plan: list[dict]):
        self.root, self.lineage = root, lineage
        self.plan = {item["stage"]: item for item in plan}
        self.receipts: dict[str, dict] = {}

    def __call__(self, stage: Path) -> None:
        if not verify_checkpoint(stage, self.lineage):
            raise ValueError("Cannot publish an unsealed stage")
        name = str(stage.relative_to(self.root / "artifacts" / self.lineage))
        item = self.plan[name]
        bundle = self.root / "data/risk_state_uploads" / (name.replace("/", "_") + ".tar.gz")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(stage, arcname=str(stage.relative_to(self.root)))
        data = bundle.read_bytes()
        request = urllib.request.Request(
            item["url"],
            data=data,
            method="PUT",
            headers=item.get("headers", {}),
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError("Private checkpoint upload failed")
        self.receipts[name] = {"key": item["key"], "sha256": digest(bundle), "bytes": len(data)}
        atomic_json(self.root / "artifacts" / self.lineage / "remote_uploads.json", self.receipts)
        print(json.dumps({"private_stage_uploaded": name, "bytes": len(data)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-plan", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lineage, _ = study_lineage(root)
    plan = json.loads(args.upload_plan.read_text())
    publisher = ArchivePublisher(root, lineage, plan)
    run_study(root, checkpoint_hook=publisher)
    # A fully completed local run verifies without refitting. Publish any remaining
    # sealed stages here after the explicit upload authorization is available.
    for manifest in sorted((root / "artifacts" / lineage).rglob("manifest.json")):
        name = str(manifest.parent.relative_to(root / "artifacts" / lineage))
        if name not in publisher.receipts:
            publisher(manifest.parent)


if __name__ == "__main__":
    main()
