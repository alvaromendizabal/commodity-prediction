"""Package only verified context-study stages for a private, resumable checkpoint."""

import json
import tarfile
from pathlib import Path

from commodity_prediction.domain.released_context.run import study_lineage
from commodity_prediction.runtime import atomic_json, digest, verify_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    lineage, _ = study_lineage(root)
    directory = root / "artifacts" / lineage
    manifests = sorted(directory.rglob("manifest.json"))
    if not manifests:
        raise ValueError("No completed context stages to preserve")
    for manifest in manifests:
        if not verify_checkpoint(manifest.parent, lineage):
            raise ValueError("Incomplete context checkpoint")
    bundle = root / "data/released-context-checkpoint.tar.gz"
    temporary = bundle.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name in ["lineage.json", "runtime_budget.json"]:
            path = directory / name
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(root)))
        for manifest in manifests:
            archive.add(manifest.parent, arcname=str(manifest.parent.relative_to(root)))
        for path in sorted((root / "reports").glob("released_context_*.json")):
            if path.name != "released_context_snapshot.json":
                archive.add(path, arcname=str(path.relative_to(root)))
    temporary.replace(bundle)
    receipt = {
        "lineage": lineage,
        "sha256": digest(bundle),
        "bytes": bundle.stat().st_size,
        "stage_manifests": len(manifests),
        "models": sum((p.parent / "model.joblib").exists() for p in manifests),
        "key": "bootstrap/released-context-checkpoint.tar.gz",
    }
    atomic_json(root / "reports/released_context_snapshot.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
