"""Restoration preserves archive confinement when reusing private worktree data."""

import shutil
import tarfile

import pytest

from commodity_prediction.runtime import atomic_json, digest, seal_checkpoint, verify_checkpoint
from scripts.restore_checkpoint import restore


def test_worktree_restoration_rejects_external_symlinks_and_accepts_local_directories(tmp_path):
    source, publication = tmp_path / "source", tmp_path / "publication"
    stage = source / "artifacts/new/summary"
    stage.mkdir(parents=True)
    (stage / "summary.json").write_text("{}")
    seal_checkpoint(stage, "new", ["summary.json"])
    atomic_json(source / "reports/risk_state_study.json", {"lineage": "new"})
    bundle = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name in ["artifacts", "reports"]:
            tar.add(source / name, arcname=name)
    publication.mkdir()
    (publication / "artifacts").symlink_to(source / "artifacts", target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe bootstrap member"):
        restore(publication, bundle, digest(bundle))
    assert not (publication / "reports").exists()
    (publication / "artifacts").unlink()
    historical = source / "artifacts/historical"
    historical.mkdir()
    (historical / "immutable.dat").write_text("preserved")
    import os

    shutil.copytree(historical, publication / "artifacts/historical", copy_function=os.link)
    restore(publication, bundle, digest(bundle))
    assert verify_checkpoint(publication / "artifacts/new/summary", "new")
    assert (publication / "artifacts/historical/immutable.dat").read_text() == "preserved"
    assert (historical / "immutable.dat").read_text() == "preserved"
