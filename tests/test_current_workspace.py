"""Offline tests for isolated recovery, checksum gates, and no-write resumption."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "preparation", Path(__file__).resolve().parents[1] / "scripts/prepare_current_workspace.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_copy_is_independent_and_resume_preserves_timestamp(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"completed model bytes")
    expected = module.digest(source)
    assert module.copy_verified(source, target, expected)
    assert not os.path.samefile(source, target)
    timestamp = target.stat().st_mtime_ns
    assert not module.copy_verified(source, target, expected)
    assert timestamp == target.stat().st_mtime_ns
    assert source.read_bytes() == b"completed model bytes"


@pytest.mark.parametrize("value", ["../escape", "/absolute", "x/../../escape"])
def test_paths_cannot_escape(tmp_path, value):
    with pytest.raises(ValueError):
        module.confined(tmp_path, value)


def test_conflicting_destination_preserved(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    target.write_bytes(b"different local work")
    with pytest.raises(ValueError, match="Conflicting"):
        module.copy_verified(source, target, module.digest(source))
    assert target.read_bytes() == b"different local work"


def test_source_tampering_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    expected = module.digest(source)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source checksum"):
        module.copy_verified(source, target, expected)
    assert not target.exists()


def test_shared_inode_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    os.link(source, target)
    with pytest.raises(ValueError, match="share"):
        module.copy_verified(source, target, module.digest(source))


def test_directory_symlinks_rejected(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.confined(tmp_path, "alias/file")
    with pytest.raises(ValueError, match="symlink"):
        module.inventory(tmp_path)


def test_checkpoint_manifests_and_corruption(tmp_path):
    stage = tmp_path / "lineage" / "fold_0"
    stage.mkdir(parents=True)
    (stage / "model.joblib").write_bytes(b"bytes only, never deserialized")
    (stage / "manifest.json").write_text(
        json.dumps(
            {"lineage": "lineage", "files": {"model.joblib": module.digest(stage / "model.joblib")}}
        )
    )
    assert module.verify_manifests(tmp_path, module.inventory(tmp_path)) == 1
    (stage / "model.joblib").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_manifests(tmp_path, module.inventory(tmp_path))
