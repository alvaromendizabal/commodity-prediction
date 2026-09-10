"""Durable checkpoint upload behavior without live credentials or network calls."""

import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from commodity_prediction.runtime import RunLog, digest, seal_checkpoint
from commodity_prediction.studies.storage import StudyStorage


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.deny = False

    def head_object(self, Bucket, Key):
        if self.deny:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return self.objects[Key]

    def upload_file(self, filename, bucket, key, ExtraArgs):
        assert ExtraArgs["ServerSideEncryption"] == "AES256"
        assert ExtraArgs["Metadata"]["sha256"] == digest(Path(filename))
        self.uploads.append(key)
        self.objects[key] = {
            "ContentLength": Path(filename).stat().st_size,
            "Metadata": ExtraArgs["Metadata"],
        }


def storage_fixture(tmp_path, monkeypatch):
    import commodity_prediction.studies.storage as module

    s3 = FakeS3()
    monkeypatch.setattr(module, "client", lambda root: (s3, {"bucket": "test-bucket"}))
    stage = tmp_path / "artifacts/study/fold_0/variant"
    stage.mkdir(parents=True)
    (stage / "result.json").write_text(json.dumps({"metric": 0.2}))
    seal_checkpoint(stage, "study", ["result.json"])
    return s3, stage, StudyStorage(tmp_path, RunLog(tmp_path / "logs/test.jsonl"))


def test_checkpoint_marker_uploads_last_and_identical_objects_are_reused(tmp_path, monkeypatch):
    s3, stage, storage = storage_fixture(tmp_path, monkeypatch)
    storage.upload_checkpoint(stage)
    assert s3.uploads[-1].endswith("manifest.json")
    assert len(s3.uploads) == 2
    storage.upload_checkpoint(stage)
    assert len(s3.uploads) == 2


def test_remote_corruption_is_replaced_from_verified_local_content(tmp_path, monkeypatch):
    s3, stage, storage = storage_fixture(tmp_path, monkeypatch)
    storage.upload_checkpoint(stage)
    key = next(k for k in s3.objects if k.endswith("result.json"))
    s3.objects[key]["Metadata"] = {"sha256": "corrupt"}
    storage.upload_checkpoint(stage)
    assert s3.uploads[-1] == key
    assert s3.objects[key]["Metadata"]["sha256"] == digest(stage / "result.json")


def test_access_denial_is_not_misreported_as_missing_data(tmp_path, monkeypatch):
    s3, stage, storage = storage_fixture(tmp_path, monkeypatch)
    s3.deny = True
    with pytest.raises(ClientError, match="AccessDenied"):
        storage.upload_checkpoint(stage)
    assert not s3.uploads
