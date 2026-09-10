"""Synchronize only verified stage files, skipping already identical S3 objects."""

from __future__ import annotations

import json
from pathlib import Path

from botocore.exceptions import ClientError

from commodity_prediction.cloud import client
from commodity_prediction.runtime import RunLog, verify_checkpoint


class StudyStorage:
    def __init__(self, root: Path, log: RunLog):
        self.root = root
        self.log = log
        self.s3, self.settings = client(root)

    def upload_checkpoint(self, stage: Path) -> None:
        manifest = json.loads((stage / "manifest.json").read_text())
        if not verify_checkpoint(stage, manifest["lineage"]):
            raise ValueError("Cannot synchronize an incomplete checkpoint")
        from commodity_prediction.runtime import digest

        files = {**manifest["files"], "manifest.json": digest(stage / "manifest.json")}
        uploaded = 0
        for name, expected in files.items():
            path = stage / name
            key = "checkpoints/" + str(path.relative_to(self.root))
            try:
                head = self.s3.head_object(Bucket=self.settings["bucket"], Key=key)
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                    raise
                head = {}
            if (
                head.get("Metadata", {}).get("sha256") == expected
                and head.get("ContentLength") == path.stat().st_size
            ):
                continue
            self.s3.upload_file(
                str(path),
                self.settings["bucket"],
                key,
                ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": expected}},
            )
            head = self.s3.head_object(Bucket=self.settings["bucket"], Key=key)
            if (
                head.get("Metadata", {}).get("sha256") != expected
                or head["ContentLength"] != path.stat().st_size
            ):
                raise ValueError(f"S3 verification failed: {key}")
            uploaded += 1
        self.log.event(
            "s3_checkpoint_verified",
            stage=str(stage.relative_to(self.root)),
            files=len(files),
            uploaded=uploaded,
        )
