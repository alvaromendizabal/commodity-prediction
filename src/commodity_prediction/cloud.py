"""Private S3 data bootstrap and verified per-stage checkpoint synchronization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config

from .runtime import RunLog, digest


def client(root: Path):
    settings = json.loads((root / "configs/aws.json").read_text())
    s3 = boto3.Session(region_name=settings["region"]).client(
        "s3",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 5},
            connect_timeout=10,
            read_timeout=60,
        ),
    )
    return s3, settings


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
        bundle.extractall(destination)


def bootstrap_data(root: Path) -> None:
    s3, settings = client(root)
    raw = root / "data/raw"
    archive = root / "data/mitsui-commodity-prediction-challenge.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    response = s3.head_object(Bucket=settings["bucket"], Key=settings["raw_archive_key"])
    expected = response.get("Metadata", {}).get("sha256")
    if not archive.exists() or (expected and digest(archive) != expected):
        s3.download_file(settings["bucket"], settings["raw_archive_key"], str(archive))
    if expected and digest(archive) != expected:
        raise ValueError("Downloaded archive digest mismatch")
    extract_archive(archive, raw)


def upload_stage(root: Path, stage: Path, log: RunLog) -> None:
    s3, settings = client(root)
    files = sorted(p for p in stage.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    # Commit markers are uploaded last so interrupted transfers are never complete.
    files.sort(key=lambda p: p.name == "manifest.json")
    for path in files:
        key = "checkpoints/" + str(path.relative_to(root))
        sha = digest(path)
        s3.upload_file(
            str(path),
            settings["bucket"],
            key,
            ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": sha}},
        )
        head = s3.head_object(Bucket=settings["bucket"], Key=key)
        if (
            head["ContentLength"] != path.stat().st_size
            or head.get("Metadata", {}).get("sha256") != sha
        ):
            raise ValueError(f"S3 checkpoint verification failed: {key}")
    log.event("s3_checkpoint_verified", stage=str(stage.relative_to(root)), files=len(files))


def restore_run(root: Path, lineage: str, log: RunLog) -> None:
    s3, settings = client(root)
    prefix = f"checkpoints/artifacts/{lineage}/"
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=settings["bucket"], Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            destination = root / key.removeprefix("checkpoints/")
            if not destination.resolve().is_relative_to((root / "artifacts" / lineage).resolve()):
                raise ValueError("Unsafe S3 checkpoint path")
            head = s3.head_object(Bucket=settings["bucket"], Key=key)
            expected = head.get("Metadata", {}).get("sha256")
            if expected is None:
                raise ValueError("S3 checkpoint has no content fingerprint")
            if destination.exists() and digest(destination) == expected:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            s3.download_file(settings["bucket"], key, str(temporary))
            if digest(temporary) != expected:
                raise ValueError("Restored checkpoint digest mismatch")
            temporary.replace(destination)
            count += 1
    log.event("s3_restore_completed", files_restored=count)
