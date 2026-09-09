"""Atomic, fingerprinted artifacts and UTC stage/heartbeat logs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RunLog:
    def __init__(self, path: Path, heartbeat_seconds: float = 15):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.heartbeat_seconds = heartbeat_seconds
        self.lock = threading.Lock()

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "event": event,
            "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.lock, self.path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            print(line, flush=True)

    @contextmanager
    def stage(self, name: str):
        started = time.monotonic()
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(self.heartbeat_seconds):
                self.event(
                    "heartbeat",
                    stage=name,
                    stage_elapsed_seconds=round(time.monotonic() - started, 3),
                )

        worker = threading.Thread(target=heartbeat, daemon=True)
        self.event("stage_started", stage=name, stage_elapsed_seconds=0)
        worker.start()
        try:
            yield
        except BaseException as exc:
            self.event(
                "stage_failed",
                stage=name,
                error_type=type(exc).__name__,
                stage_elapsed_seconds=round(time.monotonic() - started, 3),
            )
            raise
        else:
            self.event(
                "stage_completed",
                stage=name,
                stage_elapsed_seconds=round(time.monotonic() - started, 3),
            )
        finally:
            stop.set()
            worker.join(timeout=1)


def verify_checkpoint(directory: Path, lineage: str) -> dict[str, Any] | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest["lineage"] != lineage:
        raise ValueError(f"Stale checkpoint lineage: {directory}")
    if not manifest.get("files"):
        raise ValueError(f"Empty checkpoint: {directory}")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint path escapes its directory")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    return manifest


def seal_checkpoint(directory: Path, lineage: str, files: list[str]) -> None:
    atomic_json(
        directory / "manifest.json",
        {
            "lineage": lineage,
            "files": {name: digest(directory / name) for name in files},
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
