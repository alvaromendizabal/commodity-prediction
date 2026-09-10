"""Supervise feature publication with durable progress and fail-fast cloud access."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from commodity_prediction.cloud import client
from commodity_prediction.runtime import atomic_json


def stages(python: str) -> list[tuple[str, list[str]]]:
    return [
        ("quality", [python, "scripts/quality.py"]),
        ("compact_study", [python, "-m", "commodity_prediction.domain.compact.run", "--sync-s3"]),
        ("verified_resume", [python, "-m", "commodity_prediction.domain.compact.run"]),
        ("private_snapshots", [python, "scripts/snapshot_domain.py"]),
        ("rendering_dependencies", ["bash", "scripts/prepare_rendering.sh"]),
        ("notebook_generation", [python, "scripts/make_notebooks.py"]),
        ("notebook_execution", [python, "scripts/execute_notebooks.py"]),
        ("publication_verification", [python, "scripts/verify_feature_research.py"]),
    ]


def supervise(
    root: Path,
    commands: list[tuple[str, list[str]]],
    publish: Callable[[dict[str, Any]], None],
    heartbeat: float = 30,
    max_seconds: float = 2700,
) -> int:
    started = monotonic()
    state: dict[str, Any] = {"status": "running", "stage": "cloud_preflight", "exit_code": None}

    def record() -> None:
        state.update(
            utc=datetime.now(UTC).isoformat(), total_elapsed_seconds=round(monotonic() - started, 3)
        )
        atomic_json(root / "logs/compact-publication-status.json", state)
        print(json.dumps(state), flush=True)
        publish(dict(state))

    try:
        # The first remote write must succeed before any fit or quality subprocess starts.
        record()
        for name, command in commands:
            state.update(stage=name, stage_elapsed_seconds=0)
            stage_started = monotonic()
            record()
            with subprocess.Popen(command, cwd=root, start_new_session=True) as process:
                try:
                    while True:
                        try:
                            code = process.wait(timeout=heartbeat)
                            break
                        except subprocess.TimeoutExpired:
                            if monotonic() - started >= max_seconds:
                                raise TimeoutError(
                                    "Feature publication exceeded its runtime budget"
                                ) from None
                            state["stage_elapsed_seconds"] = round(monotonic() - stage_started, 3)
                            record()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            state["stage_elapsed_seconds"] = round(monotonic() - stage_started, 3)
            if code:
                state.update(status="failed", exit_code=code)
                record()
                return code
        state.update(status="completed" if commands else "ready", exit_code=0)
        record()
        return 0
    except Exception as error:
        state.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
        atomic_json(root / "logs/compact-publication-status.json", state)
        print(json.dumps(state), flush=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    s3, settings = client(root)

    def publish(state: dict) -> None:
        log = root / "logs/compact-publication.log"
        if log.exists():
            s3.upload_file(str(log), settings["bucket"], "bootstrap/compact-publication.log")
        s3.put_object(
            Bucket=settings["bucket"],
            Key="bootstrap/compact-publication-status.json",
            Body=json.dumps(state).encode(),
            ContentType="application/json",
        )

    def preflight_and_publish(state: dict) -> None:
        if state["stage"] == "cloud_preflight":
            s3.head_object(Bucket=settings["bucket"], Key=settings["raw_archive_key"])
            s3.list_objects_v2(
                Bucket=settings["bucket"], Prefix="checkpoints/artifacts/", MaxKeys=1
            )
        publish(state)

    return supervise(
        root, [] if args.preflight_only else stages(sys.executable), preflight_and_publish
    )


if __name__ == "__main__":
    sys.exit(main())
