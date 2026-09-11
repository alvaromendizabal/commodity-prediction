"""Prepare isolated current-checkout dependencies; never fit or overwrite old work."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not a regular source file: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def confined(base: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError("Unsafe relative path")
    target = base
    for part in p.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Unexpected symlink: {target}")
    if base.is_symlink():
        raise ValueError("Symlink base")
    return target


def copy_verified(source: Path, destination: Path, expected: str) -> bool:
    """Copy independently and atomically; valid existing bytes are a no-write resume."""
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(source) != expected:
        raise ValueError(f"Source checksum differs: {source}")
    if destination.is_symlink():
        raise ValueError("Destination symlink")
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Conflicting destination preserved: {destination}")
        if os.path.samefile(source, destination):
            raise ValueError("Destination must not share the source inode")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Prior incomplete copy requires inspection")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(temporary) != expected or digest(source) != expected:
            raise ValueError("Copy or concurrent-source verification failed")
        # A hard link publishes only our fresh temporary inode, never the original.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def inventory(base: Path) -> dict[str, str]:
    if base.is_symlink() or not base.is_dir():
        raise ValueError("Required real directory missing")
    result = {}
    total = 0
    for folder, dirs, files in os.walk(base, followlinks=False):
        for name in dirs:
            if (Path(folder) / name).is_symlink():
                raise ValueError("Artifact directory symlink")
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Artifact is not a regular file")
            total += path.stat().st_size
            if total > 4 * 1024**3 or len(result) >= 5000:
                raise ValueError("Artifact inventory exceeds declared budget")
            result[str(path.relative_to(base))] = digest(path)
    return result


def verify_manifests(base: Path, hashes: dict[str, str]) -> int:
    count = 0
    for name in hashes:
        if Path(name).name != "manifest.json":
            continue
        relative = Path(name)
        manifest = json.loads(confined(base, name).read_text())
        if manifest.get("lineage") != relative.parts[0] or not manifest.get("files"):
            raise ValueError("Missing or incorrect artifact lineage")
        for child, expected in manifest["files"].items():
            target = confined(base, str(relative.parent / child))
            if hashes.get(str(target.relative_to(base))) != expected:
                raise ValueError("Checkpoint manifest checksum mismatch")
        count += 1
    return count


def git_state(root: Path) -> dict[str, str]:
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    def read(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            text=True, env=environment, timeout=10,
        ).strip()
    return {"head": read(["rev-parse", "HEAD"]),
            "status": read(["status", "--porcelain", "--untracked-files=normal"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--receipt-key", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.expected_commit):
        raise ValueError("Expected commit must be pinned")
    if not args.receipt_key.startswith("operations/workspace-ready/"):
        raise ValueError("Receipt must remain in its private operational prefix")
    import boto3
    from botocore.config import Config

    home = Path("/home/sagemaker-user")
    root = home / "projects/commodity-prediction-current"
    original = home / "projects/commodity-prediction"
    context = home / "projects/commodity-prediction-context"
    old = [original, context, home / "projects/commodity-prediction-publication"]
    audit = home / "commodity-runtime-audit"
    audit.mkdir(exist_ok=True)
    lock = (audit / "prepare.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    s3 = boto3.client("s3", region_name="us-west-2", config=Config(
        connect_timeout=5, read_timeout=10, retries={"max_attempts": 0}))
    bucket = "sagemaker-commodity-prediction-560403859723-us-west-2"
    started = time.monotonic()
    done = threading.Event()
    mutex = threading.Lock()
    report = {"project": "commodity-prediction", "status": "RUNNING", "stage": "preflight",
              "new_training_fits": 0, "model_loads": 0, "notebook_executions": 0,
              "source_commit": args.expected_commit, "hard_budget_seconds": 240,
              "heartbeat_seconds": 15, "errors": [], "copies_created": 0, "copies_reused": 0}

    def save() -> None:
        with mutex:
            report.update(observed_utc=datetime.now(UTC).isoformat(),
                          elapsed_seconds=round(time.monotonic() - started, 3))
            encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
            temporary = audit / "receipt.tmp"
            temporary.write_bytes(encoded)
            temporary.replace(audit / "receipt.json")
            s3.put_object(Bucket=bucket, Key=args.receipt_key, Body=encoded,
                          ContentType="application/json", ServerSideEncryption="AES256",
                          Metadata={"status": report["status"],
                                    "sha256": hashlib.sha256(encoded).hexdigest()})
            print(json.dumps(report, sort_keys=True), flush=True)

    def heartbeat() -> None:
        while not done.wait(15):
            try:
                save()
            except Exception as exc:
                print("HEARTBEAT_UPLOAD_ERROR " + type(exc).__name__, flush=True)

    def run(command: list[str], seconds: int) -> str:
        remaining = 240 - (time.monotonic() - started)
        return subprocess.check_output(command, cwd=root, text=True,
                                       timeout=max(1, min(seconds, remaining)),
                                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    def timeout_handler(signum, frame):
        raise TimeoutError("Workspace preparation hard deadline reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(240)
    worker = threading.Thread(target=heartbeat, daemon=True)
    before = {}
    source_hashes = {}
    raw_hashes = {}
    try:
        save()
        worker.start()
        if root.is_symlink() or git_state(root) != {"head": args.expected_commit, "status": ""}:
            raise ValueError("Current checkout changed; preserve it and inspect")
        if shutil.disk_usage(home).free < 4 * 1024**3:
            raise ValueError("Need four GiB free; do not delete unrelated files")
        before = {str(p): git_state(p) for p in old}
        lineage = json.loads((root / "reports/lineage.json").read_text())
        raw_hashes = {p: h for p, h in lineage["files"].items() if p.startswith("data/raw/")}
        report["stage"] = "copy_verified_raw_inputs"
        for name, expected in raw_hashes.items():
            copied = copy_verified(confined(original, name), confined(root, name), expected)
            report["copies_created" if copied else "copies_reused"] += 1
        report["raw_files_verified"] = len(raw_hashes)
        report["stage"] = "verify_and_copy_completed_checkpoints"
        source_hashes = inventory(context / "artifacts")
        count = verify_manifests(context / "artifacts", source_hashes)
        if count != 589:
            raise ValueError(f"Expected 589 preserved stage manifests, found {count}")
        for name, expected in source_hashes.items():
            copied = copy_verified(confined(context / "artifacts", name),
                                   confined(root / "artifacts", name), expected)
            report["copies_created" if copied else "copies_reused"] += 1
        destination_hashes = inventory(root / "artifacts")
        if destination_hashes != source_hashes:
            raise ValueError("Independent checkpoint inventory differs")
        report["checkpoint_manifests_verified"] = verify_manifests(root / "artifacts", destination_hashes)
        report["artifact_files_verified"] = len(destination_hashes)
        report["stage"] = "install_frozen_independent_environment"
        if (root / ".venv").is_symlink():
            raise ValueError("Refuse a shared environment symlink")
        lock_hash = digest(root / "uv.lock")
        uv = shutil.which("uv")
        if uv is None:
            raise ValueError("uv is missing; no speculative bootstrap")
        run([uv, "sync", "--frozen", "--extra", "dev", "--python", "/opt/conda/bin/python"], 90)
        if digest(root / "uv.lock") != lock_hash:
            raise ValueError("Lockfile changed")
        python = str(root / ".venv/bin/python")
        smoke = run([python, "-c", "import json,sys,numpy,pandas,sklearn,commodity_prediction; "
                     "print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,"
                     "pandas=pandas.__version__,sklearn=sklearn.__version__,source=commodity_prediction.__file__)))"], 15)
        report["runtime"] = json.loads(smoke.strip())
        if not Path(report["runtime"]["source"]).resolve().is_relative_to(root / "src"):
            raise ValueError("Environment resolves a different source checkout")
        report["stage"] = "verify_existing_publication_without_execution"
        report["publication_verifiers"] = {}
        for script in ["verify_publication.py", "verify_risk_state_publication.py", "verify_released_context_publication.py"]:
            report["publication_verifiers"][script] = run([python, "scripts/" + script], 15).strip()
        run([python, "-m", "ipykernel", "install", "--user", "--name", "commodity-current",
             "--display-name", "Commodity Research - current"], 15)
        report["kernel"] = "commodity-current"
        report["status"] = "WORKSPACE_READY"
    except Exception as exc:
        report["status"] = "STOPPED"
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        signal.alarm(0)
        done.set()
        if worker.is_alive():
            worker.join(timeout=15)
        # Bounded metadata and copied-source verification; no automatic recovery.
        try:
            report["old_git_states_unchanged"] = before == {str(p): git_state(p) for p in old}
            report["copied_sources_unchanged"] = all(
                digest(confined(context / "artifacts", p)) == h for p, h in source_hashes.items()) and all(
                digest(confined(original, p)) == h for p, h in raw_hashes.items())
            if not report["old_git_states_unchanged"] or not report["copied_sources_unchanged"]:
                raise ValueError("Original-source preservation check failed")
        except Exception as exc:
            report["status"] = "STOPPED"
            report["errors"].append(str(exc))
        report["stage"] = "finished"
        save()
        response = s3.get_object(Bucket=bucket, Key=args.receipt_key)
        body = response["Body"].read()
        if hashlib.sha256(body).hexdigest() != response["Metadata"]["sha256"]:
            raise ValueError("S3 receipt read-back mismatch")
        print("RECEIPT_READBACK_VERIFIED", flush=True)
        lock.close()
    if report["status"] != "WORKSPACE_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
