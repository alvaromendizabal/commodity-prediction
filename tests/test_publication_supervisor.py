"""A failed cloud preflight or child must never look like completed research."""

import json
import sys

import pytest

from scripts.feature_publication import supervise


def test_child_failure_is_durable_and_prevents_later_work(tmp_path):
    records = []
    marker = tmp_path / "must_not_run"
    commands = [
        ("fit", [sys.executable, "-c", "raise SystemExit(7)"]),
        ("later", [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]),
    ]
    assert supervise(tmp_path, commands, records.append) == 7
    assert not marker.exists()
    assert records[-1]["status"] == "failed"
    assert records[-1]["stage"] == "fit"
    assert records[-1]["exit_code"] == 7
    assert (
        json.loads((tmp_path / "logs/compact-publication-status.json").read_text()) == records[-1]
    )


def test_no_subprocess_starts_when_cloud_preflight_fails(tmp_path):
    marker = tmp_path / "must_not_run"

    def reject(_):
        raise RuntimeError("No role credentials")

    with pytest.raises(RuntimeError, match="No role credentials"):
        supervise(
            tmp_path,
            [("fit", [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])],
            reject,
        )
    assert not marker.exists()
    assert (
        json.loads((tmp_path / "logs/compact-publication-status.json").read_text())["status"]
        == "failed"
    )


def test_success_follows_all_stages_and_emits_heartbeats(tmp_path):
    records = []
    commands = [("brief_work", [sys.executable, "-c", "import time; time.sleep(0.08)"])]
    assert supervise(tmp_path, commands, records.append, heartbeat=0.02) == 0
    assert sum(r["stage"] == "brief_work" and r["status"] == "running" for r in records) >= 2
    assert records[-1]["status"] == "completed"
    assert records[-1]["exit_code"] == 0


def test_timeout_stops_work_and_records_failure(tmp_path):
    with pytest.raises(TimeoutError, match="runtime budget"):
        supervise(
            tmp_path,
            [("long_work", [sys.executable, "-c", "import time; time.sleep(60)"])],
            lambda _: None,
            heartbeat=0.02,
            max_seconds=0.03,
        )
    state = json.loads((tmp_path / "logs/compact-publication-status.json").read_text())
    assert state["stage"] == "long_work" and state["status"] == "failed"
