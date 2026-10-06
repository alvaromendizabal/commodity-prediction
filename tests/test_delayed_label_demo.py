"""Point-in-time behavior of the public, authored example."""

import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "delayed_label_demo.py"
MODULE = runpy.run_path(str(DEMO))
Label = MODULE["Label"]
LabelSnapshot = MODULE["LabelSnapshot"]
UnreleasedLabelError = MODULE["UnreleasedLabelError"]
authored_labels = MODULE["authored_labels"]


def test_release_boundary_blocks_before_and_allows_at_publication():
    labels = authored_labels()
    with pytest.raises(UnreleasedLabelError, match="released on day 5"):
        LabelSnapshot(labels, 4).read("metal_beta", 1)
    assert LabelSnapshot(labels, 5).read("metal_beta", 1) == -0.10
    assert LabelSnapshot(labels, 2).available() == ()


def test_future_values_cannot_change_current_snapshot():
    labels = authored_labels()
    changed = tuple(
        replace(label, value=999_999.0) if label.released_day > 5 else label for label in labels
    )
    assert LabelSnapshot(labels, 5).available() == LabelSnapshot(changed, 5).available()
    assert LabelSnapshot(labels, 6).available() != LabelSnapshot(changed, 6).available()


def test_order_does_not_change_snapshot_and_duplicate_keys_fail():
    labels = authored_labels()
    assert (
        LabelSnapshot(labels, 5).available()
        == LabelSnapshot(tuple(reversed(labels)), 5).available()
    )
    with pytest.raises(ValueError, match="duplicate"):
        LabelSnapshot((*labels, replace(labels[0], released_day=10)), 5)


@pytest.mark.parametrize(
    "changes",
    [
        {"released_day": 1},
        {"released_day": 0},
        {"observed_day": -1},
        {"observed_day": True},
        {"released_day": 3.5},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": True},
        {"instrument": " "},
    ],
)
def test_invalid_records_are_rejected(changes):
    values = {"instrument": "fictional", "observed_day": 1, "released_day": 3, "value": 0.2}
    with pytest.raises(ValueError):
        Label(**(values | changes))


@pytest.mark.parametrize("day", [0, -1, 1.5, True])
def test_invalid_snapshot_day_is_rejected(day):
    with pytest.raises(ValueError, match="positive integer"):
        LabelSnapshot(authored_labels(), day)


def test_cli_runs_without_site_packages_or_working_directory_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(DEMO)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    report = json.loads(result.stdout)
    assert report["available_labels"] == report["unreleased_labels"] == 2
    assert report["released_label_mean"] == 0.05
    assert report["future_access"] == "rejected"
    assert list(tmp_path.iterdir()) == []
