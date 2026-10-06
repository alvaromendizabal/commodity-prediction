"""Run an authored, dependency-free illustration of point-in-time label access."""

import json
import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class Label:
    """A fictional observation with an explicit publication day."""

    instrument: str
    observed_day: int
    released_day: int
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("instrument must be nonempty")
        if any(type(day) is not int or day < 1 for day in (self.observed_day, self.released_day)):
            raise ValueError("days must be positive integers")
        if self.released_day <= self.observed_day:
            raise ValueError("release must follow observation in this delayed-label example")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("label value must be a finite number")
        if not math.isfinite(self.value):
            raise ValueError("label value must be a finite number")


class UnreleasedLabelError(ValueError):
    """Raised when a forecast requests a label before publication."""


class LabelSnapshot:
    """Expose only labels released by the start of the requested day.

    This educational guard is a timing contract, not a security boundary.
    Release dates are supplied explicitly; official competition scheduling and
    the project's private research features are deliberately outside this example.
    """

    def __init__(self, labels: tuple[Label, ...], as_of_day: int) -> None:
        if type(as_of_day) is not int or as_of_day < 1:
            raise ValueError("as_of_day must be a positive integer")
        self.as_of_day = as_of_day
        self._labels = {(label.instrument, label.observed_day): label for label in labels}
        if len(self._labels) != len(labels):
            raise ValueError("duplicate instrument and observation day")

    def read(self, instrument: str, observed_day: int) -> float:
        label = self._labels[instrument, observed_day]
        if label.released_day > self.as_of_day:
            raise UnreleasedLabelError(
                f"{instrument} day {observed_day} is released on day "
                f"{label.released_day}; requested snapshot is day {self.as_of_day}"
            )
        return label.value

    def available(self) -> tuple[Label, ...]:
        return tuple(
            sorted(
                (label for label in self._labels.values() if label.released_day <= self.as_of_day),
                key=lambda label: (label.released_day, label.instrument, label.observed_day),
            )
        )


def authored_labels() -> tuple[Label, ...]:
    """Fictional values and variable release delays; no competition data."""
    return (
        Label("metal_alpha", 1, 3, 0.20),
        Label("metal_beta", 1, 5, -0.10),
        Label("metal_alpha", 2, 6, 0.05),
        Label("metal_beta", 3, 7, 0.30),
    )


def run_demo() -> dict[str, object]:
    labels = authored_labels()
    snapshot = LabelSnapshot(labels, as_of_day=5)
    known = snapshot.available()
    try:
        snapshot.read("metal_alpha", 2)
    except UnreleasedLabelError as error:
        blocked_request = str(error)
    else:
        raise AssertionError("Future-label access was not rejected")
    return {
        "scope": "synthetic timing example; no model training or competition evaluation",
        "as_of_day": snapshot.as_of_day,
        "total_authored_labels": len(labels),
        "available_labels": len(known),
        "unreleased_labels": len(labels) - len(known),
        "available_keys": [[label.instrument, label.observed_day] for label in known],
        "released_label_mean": round(fmean(label.value for label in known), 8),
        "boundary_rule": "released_day <= as_of_day; release is at the start of that day",
        "future_access": "rejected",
        "rejection_reason": blocked_request,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
