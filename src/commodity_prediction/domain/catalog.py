"""Typed feature panels; templates, source series, and target assignments stay distinct."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Panel:
    values: np.ndarray
    dates: list[int]
    targets: list[str]
    names: list[str]
    source_series: dict[str, int]

    def validate(self) -> None:
        if self.values.shape != (len(self.dates), len(self.targets), len(self.names)):
            raise ValueError("Panel axes do not match metadata")
        if len(set(self.names)) != len(self.names) or np.isinf(self.values).any():
            raise ValueError("Duplicate templates or infinite features")
        if not np.all(np.diff(self.dates) == 1):
            raise ValueError("Panel dates must be contiguous")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "panel.npz", values=self.values)

    def inventory(self) -> dict:
        families = dict(Counter(n.split("__", 1)[0] for n in self.names))
        return {
            "templates": len(self.names),
            "family_templates": families,
            "source_series": self.source_series,
            "source_series_count": sum(self.source_series.values()),
            "target_template_assignments": len(self.names) * len(self.targets),
            "dates": self.dates,
            "targets": self.targets,
            "names": self.names,
            "counting_note": "Templates are pooled input columns. Source series count instantiated intermediate market/target series before projection, including controls and transformations of earlier features. Target-template assignments include shared and structurally unavailable values; none of these counts is a claim of independent signals or newly unique features.",
        }


class Builder:
    def __init__(self, index: pd.Index, pairs: pd.DataFrame):
        self.index = index
        self.pairs = pairs
        self.left = [p.split(" - ")[0] for p in pairs.pair]
        self.right = [p.split(" - ")[1] if " - " in p else "" for p in pairs.pair]
        self.output: dict[str, np.ndarray] = {}
        self.counts: Counter[str] = Counter()
        self.applicability: dict[tuple, int] = {}

    def add(self, family: str, name: str, value: np.ndarray) -> None:
        key = family + "__" + name
        if key in self.output or value.shape != (len(self.index), len(self.pairs)):
            raise ValueError(f"Invalid panel feature: {key}")
        value = np.asarray(value, dtype=np.float32)
        value[~np.isfinite(value)] = np.nan
        self.output[key] = value

    def asset(self, family: str, name: str, frame: pd.DataFrame, both: bool = True) -> None:
        if not frame.index.equals(self.index):
            raise ValueError("Asset dates do not align")
        self.counts[family] += len(frame.columns)
        left = frame.reindex(columns=self.left, fill_value=0).to_numpy(dtype=float)
        right = frame.reindex(columns=self.right, fill_value=0).to_numpy(dtype=float, copy=True)
        right[:, np.asarray(self.right) == ""] = 0
        supported = set(frame.columns)
        if not set(self.left + [a for a in self.right if a]).issubset(supported):
            signature = (family, tuple(sorted(supported)))
            if signature not in self.applicability:
                self.applicability[signature] = len(self.applicability)
                applicable = np.array(
                    [
                        int(a in supported) + int(c in supported)
                        for a, c in zip(self.left, self.right, strict=True)
                    ]
                )
                self.add(
                    family,
                    f"applicable_legs_{self.applicability[signature]}",
                    np.broadcast_to(applicable, left.shape).copy(),
                )
        self.add(family, name + "_difference", left - right)
        if both:
            self.add(family, name + "_sum", left + right)

    def target(self, family: str, name: str, frame: pd.DataFrame) -> None:
        if not frame.index.equals(self.index) or list(frame.columns) != self.pairs.target.tolist():
            raise ValueError("Target feature axes do not align")
        self.counts[family] += len(frame.columns)
        self.add(family, name, frame.to_numpy(dtype=float))

    def context(self, family: str, name: str, series: pd.Series) -> None:
        if not series.index.equals(self.index):
            raise ValueError("Context dates do not align")
        self.counts[family] += 1
        self.add(
            family,
            name,
            np.broadcast_to(series.to_numpy()[:, None], (len(series), len(self.pairs))).copy(),
        )

    def finish(self) -> Panel:
        panel = Panel(
            np.stack(list(self.output.values()), axis=2),
            self.index.tolist(),
            self.pairs.target.tolist(),
            list(self.output),
            dict(self.counts),
        )
        panel.validate()
        return panel
