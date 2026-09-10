"""Replay every saved domain fit and control from its sealed inputs on AWS."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from commodity_prediction.domain.run import load_panel
from commodity_prediction.runtime import RunLog, verify_checkpoint


def replay(root: Path, report: dict, log: RunLog) -> dict:
    directory = root / "artifacts" / report["lineage"]
    maximum = 0.0
    models = sorted(directory.rglob("model.joblib"))
    with log.stage("cloud_domain_model_replay"), threadpool_limits(limits=4):
        verify_checkpoint(directory / "features", report["lineage"])
        panel = load_panel(directory / "features")
        for number, path in enumerate(models, start=1):
            verify_checkpoint(path.parent, report["lineage"])
            result = json.loads((path.parent / "result.json").read_text())
            model = joblib.load(path)
            for name, weight in [("predictions.parquet", None), ("raw_predictions.parquet", 1)]:
                expected_path = path.parent / name
                if not expected_path.exists():
                    continue
                expected = pd.read_parquet(expected_path)
                actual = model.predict(
                    panel, result["validation_start"], result["validation_stop"], weight=weight
                )
                actual.index.name = expected.index.name
                pd.testing.assert_index_equal(expected.index, actual.index)
                pd.testing.assert_index_equal(expected.columns, actual.columns)
                error = float(np.max(np.abs(expected.to_numpy() - actual.to_numpy())))
                if error > 1e-12:
                    raise ValueError(f"Domain checkpoint replay differs: {path.parent.name}")
                maximum = max(maximum, error)
            if number % 20 == 0:
                log.event("cloud_domain_replay_progress", completed=number, total=len(models))
    if len(models) != report["new_fitted_models"] + report["controls"] * 3:
        raise ValueError("Incomplete domain model checkpoint inventory")
    return {"model_checkpoints_replayed": len(models), "maximum_prediction_replay_error": maximum}
