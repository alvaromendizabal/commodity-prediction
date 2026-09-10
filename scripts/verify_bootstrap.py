"""Publish concrete AWS bootstrap evidence after every required stage succeeds."""

import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from notebook_support import checked_domain, checked_reports, checked_study
from threadpoolctl import threadpool_limits
from verify_domain import replay as replay_domain

from commodity_prediction.cloud import client
from commodity_prediction.runtime import RunLog, atomic_json, digest, verify_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    audit, report = checked_reports(root)
    study = checked_study(root)
    domain = checked_domain(root)
    final_evaluation = json.loads((root / "configs/final_evaluation.json").read_text())
    stages = []
    for lineage in [report["lineage"], study["lineage"], domain["lineage"]]:
        run = root / "artifacts" / lineage
        for manifest in sorted(run.rglob("manifest.json")):
            verify_checkpoint(manifest.parent, lineage)
            stages.append(str(manifest.parent.relative_to(root)))
    log = RunLog(root / "logs/cloud_verification.jsonl")
    maximum_replay_error = 0.0
    with log.stage("cloud_model_replay"), threadpool_limits(limits=4):
        features = pd.concat(
            [
                pd.read_parquet(
                    root / "artifacts" / report["lineage"] / "features/candidates.parquet"
                ),
                pd.read_parquet(
                    root / "artifacts" / study["lineage"] / "features/candidates.parquet"
                ),
            ],
            axis=1,
        )
        for number, result in enumerate(study["results"], start=1):
            directory = (
                root / "artifacts" / study["lineage"] / f"fold_{result['fold']}" / result["variant"]
            )
            expected = pd.read_parquet(directory / "predictions.parquet")
            bundle = joblib.load(directory / "model.joblib")
            actual = bundle.predict(features.loc[expected.index])
            pd.testing.assert_index_equal(expected.index, actual.index)
            pd.testing.assert_index_equal(expected.columns, actual.columns)
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
            maximum_replay_error = max(
                maximum_replay_error, float(np.max(np.abs(actual.to_numpy() - expected.to_numpy())))
            )
            if number % 10 == 0:
                log.event("cloud_replay_progress", completed=number, total=len(study["results"]))
    del features
    domain_replay = replay_domain(root, domain, log)
    notebooks = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        if notebook.metadata.get("study_lineage") != study["lineage"]:
            raise ValueError(f"Notebook belongs to a different study: {path.name}")
        if notebook.metadata.get("domain_lineage") != domain["lineage"]:
            raise ValueError(f"Notebook belongs to a different domain study: {path.name}")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                if cell.execution_count is None or any(
                    o.output_type == "error" for o in cell.outputs
                ):
                    raise ValueError(f"Unexecuted or failed notebook: {path.name}")
        notebooks.append(path.name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    evidence = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "source_commit": commit,
        "lineage": domain["lineage"],
        "parent_lineage": study["lineage"],
        "initial_lineage": report["lineage"],
        "checkpoint_count": len(stages),
        "notebooks": notebooks,
        "pooled_templates": domain["templates"],
        "domain_fitted_models": domain["new_fitted_models"],
        "domain_outer_evaluations": domain["outer_evaluations"],
        "previous_fold_experiments_preserved": study["experiments_completed"],
        "initial_fold_experiments_preserved": report["experiments_completed"],
        "model_checkpoints_replayed": len(study["results"])
        + domain_replay["model_checkpoints_replayed"],
        "previous_model_checkpoints_replayed": len(study["results"]),
        "domain_model_checkpoints_replayed": domain_replay["model_checkpoints_replayed"],
        "maximum_prediction_replay_error": max(
            maximum_replay_error, domain_replay["maximum_prediction_replay_error"]
        ),
        "validation_dates": study["validation_dates"],
        "terminal_embargo_dates": study["terminal_embargo_dates"],
        "untouched_final_test_dates": final_evaluation["final_test_dates"],
        "final_test_start_date_id": final_evaluation["final_test_start_date_id"],
        "development_dates": audit["development_dates"],
        "feature_gate": "open",
        "holdout_evaluated": False,
    }
    atomic_json(root / "reports/aws_execution.json", evidence)
    bundle = root / "artifacts/verified-notebooks.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for directory in ["notebooks", "reports", "logs", "configs"]:
            archive.add(root / directory, arcname=directory)
    s3, aws = client(root)
    s3.upload_file(
        str(bundle),
        aws["bucket"],
        "bootstrap/verified-notebooks.tar.gz",
        ExtraArgs={"Metadata": {"sha256": digest(bundle)}, "ServerSideEncryption": "AES256"},
    )
    head = s3.head_object(Bucket=aws["bucket"], Key="bootstrap/verified-notebooks.tar.gz")
    if (
        head.get("Metadata", {}).get("sha256") != digest(bundle)
        or head["ContentLength"] != bundle.stat().st_size
    ):
        raise ValueError("Published notebook archive failed S3 verification")
    s3.put_object(
        Bucket=aws["bucket"],
        Key="bootstrap/verification.json",
        Body=json.dumps(evidence).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
