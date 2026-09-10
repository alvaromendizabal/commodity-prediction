#!/usr/bin/env bash
# Resume verified feature experiments and publish their executed evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

publish_log() {
    local pipeline_status=$?
    trap - EXIT
    .venv/bin/python - "$pipeline_status" <<'PY'
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

settings = json.loads(Path("configs/aws.json").read_text())
path = Path("logs/compact-publication.log")
status = {"exit_code": int(sys.argv[1]), "completed_utc": datetime.now(UTC).isoformat()}
s3 = boto3.client("s3", region_name=settings["region"], config=Config(
    connect_timeout=10, read_timeout=60, retries={"mode": "standard", "total_max_attempts": 4}
))
if path.exists():
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    s3.upload_file(str(path), settings["bucket"], "bootstrap/compact-publication.log",
                   ExtraArgs={"Metadata": {"sha256": checksum}})
    status["log_sha256"] = checksum
s3.put_object(Bucket=settings["bucket"], Key="bootstrap/compact-publication-status.json",
              Body=json.dumps(status).encode(), ContentType="application/json")
PY
    exit "$pipeline_status"
}
trap publish_log EXIT

.venv/bin/python scripts/quality.py
.venv/bin/python -m commodity_prediction.domain.compact.run --sync-s3
.venv/bin/python -m commodity_prediction.domain.compact.run
.venv/bin/python scripts/snapshot_domain.py
bash scripts/prepare_rendering.sh
.venv/bin/python scripts/make_notebooks.py
.venv/bin/python scripts/execute_notebooks.py
.venv/bin/python scripts/verify_feature_research.py
