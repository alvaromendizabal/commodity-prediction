#!/usr/bin/env bash
# Resume verified feature experiments and publish their executed evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
exec 9>logs/feature-publication.lock
flock -n 9 || { printf 'A feature publication process already holds the lock.\n'; exit 75; }
.venv/bin/python scripts/feature_publication.py
