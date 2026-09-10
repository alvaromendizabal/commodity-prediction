#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p logs
# SageMaker Distribution 4.4 uses Ubuntu 24.04 and omits Chrome's shared libraries.
# Install rendering dependencies before executing the canonical notebooks.
sudo -n apt-get update -qq
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libglib2.0-0t64 libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcups2t64 libdbus-1-3 libxcb1 libxkbcommon0 libatspi2.0-0t64 libx11-6 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 \
    libcairo2 libpango-1.0-0 libasound2t64 fonts-liberation
if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen --extra dev
uv run --frozen python -m ipykernel install --user --name commodity --display-name 'Commodity Research (Python 3.12)'
uv run --frozen python scripts/quality.py
uv run --frozen commodity bootstrap
uv run --frozen python scripts/restore_checkpoint.py
uv run --frozen commodity research --sync-s3
uv run --frozen python -m commodity_prediction.studies.run --sync-s3
uv run --frozen python -m commodity_prediction.domain.run --sync-s3
uv run --frozen python -m commodity_prediction.domain.attribution.run --sync-s3
uv run --frozen kaleido_get_chrome
uv run --frozen python scripts/make_notebooks.py
uv run --frozen python scripts/execute_notebooks.py
uv run --frozen python scripts/verify_bootstrap.py
