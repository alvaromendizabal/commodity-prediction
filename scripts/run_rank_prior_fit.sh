#!/bin/bash
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"

: "${RANK_PRIOR_MODE:=probe}"
case "$RANK_PRIOR_MODE" in
  probe)
    timeout --signal=TERM --kill-after=10s 120s .venv/bin/python -u -m commodity_prediction.domain.rank_prior_fit.run --first-fold --sync-s3
    ;;
  full)
    timeout --signal=TERM --kill-after=10s 210s .venv/bin/python -u -m commodity_prediction.domain.rank_prior_fit.run --sync-s3
    ;;
  *)
    echo "Unknown RANK_PRIOR_MODE=$RANK_PRIOR_MODE" >&2
    exit 2
    ;;
esac
