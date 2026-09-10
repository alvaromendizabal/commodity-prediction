#!/usr/bin/env bash
# SageMaker lifecycle entry point: retain its refreshable role-provider environment.
set -euo pipefail
test "$#" = 1
expected_commit="$1"
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
sudo --preserve-env=AWS_CONTAINER_CREDENTIALS_FULL_URI,AWS_CONTAINER_CREDENTIALS_RELATIVE_URI,AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE,AWS_CONTAINER_AUTHORIZATION_TOKEN,AWS_REGION,AWS_DEFAULT_REGION \
    -u sagemaker-user /bin/bash --noprofile --norc -s -- "$expected_commit" <<'COMMODITY'
set -euo pipefail
cd /home/sagemaker-user/projects/commodity-prediction
test "$(git rev-parse HEAD)" = "$1"
mkdir -p logs
.venv/bin/python scripts/feature_publication.py --preflight-only
nohup bash scripts/publish_feature_research.sh >> logs/compact-publication.log 2>&1 < /dev/null &
printf 'Publication supervisor launched at verified commit %s; inspect durable status for progress.\n' "$1"
COMMODITY
