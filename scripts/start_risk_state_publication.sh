#!/usr/bin/env bash
# Keep the existing checkout intact and reuse its private data and environment.
set -euo pipefail
expected_commit="${1:?Pass the reviewed source commit}"
sudo --preserve-env=AWS_CONTAINER_CREDENTIALS_FULL_URI,AWS_CONTAINER_CREDENTIALS_RELATIVE_URI,AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE,AWS_CONTAINER_AUTHORIZATION_TOKEN,AWS_REGION,AWS_DEFAULT_REGION \
  -u sagemaker-user /bin/bash --noprofile --norc -s -- "$expected_commit" <<'COMMODITY'
set -euo pipefail
expected_commit="$1"
project=/home/sagemaker-user/projects/commodity-prediction
publication=/home/sagemaker-user/projects/commodity-prediction-publication
cd "$project"
git fetch origin "$expected_commit"
test "$(git rev-parse FETCH_HEAD)" = "$expected_commit"
if [[ ! -e "$publication" ]]; then
    git worktree add --detach "$publication" "$expected_commit"
fi
cd "$publication"
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
    git diff --quiet
    git diff --cached --quiet
    git merge --ff-only "$expected_commit"
fi
test "$(git rev-parse HEAD)" = "$expected_commit"
for directory in data .venv; do
    test -d "$project/$directory"
    if [[ ! -e "$directory" ]]; then ln -s "$project/$directory" "$directory"; fi
done
# Archive restoration intentionally rejects symlinks leaving the checkout.
# Hard-link the immutable historical files on the same persistent EBS volume;
# directories and new outputs remain local to this worktree.
if [[ -L artifacts ]]; then
    test "$(readlink -f artifacts)" = "$project/artifacts"
    rm artifacts
fi
if [[ ! -e artifacts ]]; then cp -al "$project/artifacts" artifacts; fi
export PYTHONPATH="$publication/src:$publication" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
nohup bash -c 'exec 9>logs/risk-state-publication.lock; flock -n 9 || exit 75; exec .venv/bin/python scripts/publish_risk_state.py' \
    >> logs/risk-state-publication.log 2>&1 < /dev/null &
printf 'Started bounded notebook publication process %s from %s\n' "$!" "$expected_commit"
COMMODITY
