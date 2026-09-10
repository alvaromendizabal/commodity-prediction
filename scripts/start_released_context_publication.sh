#!/usr/bin/env bash
set -euo pipefail
expected_commit="${1:?Pass the reviewed source commit}"
sudo --preserve-env=AWS_CONTAINER_CREDENTIALS_FULL_URI,AWS_CONTAINER_CREDENTIALS_RELATIVE_URI,AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE,AWS_CONTAINER_AUTHORIZATION_TOKEN,AWS_REGION,AWS_DEFAULT_REGION \
  -u sagemaker-user /bin/bash --noprofile --norc -s -- "$expected_commit" <<'COMMODITY'
set -euo pipefail
expected_commit="$1"
project=/home/sagemaker-user/projects/commodity-prediction
prior=/home/sagemaker-user/projects/commodity-prediction-publication
publication=/home/sagemaker-user/projects/commodity-prediction-context
git -C "$project" fetch origin "$expected_commit"
test "$(git -C "$project" rev-parse FETCH_HEAD)" = "$expected_commit"
if [[ ! -e "$publication" ]]; then
    git -C "$project" worktree add --detach "$publication" "$expected_commit"
fi
cd "$publication"
test "$(git rev-parse HEAD)" = "$expected_commit"
for directory in data .venv; do
    test -d "$project/$directory"
    if [[ ! -e "$directory" ]]; then ln -s "$project/$directory" "$directory"; fi
done
if [[ ! -e artifacts ]]; then cp -al "$prior/artifacts" artifacts; fi
export PYTHONPATH="$publication/src:$publication" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
nohup bash -c 'exec 9>logs/released-context-publication.lock; flock -n 9 || exit 75; exec timeout 420 .venv/bin/python scripts/publish_released_context.py' \
    >> logs/released-context-publication.log 2>&1 < /dev/null &
printf 'Started bounded context publication process %s from %s\n' "$!" "$expected_commit"
COMMODITY
