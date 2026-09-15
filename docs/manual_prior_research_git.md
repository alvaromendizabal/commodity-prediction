# Manual Git publication — separate from experiment execution

This is a **manual, optional publication procedure after the two notebooks finish**. No Git action has been performed by the assistant. Do not commit inside the pinned research checkout: its runtime intentionally requires the old source commit plus explicit untracked manual additions. Use a separate publication worktree and preserve the research directory.

Never stage `data`, `artifacts`, `logs`, `.venv`, private ZIPs, prediction tables, model files, credentials, or full execution reports without a separate privacy/publication review. These steps publish source and output-cleared notebook copies only. Keep the original executed notebooks on EBS and in your private download backup.

## 1. Inspect before changing anything

In the existing JupyterLab terminal:

```bash
ROOT="$HOME/projects/commodity-prediction-manual"
git -C "$ROOT" status --short
git -C "$ROOT" diff --stat
git -C "$ROOT" rev-parse HEAD
```

HEAD must be `d142a4cb57a5c4b2880f9341619e13a735b1cddc` for the current experiment contract. Untracked manual helpers/notebooks are expected. Unexpected tracked changes are a reason to stop, not run reset or clean.

## 2. Create a publication worktree without changing the research checkout

Only do this once. A naming collision must be reviewed; do not delete an existing worktree to repeat it.

```bash
git -C "$HOME/projects/commodity-prediction-manual" worktree add \
  -b research/manual-prior-rounds-16-17 \
  "$HOME/projects/commodity-prediction-publish-16-17" \
  d142a4cb57a5c4b2880f9341619e13a735b1cddc
```

## 3. Copy only the declared source dependencies and notebooks

The new helper reuses earlier manual modules that may not yet be on GitHub. The explicit allowlist below includes those source dependencies, not private data. The command strips outputs only in the **new publication copies**. It leaves the executed research originals intact.

```bash
python3 - <<'PY'
from pathlib import Path
import json, shutil
src=Path.home()/'projects/commodity-prediction-manual'
dst=Path.home()/'projects/commodity-prediction-publish-16-17'
helpers=['commodity_first_fold.py','commodity_feature_round.py',
'commodity_session_ablation.py','commodity_close_network.py',
'commodity_feature_diagnosis.py','commodity_rank_state_ablation.py',
'commodity_target_context.py','commodity_response_encoding.py',
'commodity_information_audit.py','commodity_two_feature_rounds.py','commodity_prior_research.py']
paths=['scripts/'+name for name in helpers]+[
'tests/test_manual_prior_research.py',
'configs/manual_prior_replication.json','configs/manual_event_history.json',
'docs/manual_prior_replication.md','docs/manual_event_history.md',
'docs/manual_prior_research_git.md','reports/manual_feature_registry.json']
for rel in paths:
    a,b=src/rel,dst/rel
    if not a.is_file() or a.is_symlink(): raise SystemExit('STOP: missing or unsafe source '+rel)
    if b.exists() and b.read_bytes()!=a.read_bytes(): raise SystemExit('STOP: publication collision '+rel)
# Preflight complete before copying.
for rel in paths:
    a,b=src/rel,dst/rel; b.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a,b)
for name in ['16_prior_dynamics_replication.ipynb','17_event_history_ablation.ipynb']:
    a,b=src/'notebooks'/name,dst/'notebooks'/name
    if b.exists(): raise SystemExit('STOP: notebook publication copy already exists: '+str(b))
    nb=json.loads(a.read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code': cell['execution_count']=None;cell['outputs']=[]
    b.parent.mkdir(parents=True,exist_ok=True)
    b.write_text(json.dumps(nb,indent=1)+'\n')
print('Source copied; research originals unchanged. Review every staged path before committing.')
PY
```

If an explicitly required helper is absent, stop and return the missing filename. Do not guess a substitute or copy every file recursively.

## 4. Run the relevant source tests in the publication copy

This does not run the experiment. Tests use synthetic fixtures, with one tiny repository-model fit and real temporary Parquet I/O.

```bash
cd "$HOME/projects/commodity-prediction-publish-16-17"
PYTHONPATH="$PWD/scripts:$PWD/src" \
  "$HOME/projects/commodity-prediction-current/.venv/bin/python" \
  tests/test_manual_prior_research.py \
  --receipt /tmp/commodity-publication-tests-16-17.json
```

A failure is a publication gate. Do not remove a test to force a green result. The generated receipt is not staged.

## 5. Stage the exact source list; review for secrets and oversized files

```bash
cd "$HOME/projects/commodity-prediction-publish-16-17"
git add -- \
 scripts/commodity_first_fold.py scripts/commodity_feature_round.py \
 scripts/commodity_session_ablation.py scripts/commodity_close_network.py \
 scripts/commodity_feature_diagnosis.py scripts/commodity_rank_state_ablation.py \
 scripts/commodity_target_context.py scripts/commodity_response_encoding.py \
 scripts/commodity_information_audit.py scripts/commodity_two_feature_rounds.py \
 scripts/commodity_prior_research.py tests/test_manual_prior_research.py \
 configs/manual_prior_replication.json configs/manual_event_history.json \
 docs/manual_prior_replication.md docs/manual_event_history.md docs/manual_prior_research_git.md \
 reports/manual_feature_registry.json \
 notebooks/16_prior_dynamics_replication.ipynb notebooks/17_event_history_ablation.ipynb
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git diff --cached
```

Check the content, not just extensions. No private artifacts or credential strings should appear. The preparation registry intentionally says new work is unexecuted until updated from returned receipts; do not change that to successful without reviewing actual evidence.

## 6. Commit/push manually, then inspect PR checks

Only after the relevant tests and your diff review pass:

```bash
git commit -m "Add manual prior-replication and event-history research workflows"
git push -u origin research/manual-prior-rounds-16-17
```

Use your normal GitHub authentication; never paste tokens into chat or command-line literals. Open the repository in your browser and create a pull request from `research/manual-prior-rounds-16-17` to `main`. Inspect all checks and the complete diff. Do not automatically merge this merely because the small source tests passed: the repository's full quality rules and any conflicts still apply.

Publishing this branch does not automatically synchronize other working copies or publish executed evidence. Keep the pinned research checkout unchanged while active studies depend on it. A future reviewed source promotion needs a non-destructive synchronization plan.
