# Round 21A — Documented 3rd-Place Reproduction

## Objective

Recreate the publicly documented 3rd-place MITSUI strategy in our own code, preserve
its strategy as a standalone lineage, and compare it against the canonical
`current_market` baseline without touching the final reserved origins.

Historical reference: Ayush Khaire, 3rd place, private leaderboard 0.60229.

## Publicly documented strategy reproduced

- one prediction problem per target;
- route features from `target_pairs.csv`;
- use only the constituent asset(s) for the source-described core;
- lag features;
- rolling mean/max features;
- difference features;
- safe logarithmic transform;
- median imputation;
- standardization;
- LightGBM, Random Forest, XGBoost base regressors;
- XGBoost meta-regressor;
- source-described in-sample base predictions as meta-features.

## Critical reproducibility distinction

The writeup does not publish all exact model hyperparameters or exact rolling-window
sizes. Those are not invented and mislabeled as source parameters. They are recorded in
`configs/third_place_reproduction.json` under
`source_unspecified_reconstruction_choices`.

The writeup also describes negative lags. Literal `shift(-k)` is future-looking at an
offline prediction origin. The code preserves an explicit forensic constructor for that
description, but the default runner refuses to train it. The promotable reproduction is
strictly causal.

## Two stacks

`source_stack` reproduces the documented model-engineering structure: base learners are
trained on the outer-training data and their in-sample predictions train the XGBoost
meta-model.

`causal_oof_stack` uses the same model families but replaces in-sample meta-features
with chronological OOF predictions separated by a gap.

This directly tests whether the source stacking choice transfers or whether proper OOF
stacking generalizes better.

## Bounded modes

- `smoke`: 8 deterministic stratified targets.
- `panel`: 64 deterministic stratified targets.
- `full`: all 424 targets.

Unmodeled targets in smoke/panel retain the exact existing `current_market` fold
predictions, so the reported official metric is still calculated over all 424 targets.

The panel is a compute gate, not a permanent target or feature cap.

## Acceptance criteria for smoke

- focused unit tests pass;
- target reconstruction matches official labels within numerical tolerance;
- causal feature prefix invariance passes;
- `current_market` fold-0 metric replays near 0.40338108742296147;
- all model predictions are finite;
- `DONE.json` and `summary.json` are written;
- `reserved_final_origins_evaluated` is false.

Do not run `panel` until the smoke evidence is returned and reviewed.


## Canonical notebook-first execution

The canonical employer-facing experiment is:

`/home/sagemaker-user/projects/commodity-prediction-current/notebooks/21_third_place_reproduction.ipynb`

It performs feature creation, model fitting, stacking, scoring, Plotly diagnostics and receipt creation directly.

For terminal-only execution, run the notebook through:

`/home/sagemaker-user/projects/commodity-prediction-current/scripts/execute_round21a_notebook.py`

Smoke command:

```bash
cd /home/sagemaker-user/projects/commodity-prediction-current

PY="/home/sagemaker-user/projects/commodity-prediction-current/.venv/bin/python"

timeout 15m "$PY"   /home/sagemaker-user/projects/commodity-prediction-current/scripts/execute_round21a_notebook.py   --mode smoke   --cell-timeout-seconds 600
```

The outer `timeout 15m` is the hard milestone wall-clock bound. The script atomically
replaces the canonical notebook only after successful execution.
