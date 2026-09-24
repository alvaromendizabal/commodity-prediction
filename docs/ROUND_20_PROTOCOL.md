# Round 20 — competitive feature-forecasting gate

## Objective

Test high-value mechanisms from strong public MITSUI solutions against the project's canonical 535-date development protocol without touching the reserved final origins.

This round is independently implemented and adapted to the project's stricter point-in-time rules. It is research integration, not source-code copying.

## Public mechanisms incorporated

1. **6th place (private 0.514)** — recursive many-to-one LSTM forecasting of target-linked underlying market series, a 50-day window, hidden size 256, recursive five-step forecasting, and inference-time correction from released labels. Source: Kaggle, Artem777 / Poidenko Artem.
2. **10th place (private 0.479)** — covariance-regularized cross-sectional rank direction. This round uses Ledoit-Wolf covariance shrinkage fit only on each outer fold's training partition, avoiding outer-validation tuning.
3. **3rd place (private 0.60229)** — target-pair metadata as a deliberate noise-control representation. The full LightGBM/Random-Forest/XGBoost stack is scheduled for the next gate if this round establishes that public mechanisms transfer under our validation.
4. **5th place (private 0.532)** — short-window RNN + single-day MLP with a ranking-aware objective. Scheduled after this CPU gate.
5. **8th place (private 0.490)** — joint Transformer using raw features plus lagged targets. Scheduled after the RNN gate, but our implementation will use horizon-correct released labels rather than a uniform lag shortcut.

## Why this is first

The returned feature-evidence audit shows released-history/released-prior families have some of the largest permutation drops. The 6th-place solution independently reports released-label correction as its largest gain. Testing this mechanism has unusually high expected information value and is cheap enough for the currently running `ml.m5.xlarge` Studio app.

## Validation contract

- Same chronological folds as the repository.
- Fold lengths: 180, 180, 175 = 535 development dates.
- Five-date purge remains in the parent protocol.
- Reserved final assessment origins are not evaluated.
- Scaling/imputation is fit inside the model-selection training segment, then refit on the full outer-training segment for the selected epoch count.
- Released-label correction uses only labels whose `label_date_id` would already have been released at each prediction date.
- The 0.7 model/correction blend is fixed in advance from the public 6th-place description; it is not tuned on the outer fold.
- Candidate comparisons use the exact repository implementation of the competition metric and paired block-bootstrap intervals.

## Experiment variants

- `current_market` — exact canonical S3 replay.
- `mean_rank_prior` — training-only mean cross-sectional rank prior.
- `regularized_kelly_rank_prior` — training-only Ledoit-Wolf covariance-regularized rank direction.
- `released_mean_5` — persistence control using only horizon-correct released labels.
- `lstm_feature_forecast_raw` — closest public-mechanism reproduction.
- `lstm_feature_forecast_log` — project adaptation in log-price space, aligned with target construction.
- each LSTM + fixed 0.7/0.3 released-label correction.
- each corrected LSTM + fixed 50/50 `current_market` ensemble to measure complementary error.
- `current_market_plus_kelly_prior` — fixed 75/25 baseline/prior blend.

## Gates

### Smoke

One fold, log-space LSTM only, hidden size 64, at most 3 epochs. Purpose: correctness only.

Pass if:
- preflight succeeds;
- no NaN/inf predictions;
- baseline fold-0 replay is approximately 0.4033810874;
- all checkpoints/results are written;
- no final holdout is evaluated.

### First-fold research gate

One fold, both raw and log spaces, hidden 256, up to 25 epochs with early stopping and refit.

Continue to all three folds if at least one of these is true:
- a public-inspired candidate beats `current_market` on fold 0;
- a 50/50 ensemble improves the fold-0 metric and has a non-degenerate complementary prediction pattern;
- released-label correction clearly improves its parent LSTM;
- the experiment reveals a correctness or representation issue that materially changes the next design.

Stop rather than scaling if all LSTM variants and ensembles materially underperform and the cheap priors add no useful diversity.

## Compute bound

First-fold full mode performs at most four LSTM fits: inner selection + outer refit for raw space, and the same for log space. No managed SageMaker training job is launched. No S3 writes occur. Models/results stay on the persistent Studio volume.

## Next research gate after a positive result

Implement the 3rd-place target-pair stack with leakage-safe OOF stacking (not in-sample meta-features), beginning with a representative target panel and then full 424-target coverage only if the panel passes. Then add the 5th-place rank-aware RNN/MLP and the 8th-place joint Transformer on a bounded GPU run.

## Smoke outcome — 2026-09-16

The bounded smoke gate completed successfully on fold 0 using the CPU
`commodity-current` environment with PyTorch 2.14.0+cpu.

Operational checks:
- Round 20 focused tests: 4 passed.
- Preflight: passed with no failures.
- Canonical `current_market` replay: 0.40338108742296147, matching the
  preserved fold-0 baseline.
- Completion receipts were written.
- The reserved final assessment remained outside this experiment.

Measured fold-0 official metrics:
- `current_market`: 0.40338108742296147
- `current_market_plus_kelly_prior`: 0.1677076195805656
- `mean_rank_prior`: 0.16566303446271657
- `regularized_kelly_rank_prior`: 0.15602749162871407
- `lstm_feature_forecast_log`: -0.05782789193908985
- `lstm_feature_forecast_log_released_correction`: -0.1079085135749924
- `current_market_plus_lstm_feature_forecast_log_released_correction`:
  -0.08494017801478049
- `released_mean_5`: -0.16700304684401407

Decision:
Do not scale this exact recursive LSTM / released-mean formulation to the
larger first-fold gate. The underperformance is substantial enough that the
next compute is better spent testing a distinct leading representation and
model family. This result rejects this formulation; it does not reject richer
released-history representations already measured elsewhere in the project.

Environment note:
PyTorch was installed into the existing uv-managed `commodity-current`
interpreter with:

    uv pip install --python "$PWD/.venv/bin/python" \
      --index https://download.pytorch.org/whl/cpu \
      'torch>=2.4,<3'

Observed version: `torch==2.14.0+cpu`.
