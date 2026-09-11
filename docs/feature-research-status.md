# Feature research: measured state and next experiments

## Evidence boundary

The highest reproduced fitted development point estimate in the inspected reports is
`current_market`: **0.3097087232124053** over 535 development dates. It is not a
leaderboard score. No inspected execution receipt supports a **0.62** result for
this project. An additional result must be identified by its metric, dates,
source commit and replayable predictions before replacing the recorded leader.
The historical leaderboard target recorded in `docs/competitive-research.md` is
not a like-for-like estimate of a gap on our development folds.

Feature engineering remains open. No final model is promoted and the 247 final
origins remain gated. Passing tests or merging source does not prove feature value.

## What has been measured, and what has not

| Mechanism | Evidence | Decision |
| --- | --- | --- |
| Current OHLC/activity | Development leader 0.309709; matched gain remains uncertain | Preserve as frozen control, not final winner |
| Four-date price/activity paths | Published matched study; longer price/joint paths underperformed | Do not expand unchanged |
| Signed released peer history | Published three-fold study; underperformed | Do not repeat unchanged |
| Static diagonal target-rank feature | PR #9 execution receipt: 0.301772 vs 0.305336 and 0.307373 vs 0.309709 | Reject this fitted scalar transfer; richer grouping remains a separate hypothesis |
| Volatility-normalized OHLC, abnormal-volume confirmation, exact union | PR #10 implemented 7/7/14 templates and an experimental gate | Predictive value unmeasured; private experiment still required |

The rank-prior numerical receipt is on `feat/rank-prior-research`, at
`reports/rank_prior_fit_execution.json` (study source `50616f6558abeda63a960bb36dff85b16a6b1481`).
PR #9 remained draft at this audit. Recovery/publication of its already-executed
notebook is separate from the completed six-model experiment. Do not retrain it.

## Adversarial validation of the pending normalization family

The expanded synthetic suite found a numerical representation defect before
private training: positive infinite volume passed the nonnegative check, then
clipping and `tanh` turned it into a finite confirmation signal. Infinite highs
could similarly survive OHLC ordering checks, and an infinite previous close
could become an overnight value of -12 after clipping.

Nonfinite prices and volumes are now masked **before** logs, trailing statistics,
ratios and clipping. A supported-but-invalid observation remains missing rather
than becoming a structural zero. Duplicate market columns and empty, incomplete
or unsupported target-horizon metadata fail with explicit errors.

The expanded suite initially produced **12 failures and 18 passes** against the
parent feature implementation. After correction all **30 tests passed locally**,
including the original 11. Clean synthetic feature arrays remained exactly equal
for all three variants, including NaNs. Prefix-at-origin replay also matched the
same origins computed from the longer input table.

These are adversarial synthetic checks, not evidence that real competition rows
contain infinities or that removing them improves the official score. The research
configuration, window lengths, clipping limits, candidate counts, parent source
and model settings have not changed. The child source fingerprint changes; do not
reuse any artifact with a mismatched child lineage.

## Immediate compute gate

Reuse the existing `current_market` controls; perform no control refits. Restore
and validate private input schemas, sealed parent summaries, model hashes and
prediction replay before the first new fit. Then run the already-declared three
first-fold variants. Continue to the remaining six fits only if the predeclared
0.002 gain gate passes and integrity checks succeed. The cumulative fitting-study
limit stays 300 seconds. Preserve negative results and do not tune the threshold
after seeing a score. Per-fold metrics, uncertainty, checkpoint hashes and actual
fit counts are required before claiming completion.

The latest connected AWS read attempt in this audit returned **Resource not
found** at the connector layer. Rediscovery exposed no AWS execution action;
plugin-directory search for AWS returned no eligible plugin. This does not prove
the user's account is disconnected. It means this session could not inspect or
modify the space. No live compute status, S3 write, private experiment or EBS
synchronization is claimed here. Earlier S3 receipts are historical, not live checks.

## Competitive representation research remains necessary

Primary competitor reports motivate separate, testable learned representations:

- ZLF's fifth-place writeup reports a four-day RNN and single-day MLP using raw
  inputs, with a joint MSE/ranking loss and averaged outputs. Its reported score
  is not an evaluation on our folds.
- The eighth-place Transformer writeup reports joint prediction of 424 targets
  from raw tabular features and historical target features. Its feature-count
  narrative and uniform lag description are not an implementation contract for
  this repository: maintain the repository's horizon-specific release delay.

Primary sources reviewed on 2026-09-11:

1. https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/zlf-solution-of-the-mitsui-commodity-predict
2. https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/transformer-based-solution
3. https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation

Proposed next comparison, **not implemented or executed by this audit**: freeze a
small shared multi-target architecture and compare raw-market inputs against the
same inputs augmented with horizon-correct released-label context. Keep model,
loss, training schedule and seeds matched so the representation is the ablation.
Compare a ranking-aware loss separately; do not attribute a simultaneous feature,
architecture and loss change solely to feature engineering. Begin with shape,
missing-label and future-perturbation tests, then a bounded first-fold run.

Richer target/group structure, online adaptation using actually released labels,
and externally mapped carry, inventories, positioning, macro vintages and weather
remain open only where availability, instrument identity and data rights can be
verified. Do not treat many candidate columns as evidence that these are exhausted.
The objective remains the strongest comparable performance, not a promised record.
