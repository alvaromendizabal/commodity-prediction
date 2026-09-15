# Round19 — delayed error memory of a causal reference prior

## Question
Does a target's sequence of already observable prior errors describe changing bias/risk more usefully than another raw-return moment? This is distinct from round17's raw outcome distribution and round12's signal–outcome regression features.

## Critical alignment
For original prediction origin s, form `e[s]=(y[s]-location_126[s])/risk_126[s]`. The mean and risk are the existing causal released-history priors at **s**, not priors at the later time when the outcome became known. Clip e to[-12,12], then release that complete error at s+h+1. Missing outcomes or invalid reference scales produce missing errors.

These are errors of a deterministic historical prior, not in-sample residuals of the learned histogram model. The forecasting model is not retrained as later validation outcomes become available. New feature states may update only when the appropriate historical outcome has been released.

## 24 features and controlled comparisons
Across21/63/126-row windows: observed-error coverage; mean observed release-row age; mean error; median error; sign balance; mean absolute error; root mean squared error; fraction with |standardized prior error|>2. All summaries require an adequate valid history; no unavailable label is imputed as zero.

Timing control6; timing+bias15; timing+dispersion15; joint24. Coverage is a separate comparator, not automatically credited as numerical forecast information. Threshold2 and clipping12 are fixed investigational choices, not tuned optimal constants.

The six distribution/bias summaries are the new representation; three windows do not make them independent sources. Training-only duplicate and coverage audits are required. Final feature admission and uncertainty are reported.

## Fixed experiment
Original saved current_market control, last development fold1529–1703, original histogram configuration. Up to4 new fits with a240-second total worker allowance,14GiB monitored RSS,15-second heartbeats and sealed model/prediction stages. Conditional block intervals and matched deltas versus original/clock/half-panel controls. All required gains>=0.002; no automatic promotion or new fold sweep.

## Research basis and limits
Gama et al. (2014), *A survey on concept drift adaptation*, https://doi.org/10.1145/2523813 , describes supervised-learning relationships changing over time and evaluation of adaptive processes. It motivates monitoring error history; it does not validate this exact error-memory representation or establish its commodity value.

The prior sequence is part of the trusted cached pipeline, whose release contract is inherited. New prefix tests check this new layer, not independently every parent feature. Development periods are repeatedly inspected; a passing screen is not a leaderboard victory.

Prepared static implementation only. The user executes tests and experiments. No account operations, external-data joins, model tuning, ensemble weighting, or final-holdout evaluation are included.
