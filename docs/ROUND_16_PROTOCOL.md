# Round 16 — Prior dynamics: temporal replication and matched ablations

**Preparation status:** source and tests prepared; not executed by the assistant.
**Primary question:** does the exact 24-feature `prior_dynamics_joint` recipe from notebook 15 transfer beyond the middle development period?

## Evidence motivating the round

The supplied notebook-15 report records a middle-period control of 0.16704165319061334 and a joint-panel score of 0.1707359891281436. The matched point gain is 0.0036943359375302687. Its conditional 20-origin block interval is [-0.0208239370634823, 0.030159083666653402]; the simultaneous interval also includes zero. This is a screening lead, not established superiority. All 24 intended additions were admitted. The separate own-history and pool/risk panels scored 0.16363305728838898 and 0.14436630251772978. Their comparison suggests a potential interaction worth replicating, not a demonstrated causal synergy.

The four innovation panels in notebook 14 all underperformed on the middle period. They are not rerun or appended in this round. No current leaderboard or GitHub account state has been verified for this package.

## Representation: keep the promising experiment intact

There are **24 reused representations, zero newly invented feature templates**, and four panels of 4, 12, 12, and 24 inputs. Replacing these features during replication would make the result uninterpretable. Equal depth here means the complete original four-panel ablation, two additional periods, ten inline plots, and the same integrity standards—not forcing novel columns into a replication test.

Six bases, exactly as in notebook 15:

1. `(location63 - location126) / risk126`.
2. `(location126 - location252) / risk126`.
3. `(median63 - location63) / risk63`.
4. `(location126 - market_pair_pool) / risk126`.
5. `(location126 - canonical_cross_horizon_pool) / risk126`.
6. `log(risk63 / risk252)`.

Each base yields its level, five-origin change, surprise relative to the preceding 63 rows (minimum 42), and within-horizon rank. Nonrank forms are clipped to [-12,12]. Rankings are computed before clipping, with at least three finite members and average ties. Invalid/nonpositive risk denominators remain missing.

Panels: risk-only (last base, 4); own-history (first three bases, 12); pool/risk (last three, 12); exact joint (24). Baseline features are not removed or altered. Fit-time constants, excessive missingness and exact duplicates may be excluded using training data only; the report records every exclusion. A missing family is not silently replaced by a validation-selected alternative.

## Temporal comparison

| Fold | Training stop, exclusive | Validation origins | Count | Historical baseline |
|---|---:|---|---:|---:|
| 0 | 1164 | 1169–1348 | 180 | 0.40338108742296147 |
| 1 | 1344 | 1349–1528 | 180 | 0.16704165319061334 |
| 2 | 1524 | 1529–1703 | 175 | 0.39061886486364056 |

The four fold-1 models are reused and replayed against their saved predictions; no fold-1 refit occurs. Four variants in each of folds 0 and 2 give **at most eight new fits**. All three original baseline models are replayed, never refitted. The 355 first/last-period observations are called non-selection origins, **not an untouched test set**. All development periods have previously informed the wider research. In particular, the first fold predates the selection period, so this is retrospective temporal robustness research, not a prospective pre-registration claim.

Target normalization, model hyperparameters, seeds, date weighting and preprocessing remain those of the existing histogram learner. Training excludes each fold's five-row purge gap. Source remains at the supplied revision `d142a4cb57a5c4b2880f9341619e13a735b1cddc`. Existing cached prior inputs retain the horizon+1 release assumption. New prefix tests do not independently establish the correctness of that historical cache.

## Correctness gates before fitting

The user runs the prepared tests in the existing pinned interpreter. The preflight checks real Parquet I/O, a tiny real repository-model synthetic fit and serialization replay, formula parity with the previous helper, Copy-on-Write/read-only inputs, chronological prefix equality, path safety, hashes, and bounded subprocess shutdown. No private forecasting fit is part of preflight.

The real worker then verifies the locked runtime, source, raw input hashes, parent manifests, both returned report hashes and the previous completed round. The new prior cube must exactly match the saved notebook-15 cube through origin 1528 and the old formula implementation. Saved baseline and middle-period predictions must replay exactly before new fits. The final-test boundary remains closed. Raw tables are parsed only up to origin 1703; full-file integrity hashing does not evaluate held-out values.

## Budget and stopping

One user-started worker: **360 seconds maximum cumulative analytical time**, eight new private fit attempts maximum, four numerical threads. Linux resident-memory monitoring stops the worker above 14 GiB. The existing 4-vCPU/16-GiB-class space is the assumption; other busy kernels reduce available memory. Require at least 2 GiB free disk. RSS sampling and these limits are safeguards, not a measured peak-memory guarantee.

Each successful fit gets a model, predictions, result and completion manifest. A user-requested planned pause after a sealed checkpoint is explicitly resumable and carries forward consumed time. An error or hard timeout is not automatically retried. Partial work remains intact for diagnosis. A shared lock prevents either new worker from running alongside the other.

The two predefined periods are part of this one replication question; the code does not tune the last-period recipe based on first-period results. A negative first replication period is not a reason to rewrite the plan. A correctness/resource failure is a stop condition.

## Metric and decision

Pooled official metric: mean of the concatenated 535 daily cross-sectional Spearman correlations divided by their population standard deviation, not the mean of fold ratios. The same calculation is made on the 355 non-selection origins. All IDs, counts, feature names and finite predictions are checked.

The designated joint recipe earns review only if it:
- has positive pooled gain over the original baseline;
- has positive gain on the 355 non-selection origins;
- improves at least two of three folds;
- beats both 12-feature half-panels on the pooled metric;
- admits all intended features and passes exact integrity/replay checks.

These are descriptive review rules chosen after the middle-fold result. No automatic champion switch or holdout access follows. Paired circular block intervals use 10, 20, 40 origins and the existing 2,000-repetition configuration, resampling each fold separately. They condition on fitted models and do not correct the full adaptive search history. Middle-fold model reuse is not independent evidence.

## Presentation and artifacts

Notebook 16 has ten explicit inline Plotly display cells: selection-period evidence, three-period score matrix, pooled scores, non-selection gains, fold gains, uncertainty, temporal concentration, feature admission, coverage, and compute accountability. Charts distinguish model reliance, predictive performance and cost; cumulative correlations are not profits.

Private checkpoints are under `artifacts/manual_prior_research/prior_replication/<lineage>`. Reports/dashboard/ZIP are under `logs/manual_prior_replication`. The ZIP contains new models and derived features, not old model copies or raw CSVs. Download it privately; it is not off-disk protection while it remains on the EBS volume. Preserve the separate notebook-15 backup as the middle-fold dependency.

## Sources and interpretation

- The supplied `commodity_prior_dynamics_report.json` and `commodity_innovation_ablation_report.json` are the evidence for the numerical claims above.
- Competition primary page: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation
- Participant adaptation discussion: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th
- Plotly inline rendering: https://plotly.com/python/renderers/

The participant's online model retraining is not implemented here: this round changes neither the forecasting algorithm nor validation-time fitted model. The strongest valid comparable Kaggle performance remains the research target; neither a local ratio nor these intervals establish that a leaderboard winner has been surpassed.


## This release
The analytical recipe is unchanged. The heartbeat recovery is described in RECOVERY_AND_RUN_ORDER.md. Earlier execution evidence does not establish a pass for the new source/test revision.
