# Round 17 — Observed-event histories and observation-clock controls

**Preparation status:** implementation and tests prepared, not run by the assistant.
**Question:** do fixed-count histories of actually released target outcomes add information beyond calendar-row summaries and explicit observation timing?

## Motivation and difference from earlier features

The current model uses already-released target-history priors. The user's information audit reported a noticeable loss when those histories were time-shuffled. That does not prove any newly derived feature will work, but identifies a substantive input family.

Existing rolling/EWM priors summarize a fixed span of rows or exponentially weighted row history. When observations are missing, the amount of actual evidence in a span can differ across targets. This recipe instead keeps the **last K observed, legally released outcomes**. A summary based on eight outcomes may span more than eight rows; age/gap descriptors make that distinction visible. They are row counts, not verified physical dates or publication timestamps.

Earlier static rank priors, ordinal states, signed peer summaries and signal/outcome response encodings are not repeated unchanged. This experiment uses count-based order statistics and shape summaries of each target's own released outcomes. Some features can still overlap or correlate with existing priors; the training-only audit records exact duplicates rather than claiming guaranteed novelty for every column.

Che et al. (2018) explicitly study informative missingness and elapsed-time/mask information in multivariate time series. Their empirical setting is healthcare and their GRU-D architecture is not used here. It motivates testing observation timing, not a claim that missing financial outcomes are predictive. A first-hand Mitsui solution describes using newly available labels for adaptation; we retain a frozen forecaster and adapt only deterministic feature states.

## Exact 24-representation inventory

Windows **K = 8, 21, 63 observed outcomes**, fixed in advance. A window must contain all K valid outcomes. A genuine zero is valid. `-999999`, NaN and infinities are not events. No missing outcome is zero-filled or forward-filled.

For each K:

| Group | Feature | Formula / interpretation |
|---|---|---|
| Location | Mean/risk | Event mean divided by the existing released risk126 at prediction time |
| Location | Median/risk | Event median divided by that same risk |
| Location | Latest innovation/risk | Latest released outcome minus event mean, divided by risk |
| Shape | Positive-minus-negative balance | Mean sign; zeros contribute zero, denominator includes them |
| Shape | Tail asymmetry | `(q90 + q10 - 2*median)/(q90-q10)`; zero for degenerate identical quantiles |
| Shape | Semivariance balance | `(sum positive squared outcomes - sum negative squared outcomes) / sum all squares`; zero if all outcomes are zero |
| Clock | Mean release age | `log1p(mean(t - release_row)/K)`, clipped to [0,12] |
| Clock | Longest release gap | `log1p(max(largest within-window release gap, current staleness)/K)`, clipped to [0,12] |

Location features require a finite positive risk126 >1e-12 and are clipped to [-12,12]. Shape summaries are bounded in [-1,1]. Clock features describe the observation sequence, not estimator standard errors or effective independent sample sizes. Overlapping multi-horizon outcomes remain dependent; K is not an independent-observation count.

Total: **9 location + 9 shape + 6 timing = 24 new representations**. Four fitted panels:
- clock only: 6;
- clock + location: 15;
- clock + shape: 15;
- exact union: 24.

Numerical panels must beat the clock-only control so availability cannot be mistaken for extra predictive content. This is not an expansion or reuse of a winner from round 16; round 17 can finish regardless of round 16's scientific sign.

## Point-in-time construction

For horizon h, an outcome originating at s enters the event stream at **s+h+1**. At prediction origin t only events with release row <=t are eligible. Recent rows whose outcomes are not yet released cannot alter any window, quantile, sign, or gap statistic. The feature builder does not select parameters from validation outcomes.

The feature values can change as earlier validation outcomes are released. The forecast models and feature panels do not retrain or reselect during validation. The existing risk denominator is inherited from the cached causal parent at t, not estimated from full validation data. This inheritance remains a documented limitation; a prefix test is not a full audit of the historical parent.

Tests supplied for user execution cover each of four release boundaries, an independent observed-event mean/median calculation, sparse events, zero outcomes, sentinels, read-only inputs, future perturbations, prefix equality, constant distributions, sign/scale behavior, gap clocks, missing risk, invalid horizons, and invalid shapes. The notebook's real worker repeats exact prefix comparisons on actual development inputs before fitting.

## Model comparison and validation

Original saved `current_market` control; **no round16 features or model selection**. First-fold fitting stops at 1164, warmup252, validation1169–1348 (180 origins), expected baseline0.40338108742296147. The original forecast learner, normalization, sample weights and fitting settings remain fixed. Parent masks and imputation retain existing behavior.

All three development periods have been examined during prior research. This first period is an exploratory screen, not an untouched holdout. The pooled baseline0.3097087232124053 is not the comparison for this 180-origin experiment. No final evaluation is read or scored. Raw tables are parsed only through1348; raw-file checksums may stream full bytes solely for integrity.

Panels are declared before these new outcomes. Fit-time missingness/constant/duplicate checks and descriptive two-half associations use only the fitting prefix. No association-driven replacement, window sweep, or parameter tuning occurs. If no added feature survives screening, that panel is recorded as skipped, not refitted with an improvised substitute.

## Resources, persistence and limits

At most **four new private forecasting fits**, one **240-second** cumulative worker budget, four numerical threads, 14-GiB monitored RSS ceiling, at least2GiB free disk. These are ceilings, not measured runtimes/peaks. Each successful model is persisted with predictions/result/hash manifest. Features are separately saved. The old baseline is replayed exactly, not refitted.

A shared lock prevents parallel workers. Completed compatible results can be reopened without fits. Intentional checkpoint pauses can be explicitly resumed using the remaining allowance. A failure or timeout stops for diagnosis; it cannot automatically restart or renew its budget. Backup/presentation failures do not authorize training again. No package installation, cloud/Git write, submission, or automatic instance shutdown occurs.

## Decision and interpretation

Clock-only must improve the saved baseline by at least0.002. Location/shape require at least0.002 over both baseline and clock-only. Joint requires at least0.002 over baseline, clock-only, location and shape panels. All intended inputs and exact replay/integrity checks are required for the review gate. Negative or inconclusive findings are preserved. A clock-only gain cannot be credited to outcome magnitudes.

Conditional paired block intervals10/20/40, 2,000 resamples, include the declared comparisons. They do not compensate for the complete adaptive feature search. The gate allocates further research compute and never promotes a model automatically.

Ten inline plots: training coverage, observation clocks, training-half associations, scores, comparator deltas, intervals, temporal concentration, admission, horizon diagnostics, compute accountability. The HTML file supplements rather than replaces inline output. Plots use actual user-executed results only.

## Primary sources, with limitations

1. Che, Purushotham, Cho, Sontag & Liu (2018), *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports8:6085. https://doi.org/10.1038/s41598-018-24271-9 . Supports explicit representation of missingness/time gaps in their setting. No commodity-return claim and no reproduction of GRU-D.
2. *MITSUI&CO. Commodity Prediction Challenge — 15th Place Solution Writeup*. https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th . Supports adaptation as a competition research lead, not this exact event-window recipe or our validation.
3. Official competition overview: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation . Local cross-validation and the competition's real-market forecasting phase are distinct evaluation settings.
4. Plotly renderers: https://plotly.com/python/renderers/ . The notebooks explicitly show MIME-bundle figures inside JupyterLab.

Public leaderboard values could not be extracted from the public page during preparation. No new winning-score verification or leaderboard-equivalence claim is made. Historical benchmark entries need a preserved official receipt before they are used as a verified comparator.
