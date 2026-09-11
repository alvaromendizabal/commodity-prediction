# Four-date market paths: a matched representation test

## Research question and overlap audit

The completed released-context study tests labels, not a raw market sequence. The
existing source already contains return lags 0, 1, 2 and 5, risk scaling, OHLC
summaries, activity summaries, backward horizon returns, and released priors.
Therefore, neither recent returns nor generic OHLC information is new here.
The broader intraday family has previously hurt a full-tree point estimate.
The question is narrower: does retaining the order of the last four observed
intraday and activity states help the frozen compact admitted-tail representation?

The prior [competitive audit](competitive-research.md) describes short sequences
in leading competition writeups. This is an inexpensive hypothesis motivated by
that audit, not a reproduction of their neural architecture or leaderboard score.
No new external dataset, real-date mapping, or market-close timestamp is asserted.
The [fixed estimator](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
is unchanged so this study can investigate information representation first.

## Mechanisms and availability

Four price channels retain close/open repricing, opening gaps from the preceding
observed row, high/low log range, and close location within the bar. Two activity
channels retain log-volume surprise against a prior-only 21-row median and the
one-row log-volume change. Each eligible target leg supplies a difference and a
sum, preserving pair orientation and a common-state channel. Applicability counts
separate unsupported fields (structural zeros) from missing observations (NaN).
Every rolling baseline is trailing; every snapshot uses rows t, t-1, t-2 or t-3.
No target label is an argument of the new feature builder. No backward filling,
full-period normalization, or guessed calendar is used.

These signals are proxies for repricing path and participation, not direct
order-flow, physical inventory, delivery-curve carry, or macro surprises. Those
external-data avenues remain prerequisite-blocked rather than experimentally
rejected. Current-row availability follows the project's existing prediction
contract; exact cross-market session timestamps remain unavailable.

## Predeclared experiment

Reuse the exact admitted-tail control and all parent checkpoints. Four variants,
three frozen purged folds: at most twelve new fits, including four first-fold
probe fits. All keep the existing histogram hyperparameters, risk-normalized
residual target, residual strength one, and training-only usable/nonduplicate
admission. The 247 final-test origins remain untouched.

| Variant | Added representation | Candidate additions before train-only exclusions |
|---|---|---:|
| current_market | Six contemporaneous channels, signed/sum projection and applicability | 18 |
| price_path | Four price channels at lags 0–3 | 36 |
| activity_path | Two activity channels at lags 0–3 | 18 |
| joint_path | Both ordered channel groups at lags 0–3 | 54 |

Each compares with the frozen admitted-tail and historical-mean controls. Joint
minus price and joint minus activity are conditional family removals. Joint minus
current tests additional history with the same current channels, but also changes
column count: a positive result would require a later capacity-matched diagnostic,
not prove that temporal ordering alone caused it. Existing return lags are not
counted as newly discovered signals. Static applicability duplicates are reported
through the existing training-only rejection audit.

The first-fold probe stops continuation if all four scores are more than 0.10
below the frozen control or if any integrity, timing, schema, memory, or replay
check fails. Do not select/tune individual variants on the probe. Otherwise retain
all four variants for the other folds, reusing the first four checkpoints.
Cumulative experiment wall time is limited to 300 seconds across probe and resume,
with 15-second heartbeats, exclusive-run locking, per-fit sealed checkpoints and
private S3 copies. A stop preserves completed stages and does not authorize a
budget extension. The complete run reuses checkpoints without loading data or
fitting. Preserve the prior 263 comparisons; eleven added contrasts bring the
joint history to 274, using the existing within-fold 10/20/40-row block bootstrap.
All uncertainty remains conditional on fitted predictions and does not remove
adaptive-development bias.

## Status

Implementation and local feature tests are prepared. Actual AWS probe/full-study
results are separate evidence; no predictive improvement is claimed by this
protocol. Runtime preparation reused 589 sealed stages and installed the frozen
independent current-checkout environment without refitting prior models.

## Measured results

## Four-date market-path experiment

Measured on the same 535 purged development origins. These are exploratory offline scores, not leaderboard results.

| Representation | Official metric | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| admitted_tail | 0.305336 | 0.390579 | 0.152341 | 0.399881 |
| current_market | 0.309709 | 0.403381 | 0.167042 | 0.390619 |
| price_path | 0.300308 | 0.392004 | 0.149452 | 0.384947 |
| activity_path | 0.307916 | 0.392773 | 0.162730 | 0.396659 |
| joint_path | 0.299252 | 0.392028 | 0.132967 | 0.399023 |
| screened_tail | 0.309088 | 0.405901 | 0.154856 | 0.400172 |
| tail_states_and_priors | 0.309186 | 0.386784 | 0.178534 | 0.383185 |
| short_own | 0.308890 | 0.394139 | 0.166773 | 0.396641 |
| historical_mean | 0.217739 | 0.162550 | 0.182488 | 0.296861 |

Do not advance the four-date OHLC/activity path expansion unchanged. The single-row current-market block produces the highest reproduced development point estimate (0.309709), but its predeclared gain over the admitted-tail control is small and uncertain; adding prior-day price paths or joint price/activity paths reduces the pooled metric. Retain current OHLC/activity as an exploratory compact lead, keep the feature gate open, and next isolate whether market-state normalization or a shared raw-input architecture can use this information without merely adding chronology.

[Full protocol and limitations](../docs/market-path-research.md). No previous model was retrained. The final test remains gated.
