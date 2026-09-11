# Matched fitted diagonal-rank ablation

## Hypothesis

The no-fit target-rank probe earned one bounded follow-up: `diagonal_rank` improved the pooled raw-mean ordering by +0.023016 and improved two of three development folds. The next question is narrower: **does that training-only target-rank metadata add incremental value when the existing pooled histogram model already has the admitted tail representation, and when it already has the current-market OHLC/activity block?**

This is not a broad hyperparameter search. It tests one rank template under two matched bases:

- `rank_tail` = frozen admitted-tail representation + one fold-local diagonal-rank target template.
- `rank_current_market` = completed `current_market` representation + the same fold-local diagonal-rank target template.

For each outer fold, the rank template is recomputed from labels strictly before that fold's `train_stop` and then held constant across dates for each target. It therefore represents training-history target metadata rather than released validation outcomes.

## Why these controls

`admitted_tail` is the frozen matched control used by the recent released-context and market-path studies. `current_market` is the highest reproduced fitted 535-date development point estimate at 0.309709, but its gain is small and uncertain. Testing rank metadata on both bases distinguishes a generic rank effect from an interaction with the compact current OHLC/activity representation.

The design is motivated by competition writeups that found stable target rank or target-group information useful, including the 26th-place mean-rank feature and the 10th-place covariance-regularized target ordering. Their validation and leaderboard scores are not treated as directly comparable with this project's purged development protocol.

## Bounded execution plan

- two variants × three frozen outer folds = **at most six new fits**;
- first run only fold 0 for both variants = **two-fit smoke/probe**;
- continue only if at least one variant is within 0.03 of its matched first-fold control and every lineage/admission/replay check passes;
- cumulative study hard budget = **180 seconds**;
- fixed histogram model and residual weight 1; no model hyperparameter sweep;
- all usable templates admitted except exact duplicates/constant-invalid inputs; the diagonal-rank template itself must be retained or the study stops before claiming a fitted result;
- exact serialization replay required after every fit;
- per-fit atomic checkpoints and private S3 synchronization;
- 10/20/40-date paired uncertainty added to the cumulative comparison family;
- final origins 1714–1960 remain untouched.

## Decision rule

Retain the diagonal-rank fitted family as a promising representation only when its matched pooled delta is positive **and** at least two fold deltas are positive. An interval crossing zero remains inconclusive and does not justify model promotion. A negative result retires this specific fitted formulation, not the broader target-grouping or online-adaptation research avenues.
