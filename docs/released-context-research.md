# Short released-history and signed peer context

This is the next bounded feature experiment after the inconclusive risk-state expansion. The feature gate remains open. The final 247 origins are excluded.

## Hypothesis and prior evidence

The verified [fifth-place writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/zlf-solution-of-the-mitsui-commodity-predict) uses a short raw-input sequence; the [sixth-place writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/artem777-in-mitsui-6th-Place) emphasizes recent released labels; the [eighth-place writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/transformer-based-solution) describes joint target-history inputs. These motivate an information-representation study, not a claim that their architectures or scores have been reproduced. See the [competitive audit](competitive-research.md) for source limitations and the verified historical benchmark.

Our earlier target-aware study already exposes the latest own-target label. The pooled representation instead emphasizes long-window priors and robust innovations. This study asks whether a short sequence and related-target history add value under the current fixed pooled model. Reusing a familiar information source in a matched representation is not a claim of inventing that source.

## Exact candidate families

| Family | Twelve candidate templates | Availability |
|---|---|---|
| Own short history | Four released snapshots, two-/four-date means, one-/three-date changes, four-date innovation, volatility, sign balance and availability | For each target, shift labels by its own horizon + 1 before all operations |
| Signed peer context | Latest, two-/four-date means, one-date change, dispersion and coverage for each of two peer groups | Every peer is delayed separately before aggregation; no backward fill or future estimates |

Same-pair peers share the unordered underlying asset expression and exclude the own target. Reversed pairs receive the opposite sign. Shared-asset peers have intersecting long/short exposures and exclude the entire same-pair group. Fixed metadata incidence weights align positive and negative exposures; no validation correlations or learned graph edges select peers. A peer label is divided by its horizon before aggregation and multiplied by the receiving target's horizon afterward. This creates a noisy average daily-return context from differently timed available observations, not a reconstruction of a contemporaneous return.

Every peer mean renormalizes over currently observed constituents. No-neighbor values stay missing, with zero coverage. Dispersion uses the same signed observations and weights. These features neither require nor infer a real calendar. Temporal blocks are not asserted to be economic seasons.

## Frozen design and bounded decision

- Reuse the exact admitted-tail control (69 candidate templates) and the previous reference summaries. Keep the histogram learner, target normalization, residual weight one, training-only preprocessing, purges and 535 validation dates fixed.
- Fit own history, peer history and their combination: at most nine new fits across three folds. Their panels contain 81, 81 and 93 candidate templates before training-only missingness/constant/duplicate exclusions. Each family is tested through additions and the joint panel's component removals.
- Test causal prefix equivalence, unreleased-label perturbations, orientation reversal, excluded-own-target behavior, missing neighbors, checkpoint reuse and tampering before the experiment.
- Execute the three first-fold fits, preserve their checkpoints, inspect timing and replay. Stop on invalid timing/schema/replay, exceeded memory/runtime, or all three first-fold scores more than 0.10 below the frozen control. Otherwise complete the remaining six fits; never tune or select variants from the first fold.
- The cumulative computation budget is 300 seconds across the probe and resumed study, enforced by a persisted timer and a hard process deadline. Each fit seals model, predictions and results atomically; 30-second heartbeats and per-fit counters expose progress. Preserve the first-fold checkpoint before continuing.
- Report every fold, candidate/retained/rejected counts, matched official metrics and all eight declared contrasts. Include them in the previous 255-comparison history, using the same paired block bootstrap. Conditional intervals do not undo adaptive research or fitted-model uncertainty.

The information gain is whether very recent, correctly available history or shared asset exposure improves the representation. The experiment uses existing data and CPU estimators. A negative result closes this specific formulation, not every possible short-sequence or joint-model approach. Broader feature maturity and likely-winning performance remain unproven.

## Results

The experiment completed in **108.025 seconds** across the first-fold probe and resumed stage, below the cumulative 300-second limit. The probe's three model checkpoints were uploaded and verified before the remaining six fits. All nine final models replay exactly; ten sealed stages are preserved in the private bucket with a full-download SHA256 check. Complete resume verifies the stages in 0.031 seconds without loading data or fitting.

| Representation | Candidate / retained / rejected, each fold | Official metric | Fold 1 | Fold 2 | Fold 3 |
|---|---|---:|---:|---:|---:|
| Frozen admitted-tail control | 69 / 69 / 0 | 0.305336 | 0.390579 | 0.152341 | 0.399881 |
| Own short history | 81 / 81 / 0 | 0.308890 | 0.394139 | 0.166773 | 0.396641 |
| Signed peer context | 81 / 81 / 0 | 0.243727 | 0.335201 | 0.120210 | 0.291910 |
| Both families | 93 / 93 / 0 | 0.249300 | 0.335201 | 0.147143 | 0.282981 |

Own history adds **+0.003554**, with conditional 20-date-block 95% interval **[-0.008937, +0.017018]**. It helps the first two periods slightly, falls in the third, and remains below the existing 0.309186 leader. Its middle-fold score 0.166773 also remains below historical means at 0.182488. It is not a demonstrated improvement.

Peer context loses **0.061608**, interval **[-0.114687, -0.014059]**, and lowers all three fold scores. Adding it to own history loses **0.059590**, interval **[-0.109289, -0.016692]**. Adding own history to the peer panel recovers only +0.005572, interval [-0.008359, +0.021655]. All candidate columns are admitted: weak performance cannot be explained by a feature-budget exclusion.

The expanded **263-comparison** history has zero positive simultaneous lower bounds at block lengths 10, 20 and 40. The conditional negative intervals are evidence under this fixed representation; they do not establish a universal economic effect or survive every adaptive-search correction.

**Decision:** do not promote either family as proven. Stop expansion of this signed peer-aggregation formulation: it loses consistently and degrades the own-history panel. Preserve the own-history result as an inconclusive compact alternative. The overall development best remains **0.309186**. Adding plausible columns alone has not closed the competitive gap.

Eleven new tests passed for release boundaries, prefix equivalence, peer exclusion, signed orientation, missingness, partial reuse and tamper detection. New source, aggregate results and the completed models are fingerprinted independently of all previous studies. The canonical research notebook is being updated with the measured ablations; final publication evidence is recorded separately.

The next highest-value question is whether a short raw-market sequence with an appropriate shared representation captures information that engineered summaries discard. Reconcile existing raw/lag experiments first, then compare contemporaneous and four-date inputs under the same fixed learner before any larger sequence architecture or ensemble search. The gold writeup motivates that hypothesis; it does not prove the outcome. This study does not close the feature gate.
