# Current-market normalization and volume confirmation

## Question and frozen design

The current fitted development leader is `current_market` (0.3097087232124053).
The completed diagonal-rank transfer experiment did not improve either frozen
control, so its unchanged fitted formulation is not repeated. This study asks a
different question: does the meaning of the current OHLC bar depend on prior
volatility and unusually high or low trading activity?

The frozen current-market panel is retained in every variant. The new blocks
are: (A) volatility-scaled intraday return, overnight gap and high-low range;
(B) bounded unusual-volume confirmation of scaled intraday direction,
close-within-range location and scaled range; (A+B) their exact union. Each asset
channel is projected as target-pair difference and sum. Each family adds six
numeric channels and one structural applicability indicator: 7, 7 and 14
candidate templates, respectively. No feature count is a performance claim.

## Primary research and applicability limits

- Moreira and Muir (2017), *Volatility-Managed Portfolios*,
  https://doi.org/10.1111/jofi.12513, documents time-varying risk/return scaling.
  It motivates testing volatility conditioning; it does **not** establish that
  these short-horizon commodity features predict the competition targets.
- Gervais, Kaniel and Mingelgrin (2001), *The High-Volume Return Premium*,
  https://sites.duke.edu/sgervais/research/gervais-kaniel-mingelgrin-2001/,
  finds information in abnormal trading activity in equities over a longer
  horizon. Here volume interactions are a testable transfer hypothesis, not a
  presumed commodity premium or a signed trading rule.
- The competition's fifth-place solution uses raw-input LayerNorm and short
  RNN/MLP representations with a ranking-aware loss:
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/zlf-solution-of-the-mitsui-commodity-predict
- The eighth-place report combines raw inputs and historical targets in a shared
  model: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/transformer-based-solution.
  Its stated lag construction is not copied: this repository retains explicit
  horizon-plus-one release boundaries. The reported feature arithmetic in that
  writeup is inconsistent, and its leaderboard score is not our validation score.

These reports also show that learned representations and objectives must remain
in the feature-research program. Adding hand-built columns alone is not evidence
that the gap to leading systems has been closed. These ablations isolate one
mechanism cheaply before allocating neural-model compute.

## Availability, leakage and numerical contract

Only market observations at or before origin t enter these additions. Every
21-observation risk and volume reference is shifted by one row, excluding t;
at least 14 past observations are required. Price channels use log prices;
volatility denominators have a fixed 1e-6 numerical floor, not a fitted tuning
parameter. Scaled moves are capped at magnitude 12. Abnormal log1p volume uses
its prior rolling median and standard deviation, then a fixed tanh(z/3) bound.
Invalid bars, negative volume and observed missing bars remain missing. Missing
instrument fields are structural absence and carry applicability indicators.
There is no backward fill, random split, future-label use or external join.

Existing train-only missingness, constant and duplicate screening is preserved.
The admission policy forbids budget/correlation/sign screening from silently
excluding the declared additions. A new usable column must be admitted in every
fit, or the study stops. The design does not change the control's model settings.

## Bounded execution and decision

The experiment first fits three variants on fold 1 and saves/replays every model.
Continuation requires at least one matched fold-1 gain of **0.002**, successful
preflight, and explicit inspection of that receipt. If no variant meets this
information-value gate, stop this exact formulation without claiming it has been
fully tested across time. Otherwise perform only the six remaining fits.

The total work budget is **nine fits and 300 cumulative study seconds**. Already
sealed stages are reused. Control models are replayed, never refitted. Every
successful stage is privately checkpointed to S3. Only 1709 development input
rows are read; the 247 final origins remain outside evaluation.

The full design compares A, B and A+B to the same current-market control and A+B
to A and B. Report 535-date official score, three folds, retained/rejected columns,
exact replay and block-bootstrap uncertainty. Family-wise intervals cover these
five declared contrasts only; they do not correct the whole adaptive history.
No outcome here by itself authorizes model promotion or final-test access.

## Outstanding high-value representation work

Target/market-specific raw feature visibility, causal released-label adaptation,
shared multi-target learned representations, ranking-aware training, and
cross-market context remain unresolved. Real carry, inventories, positioning,
weather and event features require verified calendar mappings, availability
vintages and data rights before use. Rejected rank transfer and longer OHLC paths
must remain negative evidence rather than becoming automatic repeat sweeps.

Feature engineering remains open. The historical leading leaderboard result is
a research objective, not a directly comparable score or a performance guarantee.
