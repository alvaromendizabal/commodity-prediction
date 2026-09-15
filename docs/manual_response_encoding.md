# Delayed conditional response encoding — notebook 12

## Measured starting point
Notebook 11 completed three middle-fold fits. Identity scored 0.15334865292560604,
identity-aligned shocks 0.1336489611227173, and their union 0.12888769504762737,
versus the saved current_market score 0.16704165319061334. None passed its gate.
Those panels stay closed unchanged. These are reported results, not new replays
performed while preparing this package. The pooled 535-date control is 0.3097087232124053.

## Hypothesis (not a demonstrated predictive relationship)
Encode each target's historically estimated RESPONSE to a market signal, rather
than add more static identity indicators or raw transforms and require a single
pooled learner to infer every changing response. Learned feature coefficients are
supervised. The downstream histogram forecaster and its settings stay unchanged.

Two drivers use the directed log-price spread, or a single log price for a single
asset target: 1-row and 5-row changes, scaled by sqrt(lag) times trailing 63-row
spread-return SD through t-1 (minimum 42 observations; floor 1e-6). Driver values
are clipped to [-12,12]. No prices are forward-filled, and the short path requires
all adjacent observations. All target legs must be present in input metadata.

For target j with horizon h, let d=h+1. At prediction t, pair signal s[u,j] with
its corresponding outcome y[u,j] only for u<=t-d. For each 63/126-row window,
compute all moments over the SAME pairwise-finite observations (minimum 42/84).
No origin u is paired with an outcome from a different origin. Missing outcomes
are never economic zeros. Pair counts are used internally, not added as features.

The centered standardized response feature is

    n/(n + 63*h) * corr(s,y) * (s[t]-mean(s))/sd(s)

clipped to [-6,6]. Target variance <=1e-16 or signal variance <=1e-10 makes the
feature missing. No outcome intercept is added: this does not rename the existing
released-target mean priors. The fixed 63*h pseudo-count conservatively attenuates
noisy overlapping-horizon estimates; it is NOT a proven effective-sample formula.
No selection of windows or shrinkage strength on validation is permitted.

The feature states update on already-released earlier validation outcomes. This
is explicitly prequential supervised feature generation, NOT zero-learning
preprocessing and NOT refitting the downstream forecasting model on validation.
The latest usable feature-label origins at prediction 1528 are 1526,1525,1524,1523
for horizons 1,2,3,4. No data at or beyond the final-test boundary are evaluated.
The historical parent feature panel is reused under its existing release contract.

## Matched experiment
Three new models on middle development fold 1 only:

1. response_drivers: two label-free spread drivers added to current_market.
2. response_encoded: four centered learned responses added to current_market.
3. response_joint: exact union of the two and four added features.

No new availability or identity indicators. Model settings, training preprocessing,
and target standardization remain fixed. Exact duplicates, constants and missingness
are checked using the fitting interval. The declarations are frozen before results.

Training ends at 1343 (stop1344); validation origins1349-1528. This period has
already informed research. It is exploratory, not fresh confirmation. At most
three fits within one240-second worker. Existing models must replay exactly.
A previously interrupted attempt cannot be silently rerun or have its budget reset.
Completed stages are retained. A completed report can be re-opened without fitting.

Drivers earn review with >=0.002 gain over the saved same-period control. The
encoded/joint panels must gain >=0.002 over BOTH the saved control and drivers.
All intended added inputs must be admitted and integrity/replay checks pass.
Passing only earns review; no automatic other folds or promotion. Five conditional
block contrasts at10,20,40 dates do not correct all adaptive research history.
A future frozen-coefficient comparison would be required to isolate the value of
sequential updating itself; this experiment tests the combined response encoding.

## Novelty / overlap boundary
Existing source has raw released target means/risk/ranks and asset-factor return
covariances. This recipe pairs TARGET outcomes with their own ORIGIN-aligned
market signals under each horizon release delay. It is not another target-rank
transform, asset-to-asset lead-lag graph, or anchor one-hot array. The exact
numerical duplicate audit covers the active87-template baseline and this family,
not every historical abandoned representation. Positive performance is unproven.

## Primary sources reviewed for rationale
- Stefan Nagel, Evaporating Liquidity, NBER17653 (2011):
  https://www.nber.org/papers/w17653
  Equity reversal returns vary with market conditions. This motivates allowing
  response changes; it is not evidence this commodity feature predicts well.
- Lonnie, Mitsui15th-place participant writeup (2026):
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th
  Describes adaptation to newly released labels. We do not reproduce that neural
  ensemble, its model-retraining schedule, or claim its score for our folds.
- Repo domain/features.py (pinned d142a4cb57a5c4b2880f9341619e13a735b1cddc):
  existing per-target release delay h+1 is retained, not weakened.

## Deliverables and preservation
Canonical source/config are unchanged; new script/config/protocol/notebook are
local additions, not automatically pushed to GitHub. The report and self-contained
Plotly HTML contain evidence, not raw target tables. The private ZIP includes
label-derived features, models and predictions: keep it private and download for
an off-disk copy. Stop the SageMaker application, never delete its space. Keep
older project folders because the notebook kernel reuses their verified environment.
