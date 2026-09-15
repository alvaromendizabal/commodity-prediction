# Information audit and relative-state laboratory — notebook 13

## Why this milestone
Notebook 12 produced 0.154426, 0.165238 and 0.158861 on the middle development fold,
all below the saved 0.167042 control. All intended inputs were admitted. Close the
unchanged recipe. More automatic addition fits are not the next default.

## Frozen-model diagnosis: no fitting
Replay the three saved current_market models on exactly 180/180/175 dates. Score
both their saved predictions and the target-mean vectors already stored inside
those models. This comparison isolates the fitted correction above historical
means; it is not a newly optimized mean model or a deployable improvement claim.

For reference, released_priors, tail_risk and market_path, reorder 20-row blocks
within each fold twice (seeds 42 and 20260912). Use one shared order across all
424 targets and every column in the family. Other families remain untouched.
The fitted trees and preprocessing remain frozen. Inspect time-alignment reliance,
not causal effects and not the performance of a model retrained without a family.
Target identity that is constant through time is not disturbed. Correlated
alternatives can hide reliance. Shuffled inputs may combine later validation
values with earlier rows: they are synthetic diagnostic perturbations, NEVER
predictive inputs or candidates for promotion. Two permutations are NOT confidence
bounds. Do not select features using these diagnostic scores in this milestone.

Pooled scores are recomputed from 535 daily correlations, never from the average
of three ratios. All dates are previously used development dates. Final origins
1714–1960 remain untouched; rows1704 onward are not numerically read by the worker.
Full raw-file hash checks read bytes, not final-label values for feature selection.

## Concrete new feature engineering: 18 representations
Six fixed existing target-level inputs are used: return differences at lags0,1,5;
63-row risk difference; and released-target location estimates over63 and126.
For each build global rank, within-horizon rank, and within-horizon rank of its
own historical innovation. The innovation uses mean and population SD from the
preceding63 rows (minimum42), excluding the current row. SD<=1e-12 yields missing.
Average tied ranks are mapped to[-1,1] among finite values, minimum3 per cohort.
No missing values are forward-filled or converted into economic zeros.

These are new REPRESENTATIONS of existing information, not18 independent sources.
The prior columns already depend on legally released labels; this lab inherits
that causal contract. Its transforms do not consume outcome labels. Training
outcomes0–1163 are used only for association screening. New-transform prefix
checks do not independently recertify every historical parent feature formula.

Important mathematical control: global ranking is monotonic within a date, so
its daily Spearman against the same observed targets is unchanged. That equality
is tested, not presented as a predictive discovery. Such reparameterization may
still change what a pooled learner learns across dates, but requires a fitted
comparison. Within-horizon normalization changes comparisons across horizons;
innovation ranks change temporal/contextual representation.

Screen only the six innovation candidates on training rows252–1163, split at708:
>=70% coverage, >=40 eligible dates per half, nonzero same sign, no exact duplicate
of the active87-template baseline or another candidate. Rank eligible candidates
by the weaker absolute half-mean association; cap at6. Raw/horizon rank controls
are retained as controls, not selected as new marginal signals. Exact historical
380-template numerical overlap is not exhaustively checked. No candidate fits
or new validation scores are produced. A shortlist is NOT evidence of conditional
improvement beyond the saved forecasting model.

## Stop conditions and artifacts
180-second supervised worker,15-second heartbeats, zero .fit calls. Persist the
training cube/screen and each of three fold audits independently with hashes.
Failure retains completed stages and requires diagnosis, not budget resets.
Complete result reuse reads sealed evidence rather than predicting again. The
HTML dashboard uses Plotly without Chrome/Kaleido. Artifact ZIP is private and
is an off-disk copy only after download. No Git/cloud changes or automatic stop.

## Research context and limitations
Scikit-learn's primary documentation explains model dependence and correlated
feature limitations of permutation importance:
https://scikit-learn.org/stable/modules/permutation_importance.html
The competition participant's historical-rank features are contextual motivation,
not proof for our different predictor-normalization construction:
https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
No leaderboard-equivalence or performance guarantee. The next fitted experiment
must specify whether it tests added temporal information or reparameterization,
use isolated chronological folds, and compare against a frozen matched baseline.
Feature engineering remains open, including conditional effects and learned raw
representations. Negative evidence does not prove the learner is never limiting.
