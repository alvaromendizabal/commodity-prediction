# Close-only, leave-pair-out peer networks

## What the completed experiment said
Notebook 07 completed four fits. Availability only matched the old first-fold
score exactly (0.403381); intraday 0.397112, overnight/risk 0.401594, combined
0.375454 were worse. All numerical candidates were admitted. Do not run additional
session folds or tune that unchanged formulation. These are conditional negative
results, not a proof that all session information is useless for every learner.

## New mechanism and scope
Can other assets' recent moves inform a target whose own lag features miss the
incoming market move? Use only the price/exchange-rate series named by target
legs. No open/high/low/volume requirement. All 424 targets have a *potential*
close-based definition; actual graph/observation availability is measured.

Estimate both a synchronous graph C(i,j)=Corr(r_i(s),r_j(s)) and a directed graph
D(i,j)=Corr(r_i(s),r_j(s-1)). Each uses 126 past rows, at least 84 pairwise
observations, and is rebuilt every 21 rows beginning at origin 127. Scales and
means use those same past follower-return rows. At refresh t, s ends at t-1.
The directed peer sample therefore ends at t-2. No current/future data enter
graph fitting. The block's graph and scaling remain frozen until its next update.

For each target leg, exclude that leg AND its target's other leg. Select at most
five peers by absolute correlation (>=0.10), with canonical-name tie breaking.
Require at least three selected and observed peers and 60% observed absolute
weight. Use signed correlation-weighted means; missing real observations are
not zero-filled. A genuinely absent second leg is zero only at target projection.
Current standardized returns are clipped at 8 under past-only moments.

Each graph produces current peer impulse, a five-row peer mean (min 4 observed
rows), and previous-peer-impulse minus current own standardized return. Project
each as target-leg difference and sum: 6 numeric columns per graph. Four common
availability columns describe observed absolute-weight fractions. Total: 16
candidate templates, not 16 independently proven signals.

The directed graph captures historical one-row response ordering; it is not
causal identification, a verified economic customer/supplier map, or evidence
that international closing times are simultaneous. Signed correlations may
reflect FX identities/common drivers. Their estimated links are not individually
significance tested; top-K is a fixed representation choice, not verified edges.

## Existing-feature overlap audit
Inspected frozen source at d142a4cb57a5c4b2880f9341619e13a735b1cddc:
- domain/market.py already has return lags, global momentum ranks, observation-age
  signals and trailing PCA. Do not relabel these as new.
- domain/relationships.py has named-proxy exposures, a structural currency graph,
  and within-target pair dynamics. Its graph is not this empirical cross-asset
  incoming-neighbor graph excluding both prediction legs.
- domain/features.py already has released target priors. These new features do
  not use target labels, static target mean ranks or signed released-label peers.
The runtime checks exact duplicates against the active 87-template base and
within this family, not against every historical intermediate array.

## Fitted additions/removals and budget
Four fixed panels: availability-only (4), synchronous (4+6), directed (4+6), exact
union (4+12). The union comparisons expose removal of one numerical graph family.
Hold histogram learner and original fitting settings fixed. Existing training
preprocessing handles missing/constant/exact-duplicate screening. No label-driven
graph selection, hyperparameter sweep or univariate-sign shortlist. The original
current_market model is checksum-verified and replayed exactly, never refitted.

Training stop 1164, warmup 252, validation origins 1169–1348. At most four fits;
240 seconds including checks/construction/replay/fitting, with 15-second logs.
Only first-fold labels are read. Final 247 origins remain gated. Every model is
sealed independently, feature checkpoint preserved, and completed runs reopen
without refitting. Interruptions stop for diagnosis, not automatic retries.

Before fitting: two exact prefix comparisons (505 and 1164 rows) and numerical,
axis, raw-source, parent-checkpoint and environment checks. Graph selection uses
only raw historical prices; model screening uses training outcomes only.

Gain >=0.002 versus BOTH saved original and fitted availability controls earns
review, not promotion or automatic later-fold execution. This rule is declared
before network scores but after prior studies used the same validation period.
Eight contrasts use 10/20/40-row block uncertainty, conditional on current fitted
models. Those intervals do not correct the full adaptive research history. A
first-fold winner must later survive separate periods and family ablations.

## Research basis and transfer caveat
Bennett, Cucuringu & Reinert (2022), *Lead-lag detection and network clustering for
multivariate time series with an application to the US equity market*:
https://arxiv.org/abs/2201.08283
The authors study directed lead-lag structures and predictive signals. This
implementation is a simple feature hypothesis inspired by that mechanism, NOT
a replication of their clustering/validation procedure or guaranteed commodity
alpha. The paper's results cannot be substituted for this experiment's metrics.

Curme et al. (2014), *Emergence of statistically validated financial intraday
lead-lag relationships*: https://arxiv.org/abs/1401.0462
Their multiple-testing-aware network study reinforces the distinction between
candidate correlation links and validated economic/predictive relationships.
Here the fitted ablation is the predictive test; individual links remain unproven.

## Research remains open
Keep the current pooled development leader 0.309709 unless a comparable full
study actually changes it. Do not compare this first-fold score with .63834 on a
different leaderboard period. Residual label context, richer target identity,
learned shared representations and real dated commodity fundamentals remain
separate hypotheses, conditional on leakage-safe timing and actual data rights.
Never fabricate carry, inventories or economic dates from anonymous row IDs.

## Operating instructions
Upload commodity_close_network.py to JupyterLab home and run it once in a terminal.
It installs notebook 08 and this declaration without training. Run notebook 08
using Commodity - manual (verified); its execution cell can fit four models.
Save the notebook, download the JSON/HTML and private ZIP, then STOP SPACE.
Do not delete old working copies, which contain the reused Python environment.
The script makes no AWS API calls, Git writes, S3 changes or package installations.
New notebook/helper/config/docs are local additions, not published to GitHub.
