# Feature diagnosis and released-rank candidate laboratory

## Measured starting point
Notebook 08: baseline 0.403381; network availability 0.372697; synchronous
0.374725; directed 0.363208; joint 0.363892, on the same first 180 dates. All
numeric inputs were admitted. The network experiment failed its declared gate.
No old fit or graph construction is repeated in this milestone.

## Part A: isolate a representation decision, two fits only
The mask-only model was substantially lower. This motivates a *changed*
experiment rather than expanding the same negative panels. Remove the four
explicit network-availability columns while preserving the numerical cube
byte-for-byte. Fit synchronous-only (6) and directed-only (6), each added to the
same 87-template current_market panel. No union model; no extra mask control.
Replay the old baseline and both already-fitted masked counterparts first.
Keep the original learner, train-only preprocessing, missingness handling and
model settings. Six numerical inputs must be admitted or stop before fitting.
A refitted model changes all its splits: this is a representation ablation, not
proof that a coverage feature is universally harmful or a causal market effect.
Numerical missingness still exposes some availability even without the explicit
mask columns. Do not claim that this removes all missingness information.

Compare each result with the original and its own masked counterpart. +0.002
against both earns review, not promotion or automatic new folds. Threshold is
chosen after observing 08, so the analysis is diagnostic/adaptive. Four declared
contrasts use 10/20/40-date blocks; intervals condition on fitted models, not the
entire history of feature selection. Repeated first-fold use remains a limitation.

## Part B: engineer information in rank space, no fitted model
Predictive ranking and regime adaptation motivate representing a target's
historical ordinal state rather than just raw magnitudes or a static mean rank.
The existing repository has long raw-value released priors; the earlier scalar
rank-prior transfer was negative. This dynamic, common-origin cohort construction
is different but still unproven. No new labels are invented and no economic date,
carry, inventory, weather or external supply-chain mapping is assumed.

Target horizon h is released at origin+h+1. Since h is 1..4, a cross-sectional
cohort from origin s is usable at s+5. First shift the ENTIRE label matrix by 5;
only then rank observed labels. This avoids the common leakage mistake of ranking
a short-horizon label against longer-horizon outcomes not released yet. Shorter
horizons deliberately wait longer here to keep every cross-sectional cohort at
the same origin. Missing values (including sentinel -999999 and infinities) remain
missing, not zero. Average ties are mapped into [-1,1]. Sparse groups with fewer
than 3 labels are missing. Equal-valued groups map to zero, a representational
convention; constant dates are undefined in correlation diagnostics.

Twelve numeric templates: global and within-horizon latest ordinal position,
EMA21, EMA63, EMA21-minus-EMA63, latest-minus-EMA63 (10); global-minus-horizon
latest and EMA63 (2). Two coverage diagnostics: cohort observed fraction and own
rank availability over 63 rows. EMA min histories are 14 and 42 observations.
EMA state persists across missing observations; raw outcomes are not forward
filled. Coverage diagnostics are NEVER automatically included in a fitted model.

Only rows 0..1163 enter this lab. At the last training row, the newest source
origin is 1158. Runtime prefix equality is checked at 505 and 917 rows. Screen
rows252..1163 using pairwise-observed daily feature/target Spearman, split at708.
Require >=70% raw feature coverage, >=40 eligible dates in both halves, consistent
nonzero direction, and no exact duplicate against the active base or candidates.
Retain at most6 by the smaller absolute half-mean association. This is an
exploratory screen, not conditional importance or out-of-sample evidence. A
rejected candidate is not proven useless for nonlinear/interaction models.

## Research sources and limits
- MITSUI 26th-place author writeup (historical mean ranks and clustering):
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
  The author uses a static training mean of daily ranks, with zero filling in the
  example. We do not inherit that missing-value or full-prefix timing policy.
- MITSUI 15th-place author writeup (online adaptation and ranking-heavy training):
  https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th
  This motivates adapting the information state; it does not prove this feature
  recipe works. We do not reproduce the author's architecture/retraining claims.
- Existing source: src/commodity_prediction/domain/features.py at d142a4cb:
  horizon-delayed raw-value EWMs, market-pair pooling and cross-horizon priors.
- Bennett, Cucuringu and Reinert, https://arxiv.org/abs/2201.08283, original
  motivation for the prior incoming-peer network; no new network estimation here.

## Runtime and records
One 240-second worker covers checks, exact old-model replay, cached-feature load,
rank candidate construction/screening, two fits and matched diagnostics. 15-second
supervisor heartbeats; terminate process group at cap. Preserve independent sealed
model and candidate stages. A complete rerun reads/validates checkpoints with zero
fits. An incomplete run stops for diagnosis, not automatic budget resets/retries.
Self-contained HTML, JSON and notebook output are display artifacts; private ZIP
also contains label-derived candidates, models, and predictions. Download ZIP
privately; do not send it to ChatGPT or public GitHub. No space shutdown occurs.

Source/kernel/raw/parent checks precede model loading. Original tracked source and
08 checkpoints remain unchanged. No package installation, AWS/Git API call, final
evaluation, GPU, ensemble, more folds, or fitted rank model is executed. New files
are local additions; they are NOT already on GitHub. Feature engineering stays
open; highest reproduced pooled development metric remains 0.309709 unless a
comparable complete study changes it. No promise of a historical leaderboard win.
