# Released-rank feature ablation — notebook 10

## Source-derived state
Notebook 09 completed after the read-only-array correction. The numerical-only
network scores were 0.377545 and 0.399654 versus 0.403381 on the same first fold;
neither met the review gate. Close those tested formulations unchanged. The
pooled current-market development score remains 0.3097087232124053.

The training-only rank laboratory shortlisted six of twelve numerical candidates.
The six names and their group membership are fixed in the adjacent declaration.
No coverage diagnostic is included as a fitted feature. No new shortlist is chosen
using the validation outcomes. Features rejected by this screen are not universally
proven useless; they do not enter this particular small follow-up.

## New, as-yet-unmeasured hypothesis
Recently released cross-sectional ordinal states may add useful information to
raw-return priors. Global latest rank, innovation versus the 63-row exponentially
weighted mean and a 21-row mean form one panel. Within-horizon latest rank,
within-horizon innovation and the global-minus-horizon latest contrast form a
second. A third is their exact union. Panel sizes are 3, 3 and 6.
This is a time-varying representation; it is not the earlier static rank-prior
experiment. It may still partly encode persistent target differences. The
training-only variance chart diagnoses that possibility without changing the fit.

## Availability and leakage contract
For prediction origin t, all targets' outcome cohort comes from origins <=t-5.
Waiting five rows before global or horizon ranking respects h+1 releases for h=1..4
and prevents a short-horizon observation being compared against an unreleased
long-horizon label from the same origin. Ranking uses observed labels, average
ties, minimum three observations, maps to [-1,1], and does not fill missing labels
with zero. EWM spans/minimum histories are 21/14 and 63/42, adjust=False,
ignore_na=False. The existing corrected training builder is never modified.

The new builder extends only to the 1349-row first-fold prefix. The last five
outcome rows are removed from its private feature-input copy before ranking.
The 1164-row constructed prefix must exactly match notebook 09's sealed array
and corrected builder, and full-prefix comparisons are checked at prediction
origins. A synthetic future-label perturbation test runs in the locked AWS runtime.

Later validation predictions may use earlier validation outcomes AFTER release;
this is sequential feature updating, not refitting. The model and shortlist stay
fixed. There is no claim that validation labels are never read: separate labels
also support honest validation scoring. Final origins 1714-1960 are never evaluated.

## Matched fitting and evidence
Use the unchanged current_market control, replayed exactly; zero control refits.
Use the existing histogram model settings, training-only preprocessing, and
existing admitted-feature policy (max_features equals panel width; correlation
cap 1.01). All declared numerical additions must be admitted before fitting.
At most three new fits, first fold only, and a 240-second supervised worker.
Each successful model/prediction/result stage is sealed and replayed independently.
An interrupted run is not retried automatically and never gets a silent time reset.
A completed run is verified/reused without fitting. No prior network models run.

Compare all three additions to current_market plus union-versus-each-group,
using the original 10/20/40-date paired block intervals. A +0.002 matched gain
is only a resource-allocation gate for review of later periods. The first fold
has repeatedly informed research; neither five-contrast intervals nor the gate
correct the full adaptive search history. No model promotion, final evaluation,
ensemble, hyperparameter changes or leaderboard-equivalence claims.

## Research context — distinct from measured project findings
A participant's 26th-place MITSUI writeup reports stable historical target-rank
features and clustering. Its displayed construction includes zero-filled labels;
we deliberately do not adopt that missingness convention. The current time-varying,
common-release-cohort representation is a separate hypothesis, not a reproduction
of that complete solution or a promise of its leaderboard performance.

Primary sources reviewed for this milestone:
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview/evaluation

## Operations and publication
The helper installs one notebook, one script, this protocol and a fixed declaration.
It makes no AWS/Git calls, installs no packages and changes no old working copies.
Use Commodity - manual (verified). Keep old folders because the kernel environment
is reused from commodity-prediction-current. Save the executed notebook and JSON,
download the private ZIP off disk, and STOP (not DELETE) the space. Model/data ZIPs
are private. The notebook/report contain aggregate research evidence. Generated
source is local, not already published to GitHub. Further research remains open.
