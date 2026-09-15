# Target-leg identity and aligned shocks — notebook 11

## Prior measured result
Notebook 10 completed three rank-state fits with exact replay. Global rank state
scored 0.390577, within-horizon/relative rank 0.399073, and their union 0.388567
versus the first-fold control 0.403381. All intended inputs were admitted. Do not
rerun those panels unchanged. The pooled 535-date leader remains 0.309709.

## Distinct representation hypothesis
The active pooled panel projects many asset-level values into shared difference
and sum columns and exposes market/horizon descriptors. It does not provide one
explicit column per underlying instrument in these new names. A similar price
shock may have different predictive implications for different underlyings.

Select at most 16 recurring underlyings from provided target-pair metadata by
appearance count, descending, with lexical tie-breaking. No outcomes, raw-price
variance or validation score select anchors. This is a deliberate bounded probe,
not a claim that frequency equals predictive importance or covers every target.
For each anchor define signed left-minus-right membership and unsigned membership.
Compute its current log return, normalized by its own 63-row population standard
deviation through t-1 (42 returns minimum, scale floor 1e-6, clip to [-12,12]).
Multiply that shock by each membership. The three panels add identity only,
identified shocks only, and their exact union (at most 32, 32, 64 templates).
No arbitrary integer target codes, target encodings, labels or external data enter
this builder. This is information-preserving target alignment, not a new data feed.

Structural nonmembership is zero. An anchor target with a missing/nonfinite or
nonpositive quote keeps a missing shock, not a fabricated zero. There is no price
forward-fill. Pair reversal negates signed membership and leaves unsigned
membership invariant; signed shock interactions follow the same parity. Current
shocks are never in their own normalization window. Prefix and perturbation
checks run before fitting. Daily rows retain the dataset's supplied alignment;
no synchronized exchange-time claim or anonymous-date reconstruction is made.

## Controlled comparison and evaluation period
Use the existing histogram learner and preprocessing. Training-only duplicate,
constant and missingness exclusions remain; all usable candidates fit within the
expanded exact panel budget. Do not tune hyperparameters. Compare against saved
fold 1 current_market (0.16704165319061334), train stop 1344, warmup 252, validation
origins 1349-1528 (180 dates). Existing model is replayed, never refitted.

The middle fold was chosen after seeing its weaker historic baseline, before the
new results. This reduces another search round on fold 0 but DOES NOT create a
fresh holdout. It is an exploratory period-specific diagnostic. No pooled score
is updated by it, and any advancement must be evaluated on other periods before
promotion. Negative first-fold families are not being rerun on another fold.

## Decision and budget
At most three new models in a 240-second worker, 15-second heartbeats, exact saved
prediction replay, separately sealed model stages, and preservation of parents.
Identity-only/shocks-only earn review at +0.002 versus original control. Joint
must reach +0.002 versus both original and identity-only controls. Numerical
panels must actually admit numerical inputs. This is a cost-allocation gate, not
proof of statistical significance or a leaderboard record. Five declared contrasts
are reported with the existing conditional 10/20/40-date block procedure. Those
intervals do not correct the whole adaptive search. No automatic other folds,
model promotion, ensemble, final test, package installation or AWS/Git writes.
Completed results reuse without fitting. Interrupted stages stop for diagnosis;
no automatic budget reset, no overwriting edited notebooks or previous studies.

## Coverage, overlap and limitations
Anchor counts, members/nonmembers, actual finite member observations, exact
training-array duplicate checks against the active baseline, and admitted inputs
are exposed. No claim of exhaustive numerical deduplication against the entire
historical feature catalog. Existing graph/PCA/proxy and FX-conditioned families
are not relabeled: this probe uses explicit identities and their OWN shocks,
not statistical incoming peers or newly estimated covariance clusters.

A gain can be driven by target identity rather than dynamic financial information;
identity-only is an explicit control. A loss can reflect fixed-learner interactions
as well as weak features. Many unproductive panels do not establish that algorithms
are irrelevant or that high-value representations are exhausted.

## Research rationale — outside the uploaded result
The 25th-place participant reports grouping targets using target_pairs market
metadata and restricting market-relevant inputs. The 97th-place participant
reports instrument A/B identifiers among inputs. These motivate, but do not
validate, this different one-hot membership/own-shock ablation. We do not reproduce
their learner, ensemble, label blending or retrospective validation practices.
Primary sources reviewed for this design:
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/25th-place-silver-mitsui-commodity-prediction
- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/97th-place-solution

Shared raw-input learned representations and ranking-aware losses remain distinct
research axes. Authentic carry, inventories, positioning, macro vintages and
weather require verified mapping, availability and rights; do not invent them.

## Manual outputs
Save notebook 11. Download the report, dashboard and private checkpoint ZIP, then
stop the space without deleting it. Only the report and executed notebook return
to chat. Downloaded ZIP is private, not public GitHub content. Source and notebook
additions are local until explicitly published; no automatic Git synchronization.
