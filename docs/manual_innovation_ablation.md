# Round 14: Historical surprise: magnitude, ranks and robust scale

## Evidence, not promises
Notebook13 replayed three saved models exactly with zero fits. The pooled model
scored 0.3097087232124053 versus 0.217739151200211 for its stored training means.
Perturbing released-prior timing lowered pooled scores by 0.136151523334791 and
0.057388292272549435. These are two dependence diagnostics, not causal importance
or confidence bounds. Prior reliance is not proof any new prior feature works.
The middle fold scored 0.16704165319061334 versus mean-only 0.1824882692897385.
Three innovation-rank features passed a training-only screen, not fitted validation.

## Common comparison contract
Both rounds are specified before either is run. Each has 24 representations,
four separate ablation panels, one 240-second worker, ten Plotly charts, a private
checkpoint ZIP and an independent result. They share the frozen original baseline,
not each other's outputs, winning features or fitted models. Run sequentially.
A negative completed round does not block the independent other round.
Training stop1344; validation1349:1529 (180dates), warmup252. No final-test rows.
No source/config changes to historical parents, no control refits or optimization
of learner parameters. Admitted-only screening retains usable nonduplicate inputs;
training associations are recorded, not used to improvise replacements after scores.
The current downstream fit code and data-missingness rules remain unchanged.
All feature transforms are as-of, assuming the verified parent's h+1 release rule.
No direct new outcome argument enters either builder. Scaling windows use t-1
and earlier; already-released inputs may update during validation with frozen models.
Numeric checks use copied pandas/NumPy arrays, preserving Copy-on-Write compatibility.
The six notebook13 innovation-rank arrays are reproduced exactly before round14 fits.
Both rounds independently verify the saved notebook13 input prefix and old models.

## Inventory and gate
Six source channels: return differences at lags0/1/5, risk63 difference,
and released location63/126. For each source, over the preceding63 rows (minimum42):
1. (x_t - historical mean) / population SD, clipped to [-12,12].
2. Within-horizon ordinal rank of the UNCLIPPED ordinary surprise.
3. (x_t - historical median) / IQR, clipped to [-12,12].
4. Within-horizon ordinal rank of the UNCLIPPED robust surprise.
Zero/insufficient scale produces missing, not infinite/saturated output. IQR is not
multiplied by a normal-distribution conversion; this is a fixed scale definition.
Ranking uses average ties among observed inputs, minimum3, mapped to [-1,1].

24 templates = 18 newly constructed representations + 6 existing notebook13
innovation ranks carried forward exactly. The six sources are not newly discovered
information sources. Robust within-target scaling of these target-level channels
is not the old asset-level robust-shock recipe, but semantic overlap is explicit.

Four fits: the exact3-name audit shortlist (ordinary ranks for lag0, location126,
lag1), all12 ordinary representations, all12 robust representations, and all24.
The extra ordinary inputs did not all pass the previous marginal screen; they are
fixed family-level controls for complementary nonlinear effects, not reclassified
as passed candidates. The shortlist model tests whether broader panels dilute it.

Shortlist/ordinary need +0.002 vs saved baseline. Robust needs +0.002 vs baseline
AND ordinary. Joint needs +0.002 vs baseline AND both full separate groups.
All intended inputs must enter a panel before it qualifies for review.

## Leakage, uncertainty, and stopping
The full prediction label vector is read for fitting/evaluation only up to origin1528;
feature builders accept the previously sealed parent array, not y. Parent release
provenance is inherited, not independently re-proven by the new prefix test.
Exact full-target prefix checks at1164/1349 and future-perturbation smoke tests precede
fits. Missingness/constant/duplicate checks use fitting data only. Exact duplicates
are audited against the active baseline, within this round, and against the prior
18-candidate training cube; not every historical representation.

The +0.002 rule allocates compute, not establishes significance. Paired block
intervals10/20/40 dates, 2000 draws and simultaneous within-family bounds are
conditional on these models and do not correct repeated development inspection.
No automatic extra folds or promotions. A useful candidate requires later temporal
replication and a matched removal; a negative panel remains closed unchanged.
The fixed learner is an experimental control, not proof its capacity is sufficient.

Both setup and notebooks preserve existing differing files. No blind retries or
budget resets: interrupted states require diagnosis. Successful stages persist;
completed results reopen without fitting. Long tasks have heartbeat logs and hard
supervision. A shared lock prevents parallel rounds. Export errors cannot justify
retraining completed models. Backups contain private derived features and models;
download privately, not to ChatGPT/public GitHub. Stop space, never delete its disk.

## Primary sources and scope
1. sklearn RobustScaler documentation (median/IQR method and outlier sensitivity):
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
Our causal rolling implementation is NOT a fit of a whole-dataset RobustScaler.
2. Campbell & Thompson, Predicting the Equity Premium Out of Sample, NBER11468:
https://www.nber.org/papers/w11468
Historical means are substantive benchmarks; small in-sample associations do not
justify abandoning out-of-sample controls. Their equity study does not validate
these commodity features or the numerical +0.002 threshold.
3. Moreira & Muir, Volatility-Managed Portfolios, Journal of Finance (2017):
https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
Risk normalization motivates a test, not a claim of short-horizon return predictability.
4. Repository source and uploaded notebook13 report pin the actual implementations,
source inputs, delayed-prior contract and all starting performance figures.

All files from this installer are LOCAL additions. No AWS API/Git writes, source
reset, environment installation, ensemble, final-test evaluation or autonomous work.
