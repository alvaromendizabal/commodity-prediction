# Metric-aligned target-rank priors

## Research question

Feature engineering remains open. After the bounded market-path study, the highest reproduced development point estimate is `current_market` at **0.309709**, but its predeclared gain over the admitted-tail control is small and uncertain. Before spending on another fitted model, test a cheap competition-specific question: does the *cross-sectional rank history of each target* contain stable information that is not captured by the existing raw target-mean control and released-value priors?

This is deliberately a **no-fit probe**. It uses the same three purged development folds and never evaluates origins 1714–1960. Every score vector is estimated from each fold's training prefix only and then held fixed over that fold's validation dates.

## Why this is distinct

The repository already contains raw target means, released raw-value priors, short own-target released history, market/asset features, tail-risk states, current OHLC/activity inputs, and a large family of causal engineered signals. The four-date market-path study showed that adding longer price/joint paths did not improve the matched control, so this probe does not add more chronology. Instead it studies the geometry of the official cross-sectional rank metric itself.

Relevant competition evidence:

- The 26th-place solution reported that a target's mean historical **daily rank** was unusually stable and useful as a feature, and also explored target clustering: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction
- A competition discussion described a regularized covariance/mean-rank ordering motivated by the metric, reporting roughly 0.4 on late validation windows and 0.479 private leaderboard. Its validation and preprocessing differ from this project, so the score is context only: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/discussion/668589
- The 25th-place solution emphasizes market/target grouping and provided label lags; the 15th-place solution emphasizes online chronological adaptation and a ranking-heavy objective; the 5th- and 8th-place writeups emphasize short raw context, ranking-aware objectives, and shared multi-target representations. These remain separate hypotheses rather than evidence that the present rank prior will help.
- The 3rd-place writeup reinforces target-specific representations tied to the exact target-pair instruments. Any ambiguous future-looking or negative-lag construction from public descriptions is excluded here; this project retains its explicit point-in-time contract.

## Declared methods

1. `raw_mean`: training-only per-target raw label mean. This is the sanity/reference ordering.
2. `mean_rank`: mean of each target's centered, unit-normalized daily cross-sectional rank vector over training dates.
3. `diagonal_rank`: mean standardized rank divided by its training-only rank variance, with a conservative numerical floor.
4. `ledoit_rank`: a Ledoit-Wolf shrinkage covariance estimate over training-only standardized daily rank vectors, followed by a covariance-adjusted mean direction.

Missing labels are excluded when ranking each date and become neutral zero coordinates only after the observed ranks are centered and normalized. No validation label enters a prior. No outer-fold score chooses a covariance hyperparameter: Ledoit-Wolf determines shrinkage from training data.

## Decision rule and execution limits

Advance this family to matched fitted-feature ablations only if a rank method improves the pooled development score over `raw_mean` by at least **0.02** and improves at least **two of three folds**. Otherwise preserve the negative probe and move to a distinct representation hypothesis rather than tuning the failed prior.

The probe performs **zero model fits**, has a 120-second hard limit and 15-second heartbeats, writes aggregate reports only, and preserves the final-test gate. A favorable point estimate remains exploratory because these development folds have already informed extensive prior research.

## Measured no-fit probe

The first verified AWS probe completed in **9.692 seconds** after the branch Quality gate passed. An independent reproduction completed in **9.580 seconds** with exact score parity. Both performed zero training fits, loaded zero models, executed no notebooks, and did not evaluate the final 247 origins.

| Prior | Official metric | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| raw_mean | 0.217739 | 0.162550 | 0.182488 | 0.296861 |
| mean_rank | 0.236130 | 0.185347 | 0.153915 | 0.371914 |
| **diagonal_rank** | **0.240755** | **0.177289** | 0.159841 | **0.385611** |
| ledoit_rank | 0.230729 | 0.172699 | **0.228095** | 0.298018 |

`diagonal_rank` improves the pooled metric over `raw_mean` by **+0.023016** and improves folds 1 and 3, so it meets the predeclared advancement gate. This is an information-gain decision, not model promotion. The paired conditional 95% interval for the diagonal contrast still crosses zero at all tested block sizes: 10 dates `[-0.040618, 0.089025]`, 20 dates `[-0.034055, 0.081588]`, and 40 dates `[-0.030487, 0.078587]`. Simultaneous intervals also include zero.

**Decision:** advance only the diagonal rank direction to a small matched fitted ablation. Do not tune the covariance methods or reinterpret this no-fit control as a forecasting model. Preserve `current_market` at 0.309709 as the highest reproduced fitted development point estimate until a matched fitted rank-prior experiment actually exceeds it under the same frozen design.

Public execution evidence is in `reports/rank_prior_probe_execution.json`; private aggregate probe/lineage/log receipts are checksum-pinned in that receipt.

## Predeclared matched fitted ablation

The fitted follow-up is a separate child lineage under `domain/rank_prior_fit/`; it does not modify the frozen no-fit probe package or any historical parent source. It adds exactly one fold-local `diagonal_rank` template to two frozen representations:

1. `admitted_tail_rank`: the admitted-tail control plus the diagonal-rank template.
2. `current_market_rank`: the current best compact OHLC/activity representation plus the same diagonal-rank template.

The diagonal rank score is re-estimated separately for each outer fold using only `y[:train_stop]` for that fold and then repeated as a target-aligned static template across dates. No validation label enters the feature. Existing `admitted_tail` and `current_market` predictions are reused as matched controls; they are not refit.

The experiment is capped at **six new fits** (two variants × three folds) and **180 cumulative seconds**. It begins with exactly two first-fold fits. The full four remaining fits may continue only if at least one rank-augmented variant is not more than 0.03 below its own matched control on that first fold and all lineage, admission, replay, and timing checks pass. This gate prevents one reused development fold from selecting a winner while still stopping a clearly harmful formulation.

All fitting uses the existing fixed pooled histogram configuration, residual weight one, and the same admitted train-only preprocessing policy. The rank template itself must be admitted; any lineage mismatch, future dependence, missing parent seal, replay difference above `1e-12`, final-boundary change, or cumulative deadline is a hard stop. No covariance search, model hyperparameter tuning, GPU, final-test evaluation, or Kaggle submission is declared.

Primary matched contrasts are `admitted_tail_rank` vs `admitted_tail` and `current_market_rank` vs `current_market`. The combined current-market rank model is also compared with the admitted-tail rank model and historical means.

## Measured fitted transfer test

The first-fold gate fit exactly two models. The diagonal-rank template was admitted in both, both checkpoints replayed exactly, and both predictions tied their matched controls on that fold. The predeclared continuation rule therefore allowed the remaining four fits. The full continuation reused those two checkpoints and fit only folds 2–3 for the same two declared variants. All six saved models replay with maximum absolute prediction difference **0.0**; the study preserves **7 sealed checkpoints** including its summary and expands the joint comparison family to **279 declared contrasts**.

| Fitted representation | Official metric | Fold 1 | Fold 2 | Fold 3 | Delta vs matched control |
|---|---:|---:|---:|---:|---:|
| admitted_tail control | 0.305336 | 0.390579 | 0.152341 | 0.399881 | — |
| admitted_tail + diagonal rank | 0.301772 | 0.390579 | 0.160756 | 0.379391 | **-0.003564** |
| current_market control | **0.309709** | 0.403381 | 0.167042 | 0.390619 | — |
| current_market + diagonal rank | 0.307373 | 0.403381 | 0.161112 | 0.388073 | **-0.002335** |

The no-fit prior did **not** transfer into a useful fitted feature. For `admitted_tail_rank` versus its control, the 20-date conditional 95% interval is `[-0.013442, 0.005374]`; for `current_market_rank` versus `current_market`, it is `[-0.008679, 0.003517]`. Both simultaneous intervals are much wider and include zero. The first-fold ties also show why the staged gate matters: the rank feature can be admitted without changing a tree's actual predictions.

**Decision: stop this fitted rank-prior family.** Do not tune the variance floor, covariance method, model hyperparameters, or feature combinations to rescue it on the same repeatedly inspected folds. Preserve the positive no-fit diagnostic and the negative fitted transfer result as separate evidence. `current_market` remains the highest reproduced fitted development point estimate at **0.309709**. Feature engineering remains open, but the next experiment must test a genuinely distinct mechanism.

Public execution evidence is in `reports/rank_prior_fit_execution.json`. Private models, validation predictions, manifests, and the full comparison history remain checksum-pinned under the study lineage in the existing encrypted S3 bucket. The final 247 origins remain untouched.

## Broader feature-research backlog

The current code already represents return lags 0/1/2/5, risk-scaled returns, trend/path summaries, activity, OHLC/intraday structure, asynchrony, factor-relative innovations, macro links, FX graph structure, contract-basis proxies, pair dynamics, released priors, horizon structure, latent factors, short released own-target context, tested signed peers, bounded market-path variants, and now a fitted transfer test of stable target-rank priors. High-value unresolved directions therefore include training-only target/group structure beyond a single scalar rank prior, market-state normalization of the useful current OHLC/activity block, leakage-safe online adaptation under released labels, ranking-aware objectives, shared raw-input multi-target architectures, and externally sourced carry/inventory/macro/weather information only where point-in-time dates, instrument mapping, vintages and data rights can be verified.
