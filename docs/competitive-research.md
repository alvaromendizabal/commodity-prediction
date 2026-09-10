# Competitive benchmark and feature research audit

Verified on 2026-09-10. Feature engineering remains open. The historical final private leaderboard is now verified: **anonemaus, 0.63834, first of 1,126 teams**. Our **0.309186** uses 535 different development dates. Subtracting the two would not estimate an out-of-sample performance gap; a probability that we would have won is not supported.

## Organizer benchmark and leading methods

The live [final private leaderboard](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/leaderboard) labels the competition completed and the standings final. It reports first 0.63834, second 0.61027, third 0.60229, fifth 0.53231, sixth 0.51444 and eighth 0.49066. The browser recovered substantive content after the public text interface returned empty page shells. No submissions, new terms acceptance, or hidden-outcome inspection occurred.

| Primary source | What the author actually describes | Implication and limitation |
|---|---|---|
| [Fifth-place ZLF writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/zlf-solution-of-the-mitsui-commodity-predict) | Raw inputs, missing-value sentinel, internal normalization; four-day RNN and single-day MLP, averaged; combined MSE/ranking objective. Author reports RNN 0.509 and ensemble 0.532. | Short context and a suitable joint objective may matter more than an indiscriminate increase in engineered columns. This is author-reported model evidence on another evaluation; it does not prove an incremental feature effect here. Longer context did not help in the author's limited experiments. |
| [Sixth-place Artem writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/artem777-in-mitsui-6th-Place) | Forecast underlying series recursively five days, derive target log returns, then combine with recent released-label means. A 50-date LSTM, chronological early stopping and training-fitted imputation/scaling. | Target construction and released-history information are high-value leads. The reported blend weight is not an authorized choice on our folds. Reconcile the author's feature-count description with actual code/schema before replication. |
| [Eighth-place Transformer writeup](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/transformer-based-solution) | Numeric columns as tokens with column embeddings; jointly predicts 424 targets from raw inputs and historical target lags; masked MSE, training scaling, chronological validation. | Cross-target historical context and shared representation deserve isolation. The prose's feature counts are inconsistent (981 versus 587 + 424), and a blanket four-row lag does not establish our horizon-4 release requirement of five rows. Audit actual inference code before borrowing its timing. |

The final ranking verifies the teams' scores; it does not independently verify every implementation detail or attribution claim in their writeups. Reported training times are authors' hardware-specific descriptions, not budget estimates for this project. Later leaderboard snapshots with different test dates are not matched ablations.

## What our representation already contains

The earlier [released-history implementation](../src/commodity_prediction/studies/representation.py) already includes the latest released own-target value, expanding mean, EWM 126, 21/63-date means and volatilities, and availability. Routing explicitly restricts that history to its own target. Repeating an own-target latest-value feature is therefore not a new family.

The pooled [domain representation](../src/commodity_prediction/domain/features.py) includes EWM locations/risks at 63/126/252, robust 63-date summaries, standardized innovations, market-pair pools and canonical cross-horizon pools. These shared priors are largely long-window summaries. The latest risk-state expansion changes their conditioning; it does not expose a short sequence of individually released values across related targets.

| Information an expert or leading system can use | Existing coverage | Remaining test |
|---|---|---|
| Very recent shape of available label history | Latest own-target value in the earlier target-aware study; long-window priors in pooled models | Matched current-model comparison of exact released snapshots over 1–4 dates, short means/differences, and explicit availability |
| Related targets' newly released information | Long-window metadata pools and canonical horizon averages | Short, sign-aligned same-pair/cross-horizon and market-peer context, each constituent delayed separately |
| Short raw market sequence | Numerous causal return and path summaries; no verified reproduction of the gold RNN representation | Fixed linear/tree lag-block diagnostic before any neural sequence model; separate representation gains from architecture |
| Mechanically consistent underlying asset forecasts | Earlier asset-label/structural methods and target algebra exist | Reconcile prior experiments before testing recursive feature forecasts; future supervision must remain inside training |
| Local cross-market relationships | Currency consistency, pair features, static projected means and broad factors | Train-only correlation-peer aggregation with market-group and identity controls |
| Carry, inventory, positioning, macro surprises, seasonal supply and weather | Market proxies and documented external-source leads | Verify real dates, instrument mapping, publication timestamps, vintages and data rights before any join |

## Public-score audit

[GC-Ridge](https://github.com/Brishian427/kaggle-mitsui-gc-ridge/tree/b005f9c2894ff663dcc4037f3c7c118eba75e324) is associated with Brishian, whose **46th-place, 0.40377** result is visible on the final leaderboard. Its graph representation remains a plausible lead. However, the public [local neutralization routine](https://github.com/Brishian427/kaggle-mitsui-gc-ridge/blob/b005f9c2894ff663dcc4037f3c7c118eba75e324/src/neutralize_destroyers.py) selects weak targets using the validation outcomes and rescores modifications on those same outcomes. Its baseline initially uses a current outcome and subsequently shifts only one row, insufficient for our h+1 releases. This local post-processing score is not an unbiased comparator. This is not an allegation about the author's actual competition submission. Its 1829–1950 window overlaps our final reservation; do not replay that window or import outcome-selected target lists.

The [sunnyreddy12 repository](https://github.com/sunnyreddy12/kaggle-mitsui-commodity-prediction-challenge/tree/76a522b9f2992ebe420f436280accada61f0a0c1) advertises 1.341129. Its [calculation](https://github.com/sunnyreddy12/kaggle-mitsui-commodity-prediction-challenge/blob/76a522b9f2992ebe420f436280accada61f0a0c1/calculate_final_sharpe.py) aligns released mock-test labels by position and averages separate lag-group score ratios. That does not implement one global daily cross-sectional metric with demonstrated release-time alignment. It is not the historical winning target.

[Sosolalt](https://github.com/Sosolalt/Mitsui-Co_kaggle_competition) supplies technical/relative/volatility coverage and a self-reported 328th-place result, rather than a leading reference. [Ohmae22](https://github.com/Ohmae22/mitsui-commodity-prediction) gives generic feature/ensemble leads, but expected-score ranges are not measured results. No third-party predictions or code were executed in this audit.

## Domain evidence and admissibility

[Mantegna](https://arxiv.org/abs/cond-mat/9802256) connects return-correlation structure with financial-market hierarchies. [Kipf and Welling](https://arxiv.org/abs/1609.02907) provide a method for combining graph neighborhoods and node attributes, evaluated outside finance. These motivate a later peer-context ablation; neither proves commodity predictability. The existing [domain research ledger](domain-feature-research.md) retains the broader economic mechanisms, primary literature and failed experiments. The [risk-state study](risk-state-research.md) records the volatility-conditioning hypothesis and its inconclusive result.

The live [competition rules](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/rules) specify competition-use data rights and protection against unauthorized redistribution. External data must meet the organizer's public/equitable-access and licensing conditions. Private raw data, labels, models and predictions stay in the project bucket. Public aggregates and source do not imply unrestricted commercial rights to the competition data. An external series is admissible only after its real date/instrument mapping, historical availability and permissible use are established. Anonymous row IDs are not a verified economic calendar.

## Next bounded feature milestone

**Hypothesis:** short, correctly released target context contains information obscured by the pooled model's long-window priors. Test this before new graph or GPU work because both current ablations and independently verified gold solutions identify released history as a strong lead.

1. Freeze the current admitted-tail control, model settings, 535 validation dates and training-only screen. Inventory the earlier own-target latest-value results first. Declare three additions: own-target short context; metadata-related short peer context; their combination. At most **nine new fits across three folds**, plus an exactly reusable control.
2. Define every target input at origin t as a value released by t. Delay each horizon separately by h+1 before constructing 1–4-date snapshots, means, innovations, peer aggregates or missingness indicators. Canonicalize pair orientation before cross-horizon aggregation. Do not substitute a universal lag.
3. Test prefix equivalence, future perturbations, horizon-specific release boundaries, pair-sign orientation, missingness and serialized replay before fitting. Train-only checks remove unusable/duplicate inputs and report exact candidate/retained/rejected counts.
4. Time a small construction/prediction smoke run, then one representative fold. Cap the full exploratory fit stage at **five minutes**, with per-fit atomic manifests, 30-second heartbeats and complete reuse on resume. Stop immediately on timing, schema, replay or resource failure; inspect the cause before any retry.
5. Report all matched official metrics, every fold, short-context additions/removals and paired block uncertainty. Treat improvement concentrated in one period as unstable. Do not promote a feature family without supported contribution; do not label an inconclusive small study as exhaustion. Preserve all prior comparisons in the adaptive research history.

This study uses existing private data and CPU estimators. Its information gain is whether a specific missing time scale and related-target context explain part of our representation weakness. A negative result would retire that concrete formulation cheaply; a consistent result would justify a separate architecture comparison later.

## Competitive claim and stopping rule

The verified target is 0.63834 on the organizer's hidden evaluation. Reproduce admissible leading-method ideas on our own matched folds before attributing any gap to features, objective, shared modeling or ensembling. The current +0.000098 expansion is not a material advance. The weak middle period, 255 inconclusive simultaneous comparisons and adaptive selection remain unresolved.

A final gate requires an audited coverage ledger with each major avenue tested, explicitly ruled out by valid prerequisites, or supported as unnecessary by evidence. This audit has not reached that gate. Origins 1714–1960 remain reserved for the eventual one-time final evaluation. Stronger development scores alone cannot support a claim that this system would probably have won.
