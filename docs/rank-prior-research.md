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

## Broader feature-research backlog

The current code already represents return lags 0/1/2/5, risk-scaled returns, trend/path summaries, activity, OHLC/intraday structure, asynchrony, factor-relative innovations, macro links, FX graph structure, contract-basis proxies, pair dynamics, released priors, horizon structure, latent factors, short released own-target context, tested signed peers, and the bounded market-path variants. High-value unresolved directions therefore include training-only target/group structure, market-state normalization of the useful current OHLC/activity block, online adaptation under released labels, ranking-aware objectives, shared raw-input architectures, and externally sourced carry/inventory/macro/weather information only where point-in-time dates, instrument mapping, vintages and data rights can be verified.
