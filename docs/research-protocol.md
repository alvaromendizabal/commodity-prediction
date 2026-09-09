# Research protocol

The initial experiment is deliberately limited to feature research with a fixed linear diagnostic model. It supplies a reproducible reference, rather than a final forecasting system.

## Prediction contract

For horizon h, target(t) uses log P(t+h+1) minus log P(t+1). Pair targets subtract the corresponding second asset return. Official release files show offsets of h+1. Current-row market data is available before prediction. All engineered features use observations at or before t.

The 1,961 ordered dates are divided into 1,709 development dates and 252 reserved dates. Development validation comprises three consecutive 180-date blocks, with expanding training and five purged dates before each block. The downloadable test set overlaps the training period and is not used to estimate generalization. Date IDs are not decoded into a guessed calendar.

## First feature search

| Family | Hypothesis | Availability and main risk |
| --- | --- | --- |
| One-date returns | Simple cross-market reference | Current and preceding observed prices; no fill |
| Momentum | Persistence and changes in trend | Trailing horizons 2–63; no future/centered windows |
| Volatility | Risk state and downside asymmetry | Trailing 5/21/63-date windows; unstable low denominators masked |
| Reversion | Departure from recent price levels | Trailing z-scores and volatility-scaled momentum |
| Missingness | Closed markets and stale observations | Missing flags and elapsed observed-date age |
| Liquidity | Trading activity and participation changes | Current/lagged volume and open interest; train-only scaling |
| OHLC | Intraday range, close location, and overnight gap | Current-row OHLC; never future price bars |
| Cross-market | Relative strength, breadth, and shared risk | Contemporaneous observed peer ranks and market aggregates |
| Target pairs | Relative returns and changing dependence | Metadata-defined spreads, 21-date z-scores, trailing 63-date correlation/beta |

Global selection is fit separately on each fold and ablation, before seeing its validation labels. Every target model excludes missing fitting labels. There is no broad hyperparameter search. The 96-feature cap is a diagnostic compute/regularization budget, not evidence that every rejected feature lacks economic signal. Adding a family can displace reference features under this cap; conditional drop-family ablations are still needed.

## Metric and uncertainty

Use average ranks across observed targets within each date, then the mean daily correlation divided by its population standard deviation. Missing solution filler -999999 is masked. Misordered target columns/row IDs, invalid predictions, and degenerate correlations fail explicitly. Do not annualize this metric or interpret it as financial return Sharpe.

Pooled OOF scores concatenate the three nonoverlapping validation periods. A paired circular block bootstrap uses 20-date blocks and 500 draws. It conditions on already fitted predictions and does not cover refitting uncertainty or multiplicity. A higher point estimate alone does not authorize promotion.

## Remaining gate evidence

1. Conditional drop-family ablations and permutation importance under correlated predictors.
2. Target/pair-specific screening and simpler structural return forecasts versus globally shared selection.
3. Nonlinear controls to distinguish genuinely weak features from linear-model limitations.
4. Longer historical windows, interactions, regime sensitivity, and fold-stable selection.
5. Release-aware target pooling/encoding only where it is useful and not a redundant re-expression of known prices.
6. A documented decision on external data, source vintages, calendar alignment, availability, and permitted use.
7. Stronger uncertainty and multiple-comparison assessment, followed by a reviewed feature-gate decision.

Only after those findings support diminishing returns should final model optimization, calibration/rank transforms where useful, and reserved-holdout evaluation begin.

## Primary sources

- [Competition dataset and release semantics](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data)
- [Organizer's target calculation](https://www.kaggle.com/code/sohier/mitsui-target-calculation-example/)
- [Official metric implementation](https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric)
- [Competition data-use rules](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/rules)

Data is subject to the competition rules, including the Competition Use Only restriction and limits on redistribution. This repository does not distribute competition records, labels, or row-level predictions. External deployment or a separate commercial dataset would require its own appropriate data rights.
