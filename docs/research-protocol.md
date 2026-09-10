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

### Follow-up study: target-aware representation

The follow-up is defined in `configs/feature_study.json` and `commodity_prediction.studies`. It preserves the original source/data fingerprint and all initial fitted checkpoints. It adds a five-date terminal embargo: the final development validation block ends at date 1703, so even its four-date target is fully observable by date 1708, before the reserved interval begins at 1709. The first two 180-date folds and all training boundaries remain unchanged; the last validation block contains 175 dates. Initial predictions are rescored on these same 535 dates without refitting. The earlier 540-date figures remain historical records and must not be compared directly with the corrected window.

Rescoring does not undo an earlier inspection. The initial validation already inspected forward outcomes through date 1713. Therefore origins 1709–1713 are a permanent boundary buffer, and only origins **1714–1960 (247 dates)** qualify for the eventual untouched final test. `configs/final_evaluation.json` records that restriction. None of those 247 final-test outcomes has been used for selection or evaluation. The nominal 252-row reservation remains outside development, but it is not described as 252 pristine outcomes.

The completed experimental summary retains its original checksum and source lineage. `scripts/review_feature_study.py` corrects its legacy 252-date wording in the public report before notebook execution, recording the immutable summary checksum and the final-test configuration checksum. This is an explicit reporting correction; no numerical result, selected feature, prediction, or model checkpoint changes.

The study declares 23 variants before its full run. It includes frozen training means and expanding means of released labels; target-specific selection from the original global pool; screening that requires a consistent correlation sign across the two training halves; metadata-directed routing to own assets, the exact target pair, and market aggregates; and a structural model that forecasts asset returns at each horizon before subtracting pair components. The feature budget is 24 per modeled output, with Ridge alpha held at 100. This new budget differs from the original shared 96-feature study, so the new within-study references are the controlled feature comparisons.

Three additional families test longer risk regimes, nonlinear interactions of observed market quantities, and released-label history. Regime features include 126/252-date returns, risk ratios, skewness, kurtosis, sign persistence, autocorrelation, drawdown, and trailing percentiles. Interactions combine momentum, relative returns, volatility, and reversion. Label history is shifted by exactly h+1 before any expanding, rolling, or exponentially weighted aggregation. During validation, earlier labels can enter only after release; the fitted model remains frozen. These are chronological history features, not random-fold target encodings. Some reproduce already observed price returns, so duplicate checks and drop-family ablations determine whether they add anything.

Each non-reference family is removed in a separate conditional ablation. A small histogram-gradient-boosting control compares the same aligned reference and extended representations, with 48 iterations, seven leaves, fixed regularization, and automatic early stopping disabled to avoid a random validation subset. This is a nonlinear diagnostic, not a hyperparameter search or final optimization. [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

Every output records its candidate, retained, and rejected counts and rejection reasons. Counts summed across target/output models are explicitly called assignments; they must not be confused with unique generated features. The study also measures adjacent-fold selection overlap and joint feature-family block permutation. Correlated predictors can substitute for one another, and permuted values need not remain on the observed data distribution, so these diagnostics are not causal attributions. [Permutation-importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html).

Paired bootstrap intervals use 2,000 resamples, with 10-, 20-, and 40-date block sensitivity. Blocks remain inside their original validation fold. Centered maximum-error bounds cover the declared comparison set simultaneously under the resampling approximation. These bounds condition on the fitted models and do not account for all earlier adaptive research or rescreening. No final model or feature gate is selected merely because one point estimate leads.

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
