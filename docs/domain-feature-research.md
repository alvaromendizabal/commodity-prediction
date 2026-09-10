# Domain feature research

Feature engineering remains the central research gate. This study asks whether economic relationships and a shared target-relative representation can improve on the preserved historical-mean control. It does not open the final test or select a production model.

## What the data supports

The supplied panel contains LME metals, JPX precious-metal futures, US equities and ETFs, and FX quotes. Targets combine 106 distinct assets into 424 horizon/pair outputs. Price units, currencies, observation schedules, and availability differ. A useful representation therefore separates shared market movements, asset-specific innovations, relative exposures, and information released over time.

The new panel uses **380 input templates** across a reference family and **13 domain families**. These instantiate **29,917 source series** before projection and **161,120 target-template assignments**. These are different counting units, not independent signals. Some series transform earlier hypotheses; shared market contexts repeat across targets; structural availability differs by asset. Counts are machine-checked against the executed inventory.

For an asset feature f, paired targets receive f(left)−f(right) and f(left)+f(right); single-asset targets use a zero right leg. Features unavailable for an entire asset class use a structural zero plus an applicable-leg indicator. Missing observations within a supported instrument remain missing. Direct spread features and release-safe target history retain target-specific construction.

## Economic hypotheses and implemented tests

| Family | Hypothesis | Implemented representation and limits |
|---|---|---|
| Reference | Recent returns, risk, and known exposure are a necessary benchmark. | Returns at lags 0/1/2/5, risk normalization, 63-date risk, horizon, pair status, signed market exposure. |
| Trend shape | Persistence and reversal may depend on where a move occurred within a path. | Nonoverlapping 1–5/5–21/21–63/63–126/126–252 segments; path efficiency; fast/slow EMA separation; raw and risk-normalized peer momentum ranks. Published longer-horizon momentum motivates a hypothesis, not proof for these short targets. |
| Tail risk | Downside asymmetry and unusually large moves may alter near-term behavior. | Median/IQR shocks, semivariance balance, jump frequency and pressure, tail quantiles, drawdowns and drawups. Jump thresholds use previous-date volatility. |
| Trading activity | The same return may mean something different under unusual trading activity. | Relative volume, signed pressure, volume acceleration, activity-normalized price impact, volume-weighted returns, and available volume/open-interest and price/position interactions. Dollar-volume proxies are not comparable contract notionals. |
| Intraday path | Overnight and within-session repricing may have different persistence. | Overnight gaps, intraday returns, gap/reversal interaction, multiday component trends, high–low and OHLC path-risk ratios. Historical session timestamps are not supplied. |
| Asynchrony | Stale or resumed observations can distort ordinary returns. | Observation age, gap length, resumed trading, observed movement per elapsed date and risk, zero-return frequency. Forward filling is confined to explicit last-observation calculations. |
| Factor relative | Broad risk movements may obscure useful asset-specific changes. | Trailing factor betas, current residual innovations, residual trends, and lagged-response exposures to equity, gold, energy, materials, Treasury, yen, and AUD proxies. Coefficients use data through t−1. |
| Macro links | Related traded markets can summarize economic conditions. | Equity/metal/energy/bond/FX contexts; miners–gold, silver–gold, energy–materials, credit–Treasury, duration, inflation-linked–nominal, and emerging–global contrasts. ETF return differences are not measured credit spreads or real yields. |
| Currency graph | FX returns should approximately share an internally consistent currency structure. | Cross-sectional currency strengths with USD as gauge, implied quote returns, cycle residuals, and trailing persistence. Missing quotes require an identifiable graph. These are consistency proxies, not executable arbitrage. |
| Contract basis | Related contract forms and settlement marks may contain relative information. | Gold/platinum mini versus standard, rolling versus standard gold, close versus settlement, yen-adjusted JPX returns, and relative JPX/GLD movements. Maturities, units, fees, and asynchronous closes prevent a true carry/parity interpretation. |
| Pair dynamics | Relative price movement can depend on covariance and adjustment speed. | Rolling correlation, risk ratio, spread risk, lagged hedge residuals, standardized relative levels, error-correction slopes and reversion pressure, asymmetric cross-lag products. Rolling slopes are not a cointegration finding. |
| Released priors | Previously observed target behavior may benefit from conservative pooling. | Horizon+1 delayed EWM location/risk, robust summaries, availability, sign frequency, market-pair pooling and shrinkage, orientation-corrected cross-horizon pooling. Every constituent is released before aggregation. |
| Horizon structure | One- to four-date targets need not share identical dynamics. | Observed backward-looking returns and variance ratios over 1/2/3/4 dates, interpreted with explicit horizon metadata. No forward target reconstruction enters predictors. |
| Latent factors | Several observed proxies may miss common return structure. | Three trailing principal components over 126 dates, refit every 21 dates; common/idiosyncratic return components and loading concentration. Fitting uses rows strictly before the current date. |

## Market conventions that constrain interpretation

FX base/quote orientation matters: the first currency is the base and the second the quote. The graph uses these signs consistently; converting a yen price into dollars divides by USD/JPY. Tests check both a consistent FX triangle and this conversion. [CME quote conventions](https://www.cmegroup.com/education/courses/introduction-to-fx/understanding-fx-quote-conventions).

JPX standard gold is quoted in yen per gram with a one-kilogram contract. Rolling gold uses a smaller contract and a different settlement mechanism. Consequently, differences between their observed series deserve investigation, but the panel does not identify a dated delivery curve. Current exchange specifications establish instrument meaning; they are not retroactively applied as a historical calendar. [JPX standard gold](https://www.jpx.co.jp/english/derivatives/products/precious-metals/gold-standard-futures/01.html), [JPX rolling gold](https://www.jpx.co.jp/english/derivatives/products/precious-metals/gold-rolling-spot-futures/01.html).

LME copper is quoted in US dollars per tonne and trades across prompt dates. A single unnamed price stream does not identify a cash-to-three-month curve. GLD seeks gold-bullion price exposure less expenses; its share price is not a directly interchangeable physical gold quote. The Japan/US gold feature therefore tests relative movement, not physical delivery parity. [LME copper specifications](https://www.lme.com/Metals/Non-ferrous/LME-Copper/Contract-specifications), [GLD issuer](https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld).

IEF represents seven- to ten-year US Treasury exposure. Its return reflects duration and other bond-market effects, rather than directly measuring a yield change. [IEF issuer](https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf).

Research on time-series momentum across asset classes motivates testing path persistence, while cross-asset value/momentum research motivates separating broad and relative movements. Those studies do not establish predictive value for this dataset's 1–4-date targets, and price-only ratios are not fundamental value measures. [Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum), [Value and Momentum Everywhere](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere).

## Controlled evaluation

The three outer validation intervals remain 1169–1348, 1349–1528, and 1529–1703: **535 dates**. The previous 69 model results remain intact. Origins 1709–1713 are a permanent boundary buffer, and only origins **1714–1960 (247 dates)** qualify for eventual untouched final testing.

The study fixes one pooled ridge model and one histogram-tree diagnostic configuration. Observations are date/target pairs. Target means and scales are estimated within each fit; models predict standardized residuals above these means. Labels are clipped only for fitting. The fitting loss gives each observed date equal total weight. Training-only missingness, constant, duplicate, correlation, relevance, and optional two-period sign-stability screening select at most 64 pooled templates. Counts refer to pooled columns, not 424 independently fitted outputs.

Each outer training range contains two 126-date chronological inner validation blocks with a five-date purge. Each inner fit independently estimates target normalization, preprocessing, screening, and coefficients. Only these inner predictions select residual strength from 0, .05, .1, .25, .5, or 1. Outer validation never chooses this weight. Unshrunk results are also published to expose weak or unstable features that calibration suppresses.

The prespecified design has 31 fitted variants: reference; 13 single-family additions; all families; stable screening; 13 family removals; tree reference; tree all. Three outer folds and two inner folds produce **279 new fitted models**. Historical mean, graph-projected mean, and 1%/99% winsorized mean add nine outer control evaluations. This gives **102 outer evaluations**, with raw and calibrated predictions retained privately.

Matched additions compare with the reference; removals compare with the full set; raw variants use the same comparisons. The official metric is mean daily cross-sectional Spearman correlation divided by its population standard deviation, without annualization. Paired circular blocks of 10, 20, and 40 dates are resampled within outer folds, with 2,000 repetitions and simultaneous centered maximum-error bounds over the declared comparisons. These intervals condition on fitted models and do not undo repeated research on the development period.

Additional diagnostics cover horizon scores, 60-date temporal blocks, selected-template overlap across folds, selected residual weights, and five repeated 20-date block permutations of each selected family in the unshrunk full model. Permutation measures model reliance under a distribution change, not causality; correlated alternatives can make it small.

## Broader domain ledger: unresolved prerequisites

| Avenue | Why it could matter | What is required before a valid experiment |
|---|---|---|
| Futures term structure, carry, convenience yield, roll effects | Storage and delivery economics distinguish scarcity from spot movement. | Historical contract identifiers, expiries, settlement curves, contract specifications and a dated roll policy. Mini/standard differences cannot replace them. |
| Inventories, warehouse stocks, positioning and producer hedging | Physical tightness and speculative crowding may influence relative prices. | Licensed historical releases with publication timestamps, revisions, and instrument mapping; no forward-filled later vintages. |
| Weather, seasonality and industrial demand | Supply shocks and seasonal demand may differ by commodity. | A verified mapping from anonymous date IDs to calendar dates, geographically relevant historical forecasts/releases, and provenance. |
| Monetary policy, inflation and economic surprises | Expectations and policy shifts affect currencies, metals and duration. | Release-time actuals, contemporaneous expectations and revision vintages. ETF proxies only partially address this. |
| Fundamentals and supply-chain equity exposures | Miners, producers and users have different commodity sensitivities. | Historical filings, point-in-time issuer classification and holdings; current classifications cannot be assumed historically. |
| Options and microstructure | Implied risk, skew, liquidity and order flow can add forward-looking information. | Dated option surfaces, bid/ask or order-book data, exchange timestamps and suitable usage rights. |
| Text and event features | News may identify disruptions that prices only partly reflect. | Timestamped licensed text, deduplication, first-publication times and strict as-of retrieval. LLM-generated retrospective narratives are not predictors. |
| Conditional regimes, interactions and nonlinear structure | A family may work only for a subset of targets or market states. | Controlled interaction/target-group experiments after this broad study; previous regime/interactions results are already preserved and cannot be relabeled as new evidence. |

These are research prerequisites, not invitations to invent unavailable data. The gate stays open while feasible, materially distinct hypotheses remain untested or evidence for diminishing returns is insufficient. A large feature count alone cannot close it.

## Reproduction and evidence

Run `uv run --frozen python -m commodity_prediction.domain.run --sync-s3` in the authorized AWS workspace after restoring the earlier study. Every feature panel, inner fit, outer fit, permutation stage, and summary has a SHA-256 manifest; a valid complete resume verifies artifacts and performs no fitting or preprocessing. The dependency fingerprint includes the new domain modules, configuration, final-evaluation boundary, and the preserved parent lineage. Numerical summaries appear in `reports/domain_study.json`; the canonical `02_feature_research.ipynb` presents the executed evidence.
