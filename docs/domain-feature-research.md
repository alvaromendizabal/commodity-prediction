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

## Adaptive nonlinear attribution

The executed domain tree improved its point estimate to 0.270612 (nested calibration), versus 0.217739 for historical means. The conditional 20-date-block interval for that difference is −0.049773 to +0.153154; simultaneous bounds also include zero. The unshrunk tree scores 0.276887. These observations motivate **an explicitly exploratory follow-up**, not model promotion.

The follow-up holds the parent tree configuration, feature panel, 64-template screening budget, outer folds, and residual weight one fixed. It adds each of the 13 families to the reference and removes each from the full representation: **78 new tree fits** across three folds. It reuses the parent reference/full/mean controls and repeats family permutation on the saved full tree. No parent feature generation or fitting is repeated, and no residual-weight or model hyperparameter selection is added.

Uncertainty is computed jointly over the parent domain study's 122 comparisons and the follow-up's 55 comparisons (**177 total**, including repeated controls). This broader bound still does not correct for every earlier adaptive research decision or represent independent confirmation. The child package under `domain/attribution/` has a separate fingerprint and preserves the completed parent source lineage. Its aggregate evidence is `reports/tree_attribution.json`.

The follow-up completed all 78 fits. The reference plus released priors scores **0.281237**, compared with **0.205111** for the raw tree reference. Their paired difference is **+0.076126**, with conditional 95% bounds **[−0.020015, +0.174534]** at block length 20. Removing released priors from the full tree costs **0.045268**, with bounds **[−0.007302, +0.100175]**. The same family has the largest mean fold-metric decrease under permutation, **0.131271**. Agreement among these diagnostics identifies a credible research lead; uncertainty prevents a validated contribution claim.

The best exploratory ablation point estimate, **0.284713**, drops intraday features. Dropping pair dynamics, currency-graph features, or latent factors also does not lower the full tree's pooled point estimate. These results do not establish that those mechanisms are universally useless: screening can substitute correlated inputs, and a pooled representation can hide subgroup effects. No positive simultaneous lower bound appears among the 177 joint comparisons at any tested block length (10, 20, or 40 dates).

## Earlier verified checkpoint and open decisions

All 357 new fitted models completed. AWS verified 480 stage manifests across the four preserved study lineages and replayed 435 saved models/controls with maximum prediction difference **0.0**. The three canonical notebooks executed in fresh kernels and contain 22 Plotly/static figure pairs. Local quality checks pass 66 tests, formatting, linting, and type checking. This is a verified research checkpoint, not feature-gate closure.

The next feasible questions are whether the 64-column relevance screen suppresses complementary nonlinear information; whether domain effects differ by market, pair type, or horizon; whether released priors retain their benefit with additional release delays and conservative historical pooling; and whether interactions between trend, risk, freshness, and activity explain temporal instability. Those experiments must preserve the current lineages and compare fixed representations under the same validation protocol. They are not yet completed by this checkpoint.

## Economic coverage and external-data audit

The prediction object is a short-horizon return or return difference, not a physical commodity price forecast in isolation. An explanatory variable for the level of copper prices need not improve the ranking of a copper–equity return difference over the next few observations. Every proposal must specify which leg it affects, its sign convention, the horizon over which adjustment is plausible, and what information has already entered the market price. The mechanism motivates a test; it does not determine the result.

The review distinguishes four evidential states: an implemented and evaluated representation; an inconclusive or unfavorable matched experiment; a feasible sensitivity that still needs results; and a blocked mechanism requiring data not present in the supplied panel. A blocked mechanism is not a rejected hypothesis. An unsuccessful proxy does not falsify its underlying economic theory. A favorable point estimate does not establish a dependable predictor.

### Storage, delivery curves, and physical scarcity

Storage, financing, and insurance can influence the relationship between cash and futures prices. The benefit of possessing deliverable inventory can change that relationship when material is scarce. CME's explanation of contango and backwardation connects these quantities and describes convergence as delivery approaches. This supports hypotheses involving curve slope and scarcity; it does not imply that a price-only spread is a valid inventory measure. [CME, *What is Contango and Backwardation*](https://www.cmegroup.com/education/courses/introduction-to-ferrous-metals/what-is-contango-and-backwardation).

With dated contracts, a useful proposed representation is the signed log-price difference divided by the difference in time to delivery, with the direction of the sign documented. Other candidates include nearby/deferred slope, curvature, spread volatility, curve changes, and interactions with inventory tightness. These require observed prices for identified maturities. Continuous-contract back adjustments, changing nearby contracts, and a revised roll convention can create retrospective features that were not available historically. The present mini/standard and settlement/close experiments address related observed-price mechanisms, not this entire curve family.

LME prompt dates require special treatment. The exchange describes daily prompts through three months, weekly prompts through six months, and monthly prompts further out, with metal-specific limits and holiday substitutions. A generic monthly-futures calendar is therefore insufficient. Historical notices and actual prompt identifiers are prerequisites for expiry-distance, roll-window, and holiday-sensitive features. The current exchange page provides a semantic guide, not a historical calendar for anonymous date IDs. [LME, *Prompt date structure*](https://www.lme.com/sustainability-and-physical-markets/physical-market-benefits/prompt-date-structure).

Cross-asset carry research finds predictive relationships across several asset classes and adverse common episodes. This makes carry a serious candidate family and suggests testing its interaction with stress. It is evidence from a different sample and strategy horizon; it does not validate one- to four-date forecasts in this panel. FX spot momentum cannot substitute for forward discounts or interest-rate differentials. Bond ETF return contrasts cannot substitute for a measured yield curve. [Koijen, Moskowitz, Pedersen and Vrugt, *Carry*, 2018](https://www.aqr.com/Insights/Research/Journal-Article/Carry).

### Inventories, warrants, and positioning

For industrial metals, proposed physical-state features include log inventory changes, withdrawals relative to opening stocks, cancelled-warrant shares, concentration across warehouses, queue changes, and trailing inventory percentiles. A seasonal inventory deficit would require both a calendar and historical seasonal estimation. Reported exchange stocks are not the entire global physical inventory; warehouse transfers and reporting changes can produce large movements without equivalent consumption changes.

The LME currently distinguishes daily two-day-delayed public stock breakdowns, one-day-delayed detailed vendor reports, monthly delayed warehouse/queue reports, and historical reports through its data portal. Availability must be encoded per source, not inferred from the stock observation date. Purchasing a current feed would not by itself establish a reproducible historical as-of series. [LME, *Warehouse and stock reports*](https://www.lme.com/Market-data/Reports-and-data/Warehouse-and-stocks-reports).

Positioning requires a different release contract for each exchange. CFTC COT generally describes Tuesday positions released Friday at 3:30 p.m. Eastern time, with schedule exceptions. LME COTR generally describes Friday positions released the following Tuesday, with holiday adjustments. Candidate features include category net length divided by open interest, changes, concentration, and trailing crowding percentiles. Category names identify predominant business activity rather than the intent behind each trade; they must not be treated as a perfect separation of speculation and hedging. [CFTC, *Commitments of Traders*](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm), [LME, *Commitments of traders*](https://www.lme.com/market-data/reports-and-data/commitments-of-traders).

These sources are identified and their timing constraints reviewed. Their histories have not been joined to this project. A verified date mapping, source coverage, suitable usage rights, and a historical publication ledger remain necessary. Do not backdate Friday's CFTC report to Tuesday or Tuesday's LME report to Friday. Do not infer release availability from a report's filename alone.

### Macro policy, fundamentals, and event information

Macroeconomic features should separate the economic reference period, publication time, and revision vintage. Candidates include policy-rate changes, nominal/real yield-curve movements, released inflation and output surprises, and interactions with commodity-sensitive currencies. A surprise requires the expectation known before the announcement; subtracting a retrospectively fitted forecast or using a later consensus is a different hypothesis. Day-level observations also need a rule for releases occurring after the relevant market close.

FRED's default real-time period reflects information available today. ALFRED-style real-time periods support retrieving what was known at a historical date. Even a correct vintage query does not automatically provide intraday release timing or prove alignment with this competition's anonymous dates. An external adapter must preserve the original source, observation date, availability time, revision identifier, and retrieval provenance, and join on availability. [Federal Reserve Bank of St. Louis, *Real-Time Periods*](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

For equities, plausible channels include commodity input costs, output exposure, leverage, inventories, geographic revenue, and earnings surprises. The SEC APIs expose filing history and XBRL facts, but calendar-aligned frames can use the last-filed value. A feature must instead select a filing available at prediction time and preserve amendments, units, and accounting context. Current tickers, issuer classifications, and ETF holdings are insufficient historical identifiers. The appropriate tests would contrast issuer-specific exposures with broad sector proxies, using dated entity mappings. [SEC, *EDGAR Application Programming Interfaces*, updated 2025](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

Weather and shipping are commodity-specific hypotheses. Mining and refining disruptions differ from harvest and heating-demand shocks. Suitable candidates could use forecasts available at the time, geographically weighted anomalies, shipping delays, or announced outages. Realized weather over a future target window would leak. A broad weather feature unrelated to the underlying production geography would add complexity without a clear channel. No such external series has been validated for this panel.

Options could provide implied volatility, skew, term structure, or risk-reversal information if historical surfaces and timestamps were available. Daily volume and price-range features are only observed-market proxies. They cannot recover order imbalance, executable spreads, or option-implied expectations. Similarly, news/event embeddings require first-publication timestamps, duplicate handling, relevant entity mapping, and a corpus with known historical coverage. Retrospective generated explanations are not admissible predictors.

### Remaining internal representation risks

Screening and feature value are separate questions. A training correlation screen is computationally useful but can omit features whose value is conditional. A low ranking can reflect nonlinear dependence, target mixtures, small applicable subgroups, or genuinely weak information. The robustness study therefore changes the feature budget while keeping the model fixed, relaxes near-correlation exclusion while retaining exact-duplicate checks, and tests compact representations against their matched controls. Removing all budget restrictions is a diagnostic, not a preferred production policy.

The 56 conditional-prior templates multiply 14 release-safe summaries by four known metadata gates: long horizon and FX, US, or metals leg share. These tests expose some conditional structure without selecting subgroups based on validation performance. The 12 mechanism products combine bounded contemporaneous risk-scaled return or relative momentum with a shock, activity, or resumed-observation state. Bounded transforms avoid unbounded products; their saturation is itself a modeling assumption. They are deliberately a compact mechanistic set, not every possible pairwise product.

Additional lag stress tests ask whether information remains useful after removing the most recent observations. They are stricter than the recorded data contract. A degradation under an extra delay can be consistent with short-lived signal and does not itself prove leakage; a stable result under delay cannot prove historical timestamp correctness. The original release tests and source-level availability audits remain necessary.

Subgroup scores examine single versus paired targets and market-involved target sets. Those groups overlap, and their within-date rankings contain different comparisons from the global metric. Use them to locate instability and define future hypotheses, not to choose an attractive subgroup and report its score as overall performance. Future group-specific fits or interaction selection would need a new prespecified exploratory study and appropriate controls.

## Coverage standard and decision record

| Question | Evidence already available | What a defensible closure would require |
|---|---|---|
| Are market observations and labels causally timed? | Target reconstruction, release-delay mutation tests, strictly past factor fits, observation-age features. | Verified historical timestamps for any new external source; conservative sensitivity results interpreted separately. |
| Are trends, reversal, risk, liquidity, and relative prices represented? | Earlier 15,098-column search and 13 domain-family additions/removals with frozen controls. | Stability and matched contribution evidence; budget and conditional tests where pooled screening can hide signal. |
| Are target-history features scientifically attributable? | Nested calibration, raw/calibrated results, released-prior additions/removals, family permutation. | Robustness to timing, group composition, and pooled-history assumptions; no use of unreleased labels. |
| Are currencies, units, and contract forms handled correctly? | FX triangle/conversion tests, return-based projections, explicit applicability. | Dated maturity and physical-unit metadata before claiming carry or parity. |
| Does broader feature count imply more information? | Candidate, retained, rejected, and rejection-reason inventories. | Evidence that extra templates add stable information; exact duplicates and budget exclusions distinguished. |
| Is apparent improvement stable? | Three outer folds, horizon diagnostics, block sensitivity, simultaneous intervals. | No dependence on a single favorable period; uncertainty compatible with the claimed effect. |
| Are fundamentals and events covered? | Named authoritative sources and availability audit. | Acquired, mapped, licensed where needed, point-in-time histories and executed matched experiments. |
| Can the evidence be reproduced? | Preserved lineages, sealed checkpoints, exact model replays, fresh-kernel notebooks, CI. | The same verification for every follow-up before publication. |

The standard is not “a large number of columns” or “all tests pass.” Feature-gate closure requires a defensible coverage argument and evidence of diminishing marginal benefit from the feasible, distinct high-value hypotheses. Unavailable information must be described as unavailable, not silently replaced with a misleading proxy. Repeated development experiments are exploratory, and adding more comparisons cannot manufacture an independent confirmation set. The gate remains open until the empirical results support a narrower conclusion about exhausted avenues; neither a completion percentage nor a 9.9/10 claim is justified by this checkpoint alone.

## Executed robustness follow-up

All **36 fixed-model fits** completed across the same three outer folds. The previous sections' 357-fit checkpoint is preserved; the three domain phases now contain **393 new fitted models**. This follow-up changes feature representations, information delays, and screening constraints. It does not tune tree depth, learning rate, iteration count, or residual weight using validation outcomes.

| Hypothesis or sensitivity | Pooled metric | Interpretation under this design |
|---|---:|---|
| Frozen full tree, 64 retained templates | 0.276887 | Matched parent control. |
| Reduce budget to 32 | 0.256377 | Narrowing the full screen hurts the point estimate. |
| Increase budget to 128 | 0.253915 | More retained templates do not improve this fixed estimator. |
| Remove budget cap | 0.240974 | Retains 363–364 templates; no point-estimate benefit. |
| Also relax near-correlation exclusion | 0.263243 | Retains 376–377; exact duplicates remain excluded. |
| Delay released priors one more date | 0.277388 | Close to the control; not proof of timing correctness. |
| Delay released priors five more dates | 0.249322 | Recent released information appears useful at the point-estimate level. |
| Delay market observations one date | 0.266008 | A conservative timing stress, with static metadata and priors unchanged. |
| Add 56 metadata-conditioned priors | 0.228681 | Retains 14–17 new interactions but displaces other inputs and performs worse. |
| Add 12 bounded mechanism products | 0.276887 | No product is retained; predictions equal the frozen control. |
| Compact reference + priors control | 0.281237 | Parent attribution result reused exactly. |
| Compact reference + conditional priors | 0.206819 | Retains 33–35 interactions; does not improve the compact control. |
| Compact reference + priors + factor-relative features | 0.258062 | No pooled benefit under this fixed configuration. |
| Compact reference + priors + tail risk + freshness | **0.297055** | Strongest exploratory point estimate; conditional contribution and stability remain unresolved. |

The 68 new interaction templates expand the research inventory from 380 to **448 distinct templates across experiments**. They are not all present in one fit: the largest candidate panel has 436 templates, while the strongest compact candidate begins with 81 and retains 64. The compact candidate retains 13 released-prior templates, 25–26 tail-risk templates, seven freshness templates, and 18–19 reference templates. Counting transformations does not count independent sources of information.

### What the improvement does and does not establish

The compact risk/freshness representation scores **0.379097, 0.161987, and 0.377367** across folds. Compared with historical means, the pooled gain is **+0.079316**; the conditional 20-date-block 95% interval is **[−0.017891, +0.168043]**. Its corresponding gains over the full tree and compact-priors control are **+0.020167** and **+0.015818**, with intervals **[−0.036632, +0.080666]** and **[−0.039355, +0.069148]**.

The historical-mean comparison barely clears zero only with 40-date blocks, whose conditional interval is **[+0.001132, +0.152308]**. Ten- and 20-date blocks do not. None of the **204 joint comparisons** has a positive simultaneous lower bound at any of the three block lengths. Report this sensitivity explicitly; selecting the favorable block length after seeing results would overstate the evidence.

The middle fold still trails historical means (0.161987 versus 0.182488). The compact combination improves over the full tree in the first two folds but loses some of its third-fold gain. That pattern is compatible with a better balance of features, yet it does not establish temporal stability. It also does not identify whether tail risk, freshness, their joint presence, or the different set of screened reference/prior features explains the gain. That requires matched conditional additions/removals inside the compact representation.

The budget experiment weakens the proposition that simply admitting every pooled template improves this estimator. It does not prove the omitted mechanisms lack information under other representations. Likewise, the mechanism-product experiment is a **screening result**, not a successful test of all 12 products' predictive value: none reaches the fitted model. A future controlled experiment must admit a declared product block, compare it with the same reference, and preserve the current negative evidence.

### Reporting correction and preserved work

The original 36-fit driver completed and sealed the models before a subgroup diagnostic encountered market-closed days and single-target horizons. A subgroup with fewer than two observed targets cannot define a cross-sectional rank correlation. A single asset's horizon-specific group contains one target and therefore cannot define that ranking at all.

The corrected analysis applies a truth-only eligibility rule, reports eligible and excluded dates, and leaves mathematically undefined subgroup scores null with reasons. It does not remove dates from the global 535-date metric. Constant daily rankings are also reported as undefined rather than assigned a favorable value. The correction has a separate source fingerprint, while the original fitting fingerprint and all 36 fitted checkpoints remain unchanged. This separation prevents a reporting correction from triggering unnecessary fitting.

| Overlapping diagnostic group | Targets | Eligible / excluded dates | Full tree | Compact risk/freshness |
|---|---:|---:|---:|---:|
| FX involved | 134 | 535 / 0 | 0.274172 | 0.251089 |
| US involved | 270 | 517 / 18 | 0.183377 | 0.208702 |
| LME involved | 287 | 521 / 14 | 0.196280 | 0.208652 |
| JPX involved | 145 | 507 / 28 | 0.270488 | 0.309399 |
| Paired targets | 420 | 535 / 0 | 0.274775 | 0.295516 |
| Single-asset targets, horizons pooled | 4 | 535 / 0 | 0.110340 | 0.121132 |

The compact model's descriptive FX-involved score falls while the other listed groups improve. Its global horizon scores are 0.192234, 0.207664, 0.299497, and 0.257561, compared with 0.176931, 0.186369, 0.297863, and 0.230525 for the full tree. Those patterns suggest where further conditional tests could be informative; they are not independently validated subgroup discoveries. Markets overlap within paired targets, and pooled single-asset scores rank four different horizons of one asset rather than a broad cross-section of assets.

The expanded verification checks 517 sealed stage manifests. It preserves zero-error replay evidence for 435 unchanged earlier models/controls and independently replays all 36 new models. All three canonical notebooks are regenerated and executed against the verified analysis lineage, with 25 interactive/static figure pairs. These checks establish implementation and artifact consistency; they are not evidence of predictive superiority.

### Ordered remaining work within feature engineering

1. **Attribute the compact lead:** compare priors + tail risk, priors + freshness, and their joint representation with the frozen compact-priors control. Keep model settings fixed and expose selected-template substitutions.
2. **Stress its information contract:** repeat additional prior and market delays within the compact representation. Results from a different full representation cannot automatically transfer to it.
3. **Test screened-out mechanisms fairly:** admit a small declared interaction block, then use matched removal or conditional permutation. Avoid a new indiscriminate product expansion.
4. **Explain instability:** inspect the existing horizon, market-coverage, and temporal diagnostics; any resulting group-specific feature hypothesis must be labeled exploratory and compared with an unchanged control.
5. **Resolve external-data prerequisites:** obtain a verifiable calendar and instrument metadata before attempting carry, inventories, positioning, macro vintages, fundamentals, options, or text. A source's current web availability is not a historical as-of dataset.

These are open research questions, not hidden completed tasks. This checkpoint provides broader executed coverage and a stronger candidate while keeping the feature gate open. Final optimization, the untouched test, and submissions remain outside the current phase.
