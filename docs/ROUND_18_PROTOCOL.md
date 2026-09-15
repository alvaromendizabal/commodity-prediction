# Round18 — horizon-spaced sequence structure

## Question
Does the order of previously released outcomes expose persistence or reversal that unordered distribution summaries do not capture? This is distinct from round17's last-K observed event moments and from round12's market-signal/outcome response coefficients.

## Construction
For target horizon h, `known[t] = y[t-h-1]`. Compare known outcomes at t and t-h, making the two constituent h-horizon returns separated by h origins. This avoids directly treating adjacent overlapping h-horizon returns as the intended serial pair. It does NOT make the sequence of rolling pairs independent or remove all calendar/asynchrony issues.

Over 21,63,126 row windows construct eight quantities: finite-pair coverage; mean release-row age; pairwise correlation; latest sign multiplied by estimated agreement; innovation relative to the historical paired regression; current-sign-conditioned transition response; excess positive-transition probability; excess negative-transition probability. Moment estimates use identical finite pairs. Transition estimates additionally require at least three positive and three negative predecessor cases. No unavailable outcomes are zero-filled.

Three windows × eight forms =24 representations. Timing control6; timing+magnitude15; timing+transition15; exact union24. Candidate coverage, missingness, exact duplicates against the active control, and training-half associations are reported, not silently ignored. Fixed clipping bounds are design choices, not optimized values.

## Validation
Original histogram and target/preprocessing settings. Last development fold: training through1523 (warmup252), five-row purge, validation1529–1703. Four new fits maximum; exact original-control replay before fitting; no selection from new round16/17/19 results. The last fold has been inspected in earlier research, so this is exploratory, not holdout confirmation.

All required comparator gains must reach +0.002; numerical panels must retain a numerical feature. The union must beat both halves and the clock control. Dropped/duplicate/constant columns remain visible in the report. Passing earns review on other periods only. Conditional10/20/40-origin paired intervals do not cover the entire adaptive search.

## Resource and recovery
240-second supervised analytical ceiling,14GiB monitored RSS,15-second progress heartbeats. Per-stage manifests; no generic retry after failure. A deliberate pause after a sealed task can resume using the original remaining budget. Forecaster and all old control models remain unchanged.

## Research basis and limits
Hansen and Hodrick (1980), *Forward Exchange Rates as Optimal Predictors of Future Spot Rates: An Econometric Analysis*, addresses k-step forecasting using observations sampled more frequently than the forecast interval: https://www.journals.uchicago.edu/doi/10.1086/260910 . It motivates explicit attention to overlapping horizons; it does not endorse this feature recipe or establish commodity predictability.

The GRU-D study motivates representing observation masks/time intervals in missing time series, based on clinical rather than commodity experiments: https://doi.org/10.1038/s41598-018-24271-9 . We use no GRU model and claim none of its published performance.

Implementation prepared, not executed or tested by the assistant. Final-test values, new external data, future estimates, and model optimization are excluded.
