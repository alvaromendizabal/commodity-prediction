# Case study · Forecast only with information available at the time

I built a point-in-time forecasting workflow spanning 424 targets and four horizons. The important engineering problem was not simply fitting a powerful model: delayed labels, mismatched date populations and repeated selection could manufacture apparent improvements.

## Final approach and decision

The final private reference combined target-routed trees with retained covariance and persistence/retrieval components. On a later 355-date comparison, the candidate's positive estimate was not supported by a strictly positive paired interval. On the locked 247-date historical assessment, the reference scored 0.190453 versus 0.212085 for a training-only mean. I recorded that negative result without reopening the assessment for reselection. These are historical correlation-ratio metrics, not trading returns. [Original evidence](../reports/portfolio_summary.json)

## Public engineering that can be exercised

The retained rank metric validates exact date and target ordering, tied ranks, missing labels and nonfinite inputs. The paired uncertainty implementation keeps resampling inside fitted-model boundaries. The release-calendar example rejects future-label reads and proves that changing unavailable labels cannot change an earlier snapshot.

The browser control really refits on eligible synthetic history. Changing the forecast horizon, label delay or training window changes the fitted result, not a prerecorded chart. Exported forecasts are independently recomputed and checked for tampering. [Reproduction](REPRODUCIBILITY.md)

## Scope and ownership

My work covers the data-availability protocol, feature/model evaluation, artifact recovery, comparison diagnostics and public review experience. Private fitted models, exact routing specifications and competition data are not distributed. The public release validates component behavior and the recorded conclusion; it does not claim to reproduce private competitive performance or establish future profitability.
