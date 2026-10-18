# Read forecast errors at the level where decisions fail

Forecast Lab now reports the same evaluated predictions at three levels: the complete panel, each fictional series, and two chronological halves. Change the candidate in the existing comparison table to inspect its error and signed bias beside the released-label mean. Nothing is refitted or selected by this view.

| Quantity | Definition | Why inspect it |
|---|---|---|
| RMSE | Square root of the mean squared forecast error | Highlights large misses on a common target population |
| Mean absolute error | Mean absolute forecast error | Describes typical absolute error without squaring it |
| Bias | Mean prediction minus mean realized target | Separates persistent overprediction from underprediction |
| Count | Number of series/origin forecasts in the slice | Makes differing denominators explicit |

Every slice is computed from individual rows. Pooled RMSE equals the square root of the count-weighted mean of slice MSEs; averaging slice RMSEs is generally incorrect. The tests verify this identity for both partitions and verify bias independently on a hand-computed example.

Chronological halves are defined from sorted evaluation origins, with one extra origin in the first half when the count is odd. A one-origin evaluation has an empty second half, whose scores are `null`, not zero or a fabricated perfect score. Duplicate identities, nonfinite errors and missing numeric predictions fail explicitly.

## Inspect a scenario

1. Open `public-demo/index.html`, or serve the checkout with `python -m http.server 8000` and visit `/public-demo/`.
2. Choose the regime-shift scenario and run the evaluation.
3. Compare errors across the six series and the two time periods. Select another candidate to update the diagnostics on the same rows.
4. Export the result and run the [replay command](DEMO_REPLAY.md). The derived slices are included in the export and recomputed during replay.

```bash
node tools/test_public_demo.mjs
node tools/test_replay_public_demo.mjs
```

The six series are fictional, and the errors use arbitrary signal units. Overlapping forecast targets are dependent. These are descriptive diagnostics, with no confidence-interval, market-profitability or model-selection claim. They explain how to inspect a forecasting system; the [historical research findings](FINAL_RESEARCH_CLOSEOUT.md) retain their original evaluation scopes.

[Computation](../public-demo/engine.js) · [Actual UI](../public-demo/app.js) · [Regression tests](../tools/test_public_demo.mjs)
