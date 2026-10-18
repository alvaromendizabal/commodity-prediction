# Forecast only with information that has arrived.

**Alvaro Mendizabal · Time-series ML / point-in-time data / experiment design**

A commodity-forecasting research system built around delayed-label availability, chronological evaluation, controlled feature studies, and recoverable execution. The strongest lesson is knowing when a promising development result does not transfer.

[Project overview](../README.md) · [Printable technical brief](index.html) · [Review configuration](project.json)

## Follow an engineering decision to its evidence

| Focus | Engineering decision | Inspect |
|---|---|---|
| Data science | **Treat publication time as part of the feature contract.** Reject labels whose release time lies after the forecast origin instead of relying only on observation dates. | [Implementation](../examples/delayed_label_demo.py) · [Tests](../tests/test_delayed_label_demo.py) · [Evidence](../docs/PORTFOLIO_CASE_STUDY.md) |
| ML engineering | **Preserve checkpoint identity across interruptions.** Bind persistent research state to input and configuration identities rather than assuming a file is reusable because it exists. | [Implementation](../src/commodity_prediction/studies/storage.py) · [Tests](../tests/test_study_storage.py) · [Evidence](../reports/execution_recovery.json) |
| Data science | **Inspect where pooled accuracy breaks down.** Recompute errors by series and chronological half, retaining sample counts, bias and matched baseline errors. | [Implementation](../public-demo/engine.js) · [Tests](../tools/test_public_demo.mjs) · [Evidence](../docs/ERROR_ANALYSIS.md) |
| Data science | **Compare models on the same target population.** Use target-composition and prediction-parity checks before interpreting an apparent metric regression. | [Implementation](../src/commodity_prediction/studies/evaluation.py) · [Tests](../tests/test_feature_study.py) · [Evidence](../reports/portfolio_summary.json) |
| Data science | **Publish a defensible negative conclusion.** Retain the locked assessment where the incumbent did not demonstrate an advantage over a training-only mean. | [Implementation](../scripts/verify_portfolio.py) · [Tests](../tests/test_portfolio_evidence.py) · [Evidence](../reports/final_research_ledger.json) |
| ML engineering | **Turn an exported result into a replayable artifact.** Regenerate the synthetic environment, recompute the output with the actual public engine, and reject stale source or altered evidence. | [Implementation](../tools/replay_public_demo.mjs) · [Tests](../tools/test_replay_public_demo.mjs) · [Evidence](../docs/DEMO_REPLAY.md) |

## One command, an inspectable result

From a checkout with Python 3.11 or newer and Node.js 24.19.0:

```bash
python3 -S review/run.py --output /tmp/ml-review --with-node
python3 -S review/run.py --output /tmp/ml-review --with-node --check
```

Open `/tmp/ml-review/index.html` for the rendered brief. Logs, `receipt.json` and `manifest.json` record the actual checks, source identities and output hashes. On Windows, choose a writable directory outside the checkout. Repeating the command reuses a verified result; changed sources or damaged evidence fail explicitly.

Omit `--with-node` to run only the Python checks; omitted browser checks are never reported as passed. CI runs both runtimes on Python 3.11 and 3.13.

## Verification scope

The public commands recompute aggregate evidence and execute a synthetic delayed-label access contract. Optional browser tests exercise an online ridge demonstration using released labels only. These checks do not establish trading profitability.

Restricted market data, private prediction matrices, fitted research weights, current feature recipes, full orchestration, and operational cloud state remain excluded. Historical source credits and results are retained.

[Case study](../docs/PORTFOLIO_CASE_STUDY.md) · [Reproduction guide](../docs/REPRODUCIBILITY.md) · [Publication scope](../docs/PUBLICATION_SCOPE.md) · [Repository map](../docs/REPOSITORY_GUIDE.md)
