# Forecast only with information that has arrived.

**Alvaro Mendizabal · Time-series ML / point-in-time data / experiment design**

A commodity-forecasting research system built around delayed-label availability, chronological evaluation, controlled feature studies, and recoverable execution. The strongest lesson is knowing when a promising development result does not transfer.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Data science | **Treat publication time as part of the feature contract.** Reject labels whose release time lies after the forecast origin instead of relying only on observation dates. | [Code](../examples/delayed_label_demo.py) · [Tests](../tests/test_delayed_label_demo.py) · [Evidence](../docs/PORTFOLIO_CASE_STUDY.md) |
| ML engineering | **Preserve checkpoint identity across interruptions.** Bind persistent research state to input and configuration identities rather than assuming a file is reusable because it exists. | [Code](../src/commodity_prediction/studies/storage.py) · [Tests](../tests/test_study_storage.py) · [Evidence](../reports/execution_recovery.json) |
| Data science | **Compare models on the same target population.** Use target-composition and prediction-parity checks before interpreting an apparent metric regression. | [Code](../src/commodity_prediction/studies/evaluation.py) · [Tests](../tests/test_feature_study.py) · [Evidence](../reports/portfolio_summary.json) |
| Data science | **Publish a defensible negative conclusion.** Retain the locked assessment where the incumbent did not demonstrate an advantage over a training-only mean. | [Code](../scripts/verify_portfolio.py) · [Tests](../tests/test_portfolio_evidence.py) · [Evidence](../reports/final_research_ledger.json) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `forecast-evidence` | python | [`scripts/verify_portfolio.py`](../scripts/verify_portfolio.py) |
| `label-timing` | python | [`examples/delayed_label_demo.py`](../examples/delayed_label_demo.py) |
| `forecast-lab` | node | [`tools/test_public_demo.mjs`](../tools/test_public_demo.mjs) |

## What is actually reproduced

The public commands recompute aggregate evidence and execute a synthetic delayed-label access contract. Optional browser tests exercise an online ridge demonstration using released labels only. These checks do not establish trading profitability.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint covers this curated review surface, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. HTML links follow the current default branch; the receipt is the record of the local bytes inspected. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Restricted market data, private prediction matrices, fitted research weights, current feature recipes, full orchestration, and operational cloud state remain excluded. Historical source credits and results are retained.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.

## Source-bound replay

The optional Node path also exercises the [export replay contract](../docs/DEMO_REPLAY.md).
Review receipts now fingerprint the complete public demo and tool source inventory,
including added and removed files. They bind command arguments, captured logs and
the rendered brief. A clean checkout records its exact commit with immutable source
links; uncommitted work is identified explicitly. These are consistency checks, not
an independent authentication of historical research scores.
