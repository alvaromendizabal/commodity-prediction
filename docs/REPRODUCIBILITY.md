# Reproducibility guide

The public release supports three distinct tasks: checking published aggregates, exercising a synthetic timing contract, and running the public source tests. Reproducing the private trained system requires artifacts that are intentionally withheld.

## Quick checks: no installation or cloud account

Use Python 3.12 from the repository root:

```bash
python3 scripts/verify_portfolio.py
python3 examples/delayed_label_demo.py
```

Both commands use the standard library, run on a CPU, and make no network requests. Expected success markers are `PORTFOLIO_VERIFY=PASS` and the synthetic example's `future_access` value of `rejected`. The [example documentation](../examples/README.md) describes its fictional inputs and timing convention.

The portfolio verifier checks:

- the frozen 424-target / four-horizon contract;
- finite aggregate metrics and arithmetic for the two matched comparisons;
- recorded uncertainty and promotion decisions;
- 134 replay dates, a 73-day terminal segment, and training-only selection;
- the expected research families and publication boundary;
- saved closeout notebook execution counts and absence of error outputs.

A pass establishes **internal consistency of the committed public record**. It does not reconstruct scores from private predictions, verify withheld AWS receipts, or establish official competition performance.

## Locked environment and source checks

The repository targets Python 3.12. Install `uv`, then use the committed `uv.lock`:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python scripts/quality.py
uv run --frozen python scripts/verify_portfolio.py
```

`pyproject.toml` declares compatible dependency ranges; `uv.lock` fixes the exact resolved dependency set. The quality command runs lint, formatting checks, typing, and public tests. CI additionally checks the historical source/result lineages and the public portfolio evidence.

## Notebook review and execution

The existing notebooks retain saved outputs. The [final closeout notebook](../notebooks/28_final_research_closeout.ipynb) reads only:

- `reports/portfolio_summary.json`;
- `reports/final_research_ledger.json`.

Its five code cells are aggregate analyses and do not fit models. This command executes a copy and verifies that its outputs persist after save/reopen:

```bash
uv run --frozen python scripts/execute_public_closeout.py
```

A local Jupyter kernel requires local socket access; a restricted execution sandbox may block its startup. The Quality workflow executes this single aggregate notebook on GitHub Actions and saves a downloadable notebook artifact. The generated local copy is ignored by Git. This verifies computational execution and saved output data; it does not claim a browser visual-render check. Earlier notebooks have their own historical data and checkpoint requirements: [manual reproduction](MANUAL_REPRODUCTION.md). Reviewing their saved outputs does not require launching the studies again.

## Evidence and privacy boundary

| Publicly repeatable | Requires private artifacts |
|---|---|
| Aggregate arithmetic, notebook summary tables, and ledger counts | Recomputing rank scores and paired intervals from full predictions |
| Synthetic delayed-label boundary and failure tests | Replaying the exact final competition stream |
| Tests for selected published historical source | Refitting the newest complete private model system |

Raw competition data, target/prediction matrices, fitted weights, private checkpoints, credentials, operational logs, and the newest orchestration package are excluded. The aggregate summaries retain historical evidence IDs as references, not as public raw receipts. See [publication scope](PUBLICATION_SCOPE.md).
