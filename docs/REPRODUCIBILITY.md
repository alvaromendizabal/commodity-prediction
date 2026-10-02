# Reproducibility guide

This closeout release is designed to be reproducible at the **public evidence layer** while respecting the boundary around restricted data and private research artifacts.

## 1. Environment

Requirements:

- Python 3.12
- Git
- a standard CPU environment is sufficient for the public closeout checks

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The pinned dependency ranges live in `pyproject.toml`.

## 2. Verify the public research record

Run:

```bash
python scripts/verify_portfolio.py
```

The verifier checks that:

- the project is marked complete;
- the 424-target / four-horizon contract is preserved;
- published deltas equal the underlying published scores;
- the final replay metadata has 134 scored dates and a 73-date terminal segment;
- the final experiment ledger contains the expected research families;
- the final closeout notebook is valid notebook JSON and references the public summary artifacts.

The verifier uses only Python's standard library.

## 3. Execute the final closeout notebook

Launch Jupyter and open:

`notebooks/28_final_research_closeout.ipynb`

The notebook reads only the two committed aggregate JSON files:

- `reports/portfolio_summary.json`
- `reports/final_research_ledger.json`

It does not require competition data, model weights, or private predictions.

## 4. Public tests

When working on source code rather than the presentation layer, run:

```bash
pytest
ruff check .
mypy
```

The project configuration for those tools is in `pyproject.toml`.

## 5. What is intentionally not reproducible from GitHub alone

The repository does not include:

- raw competition data;
- private target/prediction matrices;
- current fitted model weights;
- private AWS checkpoints;
- credentials or tokens;
- operational logs from the private SageMaker workspace;
- the newest complete private orchestration package.

Those omissions are deliberate. They prevent redistribution of restricted data and keep the repository focused on reviewable engineering evidence rather than serving as a turnkey competition replication kit.

## 6. Evidence provenance

Public metrics are derived from recorded execution receipts and aggregate closeout artifacts. Major historical evidence IDs are retained in `reports/portfolio_summary.json`. The final research ledger records the later model-family decisions without distributing underlying private predictions.

The public record is therefore reproducible in two layers:

1. **code-level reproducibility** for the published package and tests;
2. **evidence-level reproducibility** for the closeout narrative, tables, and figures.
