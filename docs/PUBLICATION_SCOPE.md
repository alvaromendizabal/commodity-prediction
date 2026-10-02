# Portfolio publication scope

## Current purpose

This repository presents Alvaro Mendizabal's completed commodity-forecasting research as an employer-facing ML engineering case study.

The research cycle is closed. GitHub is the curated public evidence layer; the canonical private AWS workspace retains detailed execution artifacts that are inappropriate to publish wholesale.

## Included in this release

The public portfolio includes:

- the final README and technical case study;
- a final closeout notebook with executable aggregate-analysis cells;
- aggregate evaluation results and a machine-readable experiment ledger;
- a reproducibility guide and portfolio-verification script;
- selected historical source, notebooks, and figures already published in the repository;
- a final research closeout describing the validation evolution, modeling breadth, engineering decisions, and limitations.

## Reproducibility boundary

The public release is **evidence-reproducible** rather than a distribution of the complete private training system.

A reviewer can recreate the public Python environment, run the repository tests, validate the aggregate summaries, and execute the closeout notebook without access to private prediction matrices or fitted checkpoints.

The newest complete orchestration, private data-derived matrices, fitted model artifacts, raw competition data, AWS checkpoints, credentials, and operational logs are intentionally excluded.

## Existing public material

Older source, notebooks, configuration files, branches, and commit history were already public before this closeout. They remain public. This release does not rewrite Git history, delete historical research context, or replace existing licenses and attribution.

When older files describe an earlier project state, the current status is defined by:

- `README.md`;
- `docs/FINAL_RESEARCH_CLOSEOUT.md`;
- `docs/REPRODUCIBILITY.md`;
- `reports/portfolio_summary.json`;
- `reports/final_research_ledger.json`.

## Attribution and claims

Publicly documented methods informed selected experiments and remain credited to their original authors. Where exact source settings were unavailable, the project records a clean-room reconstruction or adaptation rather than claiming code parity.

The repository presents recorded local, historical, and post-competition replay evidence as such. It does not claim live deployment, trading profitability, or an official competition placement for the final private research state.
