# Frontier reproducibility guide

## Reproducibility contract

This repository publishes the source, tests, protocols, notebooks, and aggregate evidence needed to inspect the research design. The integrated competitive reconstruction is directly runnable from the normal project tree. Newer frontier handoff packages remain in the private AWS execution workspace, but their exact SHA-256 identities and scientific protocols are recorded publicly so private execution receipts can be reconciled to the reviewed research state.

This boundary avoids redistributing competition data, private prediction matrices, fitted weights/checkpoints, credentials, or operational secrets.

## Environment

The repository targets Python 3.12 and uses the locked `uv` environment for the core research system. Frontier GPU experiments additionally require the model libraries declared by their reviewed package source. The recorded Transformer run used PyTorch 2.8.0 with CUDA 12.9 on an NVIDIA L40S.

## Data prerequisite

Exact numeric reproduction requires the authorized MITSUI competition files. Obtain them from the official competition source under its own rules and license. Do not silently substitute another target ordering, date boundary, or label-release interpretation.

## Reproduction rules

1. Run `uv sync --frozen --extra dev`.
2. Run `uv run --frozen --extra dev python scripts/quality.py`.
3. Run the publication verifiers, including `scripts/verify_frontier_publication.py`.
4. Review `reports/frontier_research_ledger.json` before selecting a research family.
5. Verify competition-file hashes before fitting.
6. Respect each target's horizon-specific label-release delay.
7. Frontier tuning must stop before date_id 1704.
8. The historical assessment at date_id 1714-1960 is already evaluated and must not be recycled as a tuning population.
9. Reuse checkpoints only when study/package identity and declared file hashes match.
10. Where fold 0 selects a candidate or blend, freeze it before folds 1 and 2.
11. Report fold-level and pooled evidence plus paired uncertainty.
12. Do not describe a local development score as an official Kaggle leaderboard score.

## Public source map

The third-place-inspired reconstruction is integrated under:

- `src/commodity_prediction/competitive/`
- `configs/competitive_feature_forecast.json`
- `configs/third_place_reproduction.json`
- `scripts/run_competitive_feature_forecast.py`
- `scripts/run_third_place_reproduction.py`
- `tests/test_competitive_feature_forecast.py`
- `tests/test_third_place_reproduction.py`
- `notebooks/20_competitive_feature_forecast.ipynb`
- `notebooks/21_third_place_reproduction.ipynb`

The newer frontier handoffs are reconciled by exact package hashes in [`research/frontier/package_hashes.json`](../research/frontier/package_hashes.json), while their measured aggregate outcomes are recorded in [`reports/frontier_research_ledger.json`](../reports/frontier_research_ledger.json).

## Evidence notebook

[`notebooks/28_frontier_research_ledger.ipynb`](../notebooks/28_frontier_research_ledger.ipynb) is an executed, no-training presentation notebook. It renders matched later-period scores, chronological transfer deltas, and paired uncertainty without including restricted rows.
