# Commodity Forecasting

**Point-in-time machine learning across 424 targets and four forecast horizons.**
Built by [Alvaro Mendizabal](https://github.com/alvaromendizabal).

[![Quality](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/quality.yml)
[![Portfolio evidence](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/portfolio.yml/badge.svg)](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/portfolio.yml)

This completed research project asks whether commodity return rankings survive a change in market regime. I owned the data contracts, feature and model research, temporal evaluation, resumable AWS execution, and evidence review. The central result: **a promising development improvement did not establish reliable advantage on a stricter historical assessment.** The repository preserves that result and the engineering used to discover it.

**Review in five minutes:** [Technical case study](docs/PORTFOLIO_CASE_STUDY.md) · [Executed closeout notebook](notebooks/28_final_research_closeout.ipynb) · [Reviewer guide](docs/REVIEWER_GUIDE.md)

## Results and decisions

Scores below are **mean daily cross-sectional rank correlation divided by its population standard deviation**, without annualization. Compare models within a row; the rows use different dates.

| Evaluation | Matched results | Decision |
|---|---|---|
| Historical market-path development, 535 dates | Reference **0.309709** | Archived development reference |
| Later chronological replication, 355 dates | Candidate **0.289365** vs. incumbent **0.276025**; delta **+0.013340** | Not promoted: paired 95% interval **[−0.009061, +0.035750]** crosses zero |
| Locked historical assessment, 247 dates | Incumbent **0.190453** vs. training-only mean **0.212085** | No demonstrated advantage; recorded without reselection |
| Post-competition delayed-label replay, 134 dates | Training-only selection; separate 73-day terminal segment | Material regime shift recorded; no final-period retuning |

![Candidate improvement and uncertainty across later temporal periods](reports/figures/portfolio_temporal.svg)

The 355-date replication is retrospective development evidence, not an untouched test. The public replay summary exposes timing and coverage metadata rather than final numeric scores or private predictions. These results do not establish an official competition placement, trading return, or production deployment.

## Engineering decisions worth reviewing

| Problem | Decision and evidence |
|---|---|
| Labels become available at different times | Horizon-aware release contracts and chronological preprocessing; [synthetic delayed-label example](examples/README.md) |
| A larger candidate panel appeared to regress | Reconciled saved predictions with **zero new fits**; all **37 shared targets** matched exactly. Target composition explained the difference. [Case study](docs/PORTFOLIO_CASE_STUDY.md#a-zero-fit-debugging-result-that-changed-the-research-direction) |
| Attractive local gains failed later | Froze candidate definitions, used matched populations and paired uncertainty, then increased temporal rigor. [Assessment evidence](notebooks/27_portfolio_case_study.ipynb) |
| Interrupted cloud work could waste completed training | Checkpoints, artifact hashes, process-conflict guards, and success/failure return bundles preserved expensive work. [Closeout](docs/FINAL_RESEARCH_CLOSEOUT.md#engineering-accomplishments) |

The [research ledger](reports/final_research_ledger.json) records **20 families**, including tree ensembles, neural sequence models, Transformers, graph features, covariance methods, and online adaptation. Major branches include **2,544 group-wise fits**, **206 online-refit fits**, and **24 Transformer fits**. These are study-specific counts, not independent production models. [Public method references](docs/competitive-research.md) distinguish author claims from this project's reconstructions.

## Run the public examples

From the repository root, Python 3.12 and a CPU are sufficient; neither command downloads data or models:

```bash
python3 scripts/verify_portfolio.py
python3 examples/delayed_label_demo.py
```

The first checks the published aggregate record and saved closeout notebook. The second demonstrates release-time boundaries and rejects future-label access on authored synthetic data. Neither recreates the private forecasting scores.

For the exact committed dependency set and full public test suite:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python scripts/quality.py
```

[Reproducibility guide](docs/REPRODUCIBILITY.md) covers notebook execution, evidence checks, and historical research dependencies. The existing executed notebooks remain available for review without rerunning cloud studies.

## Scope and limitations

The research cycle is **complete and frozen**. A new forecasting claim needs a new evaluation contract and unseen data. The later short-window covariance result is a recorded research decision; the public aggregate verifier does not independently reproduce its private prediction-level statistics.

This is **not a release** of the complete private training system. Public material includes selected historical source, tests, aggregate findings, notebooks, and synthetic examples. Restricted competition data, fitted weights, prediction matrices, credentials, cloud logs, and the latest private training orchestration remain excluded. Previously public source and attribution remain intact. See [publication scope](docs/PUBLICATION_SCOPE.md).
