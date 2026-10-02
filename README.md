# Commodity Forecasting | End-to-End ML Research Case Study

**Alvaro Mendizabal**  [GitHub profile](https://github.com/alvaromendizabal)

A completed machine-learning research program for ranking short-horizon returns across **424 commodity-related targets and four forecast horizons**. The project combines point-in-time data engineering, temporal validation, clean-room reconstruction of strong public modeling ideas, restartable AWS execution, controlled ablations, uncertainty-aware promotion gates, and a final post-competition replay built from the official delayed-label interface.

**Status: complete and employer-facing.** The research record is frozen. Public artifacts emphasize reproducible evidence, engineering decisions, and scientific conclusions; private competition data, fitted checkpoints, prediction matrices, and the latest full training implementation remain withheld.

**Start here:** [Final closeout notebook](notebooks/28_final_research_closeout.ipynb)  [Technical case study](docs/PORTFOLIO_CASE_STUDY.md)  [Final research closeout](docs/FINAL_RESEARCH_CLOSEOUT.md)  [Reproducibility](docs/REPRODUCIBILITY.md)  [Machine-readable summary](reports/portfolio_summary.json)

## Why this project stands out

| Area | What I built |
|---|---|
| **Point-in-time ML** | Horizon-aware target availability, chronological splits, delayed-label controls, leakage checks, and matched-population comparisons across 424 targets |
| **Research breadth** | Controlled studies spanning tree ensembles, RNN/MLP systems, recursive forecasting, feature-token Transformers, online adaptation, multi-output learning, graph representations, market-phase features, covariance/rank systems, lag-group experts, and state-space models |
| **Cloud ML engineering** | Restartable AWS/SageMaker execution with hashes, checkpoints, resumable state, resource telemetry, cost telemetry, process-conflict protection, and deterministic return bundles |
| **Scientific discipline** | Explicit promotion/kill gates, paired uncertainty intervals, zero-fit reconciliation, negative-result preservation, and increasingly strict temporal tests when earlier validation proved optimistic |
| **Evaluation engineering** | Reconstructed the official delayed-label information flow and built a post-competition causal replay over the final 134 test dates, including a separately reported 73-day terminal period |
| **Communication** | Executed notebooks, Plotly diagnostics, machine-readable summaries, experiment ledgers, and a public/private publication boundary designed for technical review |

## Research scale

The public closeout records a substantial experimental program rather than a single notebook submission:

- **424 targets** across **4 horizons**;
- **535-date** development reference plus **355-date** temporal replication and a **247-date** locked historical assessment;
- large controlled sweeps including **2,544 group-wise fits**, **206 online-refit fits**, and **24 Transformer fits** in major study branches;
- a later frontier program with **12 bounded top-level milestones** after the prior public update;
- multiple independently reconstructed public-method families evaluated under the same point-in-time discipline;
- an audited project minimum of **19 successful intended top-level executions** across the full research history.

The point of this breadth was not model-count accumulation. Each major branch had an explicit hypothesis, bounded compute, a kill condition, and a recorded decision.

## Verified evidence

Different rows below use different historical populations and should not be numerically ranked against one another.

| Evaluation population | Recorded result | Research decision |
|---|---|---|
| Development reference: 535 dates, 424 targets | Established `current_market` system: **0.309709** | Historical development reference |
| Later temporal replication: 355 dates, 424 targets | Incumbent **0.276025**; frozen panel **0.289365**; delta **+0.013340** | Positive estimate, but conditional paired 95% interval crossed zero |
| Locked historical assessment: 247 dates, 424 targets | Frozen incumbent **0.190453**; training-only mean **0.212085** | No demonstrated advantage over the simple control; result frozen without reselection |
| Competition-sized replay development | A regularized covariance correction improved the retained panel on three consecutive 73-day windows | Promoted as a useful stability component for subsequent research |
| Final post-competition replay | Training-only model lock followed by causal delayed-label replay on all **134** final test dates, with a separate **73-day** terminal segment | Exposed material regime shift and closed the research cycle without post-period retuning |

The metric throughout is **mean daily cross-sectional rank correlation divided by its population standard deviation**, without annualization. This repository does not present local or post-competition replay results as an official competition placement.

## What I learned

The strongest lesson was not that a particular architecture always wins. It was that **validation quality dominated model complexity**.

Several sophisticated systems produced compelling gains on one historical regime and failed when moved forward in time. Rather than keep optimizing those gains, the project progressively tightened the test design: matched populations, later chronological replication, locked assessments, competition-sized 73-day windows, and finally an exact causal replay using the official delayed-label structure.

That process turned model failure into useful evidence. The final research record demonstrates the ability to build an ML system that can **reject its own optimistic conclusions** and preserve the evidence needed to explain why.

## Reproducibility

The employer-facing release is reproducible at the evidence layer without publishing restricted data or the latest private training recipe.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/verify_portfolio.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the data boundary, notebook checks, and what is intentionally not distributed.

## Publication boundary

This repository is a curated technical case study, not a release of the complete private training system and not a dump of the private AWS workspace. Public artifacts include the research narrative, aggregate results, reproducibility checks, presentation notebooks, and selected historical source already in the repository. Private data, credentials, fitted weights, prediction matrices, operational logs, and the newest complete training orchestration remain private.

That boundary keeps the work reviewable and technically credible without distributing restricted competition data or every implementation detail.
