# Commodity Forecasting | ML Engineering Case Study

**Alvaro Mendizabal** · [GitHub profile](https://github.com/alvaromendizabal)

End-to-end research on return ranking across **424 commodity-related targets and four forecast horizons**. The work combines point-in-time data engineering, domain-informed representations, controlled model comparisons, restartable AWS execution, and a locked historical assessment.

**Project status: complete and closed.** The research cycle, final assessment, and employer-facing publication are complete. No further experiments, submissions, or model tuning are planned for this cycle.

**Start here:** [Portfolio notebook](notebooks/27_portfolio_case_study.ipynb) · [Technical case study](docs/PORTFOLIO_CASE_STUDY.md) · [Verified aggregate results](reports/portfolio_summary.json)

## Engineering highlights

| Area | What I built and evaluated |
|---|---|
| Information timing | Data contracts, horizon-aware label availability, chronological partitions, and a reserved assessment boundary |
| Modeling and validation | Domain-informed pooled nonlinear models and a separately evaluated top-solution-inspired stack, compared on matching dates and targets |
| Research diagnosis | A zero-fit saved-prediction reconciliation that isolated why an apparent model expansion lost its screening gain |
| Reliable execution | Bounded workers, restartable checkpoints, integrity receipts, and exact replay of 175 × 424 development predictions before final access |
| Analytical communication | Notebook-based comparisons that distinguish exploratory gains, temporal replication, uncertainty, and the final decision |

The [case study](docs/PORTFOLIO_CASE_STUDY.md) follows the most informative investigation: the shared-target predictions were identical, but the expanded experiment changed target coverage. Diagnosing that distinction avoided unnecessary retraining and led to a controlled later-period comparison.

## Recorded results

Each comparison below uses the same dates and targets for its two systems. Different rows use different populations and are not directly comparable.

| Evaluation population | Result | Decision |
|---|---|---|
| Development reference: 535 dates, 424 targets | Established `current_market` system: **0.309709** | Historical development reference |
| Later-period replication: 355 dates, 424 targets | Incumbent **0.276025**; frozen panel **0.289365**; difference **+0.013340** | Conditional 95% interval **[-0.009061, +0.035750]**; panel not promoted |
| Locked historical assessment: 247 dates, 424 targets | Frozen incumbent **0.190453**; training-only mean **0.212085**; difference **-0.021632** | Conditional 95% interval **[-0.133651, +0.080336]**; no demonstrated advantage over the control |

![Locked historical assessment](reports/figures/portfolio_final.svg)

The metric is mean daily cross-sectional rank correlation divided by its population standard deviation, without annualization. These are local historical evaluation scores, not Kaggle leaderboard results or trading returns. The final comparison was recorded without post-assessment model reselection. No competition win or production deployment is claimed.

## Review the portfolio

The [portfolio notebook](notebooks/27_portfolio_case_study.ipynb) is a read-only presentation export with two inline Plotly figures and static SVG fallbacks. It presents the problem, engineering decisions, comparisons, and conclusions without distributing executable training source. It is not a newly executed experiment.

The **market-path** study supplies the 0.309709 development reference. Earlier source, notebooks, and configuration files remain historical records; their original "research open" or "final set reserved" descriptions are not the current project status. The [aggregate summary](reports/portfolio_summary.json) records the completed assessment and closeout.

## Public portfolio, private research

This is an employer-facing presentation, not a release of the latest training system or a replication kit. The latest complete implementation, detailed training configurations, raw data, private prediction matrices, weights, environments, and AWS operational records are withheld. The original AWS research artifacts remain separate from this public presentation.

Earlier public source and history remain public; this closeout does not change visibility, rewrite history, or replace existing licenses and attribution. Leading-solution mechanisms were selectively adapted, not completely reproduced. See [publication scope](docs/PUBLICATION_SCOPE.md) for the detailed boundary.

**Closeout:** the historical assessment is final for this research cycle. The superseded publication plans are not pending work, and older bulk-publisher handoffs should not be rerun. The repository remains available as a completed technical portfolio.
