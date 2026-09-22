# Commodity Forecasting | ML Engineering Case Study

**Alvaro Mendizabal** · [GitHub profile](https://github.com/alvaromendizabal)

An end-to-end investigation of multi-horizon commodity return ranking: point-in-time data engineering, domain-informed representations, controlled model comparisons, restartable AWS execution, and a locked historical assessment.

**Start with the [portfolio notebook](notebooks/27_portfolio_case_study.ipynb).** Read the [technical case study](docs/PORTFOLIO_CASE_STUDY.md) for the engineering decisions and results.

**Status: research cycle and historical assessment completed.** This portfolio presents selected evidence for hiring review. It is not a release of the latest training system, model weights, or a replication kit.

## What this project demonstrates

| Capability | Evidence from the work |
|---|---|
| Prediction-time data engineering | Horizon-aware label availability, chronological partitions, and a reserved assessment boundary |
| Experimental design | Matched target/date comparisons; separate screening, temporal replication, and final assessment |
| ML research | Domain features, pooled nonlinear models, and a separately evaluated top-solution-inspired stack |
| Debugging and efficiency | Saved-prediction reconciliation isolated a target-coverage effect without retraining |
| Reliable execution | Bounded runs, restartable checkpoints, integrity receipts, and preserved analytical outputs |
| Technical judgment | An inconclusive improvement was not promoted; the final control comparison was reported without reselection |

## Results, with the evaluation populations kept separate

| Evaluation | Systems compared on the same population | Result |
|---|---|---|
| Development reference | Established `current_market` system; 535 dates and 424 targets | **0.309709** |
| Later-period replication | Incumbent **0.276025** vs frozen panel **0.289365**; 355 dates and 424 targets | **+0.013340**, conditional 95% interval **[-0.009061, +0.035750]**; not promoted |
| Locked historical assessment | Frozen incumbent **0.190453** vs training-only mean **0.212085**; 247 dates and 424 targets | **-0.021632**, conditional 95% interval **[-0.133651, +0.080336]**; no demonstrated advantage |

The metric is mean daily cross-sectional rank correlation divided by its population standard deviation, without annualization. These are **local historical results, not Kaggle leaderboard scores, trading returns, or a competition-win claim**. Scores from different rows of this table should not be ranked as if they used the same population.

![Locked historical assessment](reports/figures/portfolio_final.svg)

The final comparison did not justify a performance claim. The project demonstrates the ability to build and evaluate a complex forecasting system, investigate apparent gains, and distinguish a useful experiment from a deployable improvement. No production deployment is claimed.

## Read the work, not a setup manual

The [portfolio notebook](notebooks/27_portfolio_case_study.ipynb) is a deliberately **read-only presentation export** with inline visual evidence. It contains no training cells, feature formulas, fitted artifacts, or environment-setup instructions. The [case study](docs/PORTFOLIO_CASE_STUDY.md) explains the problem, ownership, architecture, experiments, and conclusions. The [aggregate summary](reports/portfolio_summary.json) records the published numbers and evidence provenance.

The **market-path** study supplies the 0.309709 development reference. Earlier public source and historical notebooks remain in this repository, but they are archival material and are not a release of the latest privately retained research workflow. Earlier configuration files describe their original study states; the current assessment status is recorded in the portfolio summary.

## Publication boundary

The current release is for employer evaluation. The latest complete implementation, detailed training configurations, raw data, private prediction matrices, model weights, environments, and AWS operational records are withheld. Existing public material is not made private by this statement. This update does not rewrite history, change repository visibility, or revoke previously granted rights. See [publication scope](docs/PUBLICATION_SCOPE.md).

Top-solution mechanisms were studied and selectively adapted; a complete reproduction of all leading systems is **not** claimed. Attribution and original notices remain applicable. The work is presented as a completed research case study, not an open-source support commitment.
