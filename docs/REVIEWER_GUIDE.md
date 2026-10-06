# Reviewer guide

This is a completed forecasting research case study by Alvaro Mendizabal. It demonstrates point-in-time evaluation, controlled model comparisons, cloud experiment recovery, and the judgment to retain a negative result.

## Five-minute review

1. Read the [README result table](../README.md#results-and-decisions). Each comparison has an explicit evaluation population.
2. Open the [case study](PORTFOLIO_CASE_STUDY.md), especially the zero-fit reconciliation and the locked historical assessment. These connect an engineering decision to a measured outcome.
3. Open the [executed closeout notebook](../notebooks/28_final_research_closeout.ipynb). Its cells read two public JSON summaries; it is an aggregate analysis, not a model-training run.
4. Run the standard-library checks in the [reproducibility guide](REPRODUCIBILITY.md). They need no cloud account, competition data, GPU, or credentials.

## Technical review

| Question | Evidence |
|---|---|
| How is future information excluded? | [Data audit notebook](../notebooks/00_data_audit.ipynb), [released-context research](released-context-research.md), and [synthetic timing example](../examples/README.md) |
| What did exploratory analysis reveal? | [Executed EDA](../notebooks/01_eda.ipynb) and [feature research](../notebooks/02_feature_research.ipynb) |
| How were contradictory experiments reconciled? | [Saved-prediction reconciliation](PORTFOLIO_CASE_STUDY.md#a-zero-fit-debugging-result-that-changed-the-research-direction): identical shared predictions, different target composition |
| Why reject an apparently improved candidate? | [Portfolio notebook](../notebooks/27_portfolio_case_study.ipynb): conditional paired interval crossed zero |
| Which methods were tested? | [Twenty-family ledger](../reports/final_research_ledger.json) and [public-source audit](competitive-research.md) |
| How is correctness checked? | [Quality workflow](../.github/workflows/quality.yml), [portfolio workflow](../.github/workflows/portfolio.yml), and [test suite](../tests) |
| What is deliberately withheld? | [Publication boundary](PUBLICATION_SCOPE.md) |

## Interpretation limits

- The 535-date development score is not directly comparable to an organizer leaderboard score.
- The later 355-date replication reused historical development periods. Its uncertainty is conditional on the selected candidate, not a correction for all exploratory choices.
- The locked 247-date assessment did not demonstrate improvement over a training-only mean control.
- The final 134-date replay is post-competition research. Public metadata verifies its recorded scope; it does not supply private predictions or a reproducible official placement.
- The aggregate verifier establishes internal consistency of committed evidence. It cannot authenticate withheld raw receipts or independently regenerate forecasting performance.

## Ownership and discussion

I owned the research loop, data contracts, implementation, AWS execution and recovery, evaluation, and communication. Publicly documented methods are credited separately from my reconstructions and adaptations. The repository does not claim live deployment or business revenue.

Useful interview topics are release timing, matched-population debugging, uncertainty-based promotion, expensive-work recovery, and why a simpler control won on a stricter evaluation. Historical documents remain available as dated research records; the current project status is defined in the [final closeout](FINAL_RESEARCH_CLOSEOUT.md).
