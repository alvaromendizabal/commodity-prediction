# Conditional risk-state feature research

**Experiments complete; publication and feature gates remain open.** This checkpoint adds 24 executed fits and 34 candidate templates. The best new development score is **0.309186**, only **0.000098** above the previous 0.309088 lead. That difference is too small and uncertain to claim a substantive advance. No final model is promoted.

## Hypothesis and information timing

[Moreira and Muir (2017), Volatility-Managed Portfolios](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513) report that expected returns do not necessarily move proportionally with volatility in their studied portfolios. We infer a testable representation hypothesis: historical priors may have different relevance as risk changes. Their results do not establish these formulas, short-horizon predictability, or trading profitability in this dataset.

The study uses six bounded states: 63/252 and 126/252 historical-risk ratios, five- and 21-date log-risk changes, 21-date log-risk instability, and a trailing 126-date risk percentile. Four bounded drivers normalize 63-date location, 126-date location, the shrunk market-pair prior, and short-versus-long location change by historical risk. Twenty-four products combine every driver with every state; product variants retain their main effects.

Every input descends from the frozen horizon+1-released historical priors. Rolling windows end at the prediction origin. No global ranks, fitted full-period regimes, calendar inference, forward filling of macro vintages, or unreleased outcomes enter these features. The delay stress shifts released inputs another date before deriving their states and products. Tests verify prefix equivalence, future-perturbation invariance, aligned delays, bounded products, invalid-risk missingness, exact replay, no-refit reuse, and tamper rejection.

## Matched results

All settings remain fixed: the same pooled histogram estimator, target normalization, training-only preprocessing, purges, and 535 development validation dates. Admission retains every training-usable nonduplicate column. The declared screened variant retains 64 of 103 columns, rejecting one correlated column and 38 by the feature budget in each fold. Counts below are pooled columns, not independent signals.

| Representation | Candidates / retained / rejected | Official metric | Fold 1 | Fold 2 | Fold 3 |
|---|---|---:|---:|---:|---:|
| admitted_tail | Frozen control | 0.305336 | 0.390579 | 0.152341 | 0.399881 |
| screened_tail | Frozen control | 0.309088 | 0.405901 | 0.154856 | 0.400172 |
| tail_states | 75 / 75 / 0 | 0.303902 | 0.386647 | 0.161222 | 0.391105 |
| tail_scaled_priors | 73 / 73 / 0 | 0.303482 | 0.393033 | 0.172189 | 0.372108 |
| tail_states_and_priors | 79 / 79 / 0 | 0.309186 | 0.386784 | 0.178534 | 0.383185 |
| tail_state_products | 103 / 103 / 0 | 0.299257 | 0.381015 | 0.164947 | 0.373113 |
| joint_state_products | 115 / 115 / 0 | 0.287788 | 0.334708 | 0.169897 | 0.371803 |
| base_state_products | 73 / 73 / 0 | 0.291457 | 0.357961 | 0.176080 | 0.345744 |
| screened_state_products | 103 / 64 / 39 | 0.278565 | 0.340575 | 0.136882 | 0.372019 |
| delayed_state_products | 103 / 103 / 0 | 0.286221 | 0.378322 | 0.121784 | 0.381642 |

The combined state/prior main effects score **0.309186**, versus **0.305336** for their matched admitted-tail control: **+0.003851**, conditional 20-date-block 95% interval **[-0.014864, +0.021876]**. The middle fold rises from **0.152341 to 0.178534**, but still trails historical means (**0.182488**). The first and third fold point estimates fall. This is mixed temporal evidence.

Adding the 24 products changes the metric by **-0.009929** against the same main effects, interval **[-0.039094, +0.022539]**. Adding freshness to the product representation, applying the 64-column screen, and imposing an additional information delay also lower their matched point estimates. States or normalized priors alone do not outperform admitted tail features. These are fitted negative results under this fixed representation, not universal claims that the mechanisms never help.

The comparison family now contains **255 declared contrasts across five domain phases**, with 2,000 paired within-fold block resamples at 10, 20, and 40 dates. **Zero positive simultaneous lower bounds** occur at every block setting. These bounds condition on fitted models and do not undo the adaptive research history. The historical-mean control remains 0.217739; the new lead's +0.091447 difference is an exploratory point estimate, not an independently established feature effect.

## Exact inventory and execution evidence

- **34 new templates:** 6 states + 4 normalized priors + 24 products; 14,416 target-template assignments across 424 outputs.
- Research-wide domain union: **524 templates**, following the earlier 490. The largest new fitted panel has 115 columns. The original 29,917 intermediate source series are reused; these additions are transformations of preserved target-level priors.
- Best new panel: **79 candidates, 79 retained, zero screened rejections per fold**. All 24 product columns are admitted in their designated ablations; their weak result cannot be attributed to exclusion by screening.
- **24 fitted models, 24 exact saved-model replays, 25 sealed local stages**, maximum prediction difference **0.0**. Complete verified resume takes **0.038 seconds** inside the verification stage and performs no data load or refit. The resumed experiment process completed in **271.5 seconds**; its first fit was reused after a blocked upload.
- **108 tests passed** locally, plus Ruff, formatting, type checks, source/result validation, and aggregate metric/count validation. The earlier three executed notebooks and 28 Plotly/static pairs still verify.
- The new notebook chapter and strict verification command are prepared. **The new chapter has not been executed or rendered.** Draft CI verifies new aggregates separately and preserves the previous notebook checks; strict risk-state notebook publication remains required before release.

Machine-readable evidence: [study](../reports/risk_state_study.json), [lineage](../reports/risk_state_lineage.json), [execution and blockers](../reports/risk_state_execution.json).

## Publication blockers and completion assessment

Automatic approval review rejected uploading the new private model/prediction checkpoint archives to the existing project bucket, citing missing explicit payload-and-destination authorization. No alternative upload route was attempted. The new checkpoints remain local; **zero new stage uploads were found in S3**. Existing S3 snapshots remain available. The project SageMaker app is **Deleted**, and this session started no AWS compute.

Local Chrome rendering cannot create required sockets, and the session does not permit sandbox escalation. The previously executed notebooks remain intact. New notebook sources compile but are not presented as executed outputs. The branch is a draft research checkpoint, not a finished release.

A subjective delivery rubric places current readiness at **67/100, approximately two-thirds complete**, not an estimate of time remaining: data/timing contract 15/15; plausible feature coverage 20/25; attribution and stability evidence 12/20; execution/lineage/durable checkpoints 11/15; current notebooks and documentation 9/15; final evaluation/release 0/10. The weights and judgments are explicit rather than an unsupported 80% claim. A 9.9/10 release claim is not defensible while these gates remain open.

The highest-value immediate action is to authorize persistence of the verified new checkpoints to the existing private project bucket and complete notebook publication in the established AWS workspace. Research still needs a defensible stopping decision about remaining feasible feature hypotheses; true carry, inventories, macro vintages, weather, seasonality, fundamentals, options, and text lack verified date/instrument/release mappings. **The final 247 origins remain untouched.**
