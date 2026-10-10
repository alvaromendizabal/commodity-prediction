# Final research closeout

**Project owner:** Alvaro Mendizabal  
**Status:** complete  
**Scope:** employer-facing research and ML-engineering record

## Public demo release

The [Forecast Lab](https://alvaro-forecasting-lab.tartmacaw2.chatgpt.site) adds a live CPU forecasting workflow over authored synthetic series: forward-only ridge fitting, release-time eligibility, matched baselines and inspectable exports. It makes the research's timing contracts interactive. It changes none of the historical results below and makes no trading-profitability claim. [Run and verify](REPRODUCIBILITY.md).

## Executive summary

This project developed a point-in-time forecasting research system for 424 commodity-related targets across four forecast horizons. What began as a modeling study grew into a broader ML-engineering program covering temporal data contracts, model-family integration and evaluation, cloud execution, experiment recovery, uncertainty-aware promotion, and increasingly strict evaluation design.

The final result is valuable because the project did not stop at the first attractive validation gain. Multiple apparently promising systems were tested on later regimes, many were rejected, and the validation design was strengthened whenever evidence showed that an earlier screen was too optimistic. The completed record therefore demonstrates both modeling breadth and scientific judgment.

## Research program

Major families I implemented, integrated or stress-tested during the project included:

1. target-routed tree ensembles and mixed-horizon panels;
2. direct MLP/RNN systems and short-sequence models;
3. recursive underlying-feature forecasting;
4. feature-token Transformers with released-target context;
5. zero-fit and refit-based online adaptation;
6. group-wise LightGBM/Ridge/CatBoost concepts;
7. attention/residual/autoencoder online ensembles;
8. stable-correlation and target-clustering representations;
9. regularized covariance and rank-ordering systems;
10. joint multi-target CatBoost;
11. heterogeneous RF/XGBoost/CatBoost/DNN ensembles;
12. market-graph and sequential-market-phase representations;
13. A/B symmetry and ordinal rank classification;
14. competition-sized 73-day stability/retrieval systems;
15. causal online strategy weighting and inverse-cluster experts;
16. feature-state LSTM forecasting with released-label correction;
17. lag-group-specific experts;
18. lag-table state-space transitions and historical-state retrieval.

Several branches involved large fit counts: 2,544 group-wise fits, 206 causal online-refit fits, and 24 Transformer fits. Later experiments intentionally used stricter early-stop gates to avoid spending compute on unstable directions.

## Validation evolution

The most important technical story is the evaluation system itself.

### 1. Matched-population debugging

A mixed-horizon candidate appeared to improve an initial screen. When a broader one-day expansion regressed, a saved-prediction reconciliation showed that shared predictions matched exactly; the result changed because the target composition changed. This avoided misdiagnosing a modeling bug and prevented post-hoc subset selection.

### 2. Temporal replication

The original candidate was frozen and evaluated on later chronological periods. It improved both later periods, but the pooled paired interval still crossed zero. The system therefore remained unpromoted despite a positive point estimate.

### 3. Locked historical assessment

A separate historical population was evaluated only after model and assessment contracts were fixed. The final comparison did not demonstrate an advantage over a simple training-only historical-mean control, and the result was preserved without reselection.

### 4. Competition-sized windows

Later research used 73-day windows to better expose regime sensitivity. A small regularized-covariance correction was the first later-stage component to improve three consecutive short windows with a positive paired interval. More complex adaptive and supervised systems were required to beat that promoted anchor consistently and generally did not.

### 5. Final delayed-label replay

After the competition period, the project reconstructed the official delayed-label information flow from the four lag files. Candidate selection remained training-only; test-period labels were revealed to the simulation only when the evaluation interface would have made them available. Final labels were used for scoring only after the prediction stream was fixed.

This replay exposed a substantial distribution shift between the historical selection windows and the final period. That was the decisive closeout evidence: continued adaptation after observing the final period would no longer be an honest generalization test.

## Engineering accomplishments

The research infrastructure was treated like production ML rather than a sequence of disposable notebooks:

- canonical AWS/SageMaker workspace;
- deterministic run IDs and artifact hashes;
- restartable checkpoints and reuse of completed expensive stages;
- structured human-readable and JSONL logs;
- CPU/GPU/RAM/disk telemetry and cost estimates;
- process-conflict guards to prevent accidental duplicate workloads;
- self-tests, smoke tests, failure-path tests, and return bundles on both success and failure;
- executed notebooks with persisted Plotly outputs;
- explicit separation between implementation failures and valid negative experiments;
- public/private artifact boundaries that preserve technical evidence without publishing restricted data or private prediction matrices.

## What the project establishes

This body of work demonstrates the ability to:

- own an ML problem end to end;
- translate an evaluation rule into point-in-time data contracts;
- translate documented methods into tested implementations and controlled experiments;
- design controlled ablations and matched comparisons;
- debug apparent model regressions without unnecessary retraining;
- manage GPU and CPU experimentation in AWS;
- preserve negative results and update priorities from evidence;
- identify validation mismatch and strengthen the test instead of rationalizing unstable gains;
- produce a reproducible technical record suitable for engineering review.

The final conclusion is intentionally methodological: **robust forecasting research requires the evaluation system to be at least as carefully engineered as the model.**

## Public release boundary

The closeout release contains the narrative, aggregate evidence, machine-readable summaries, reproducibility checks, and presentation notebooks. It does not publish raw competition data, private prediction matrices, fitted weights, AWS logs, credentials, or the latest complete private orchestration code.

The research cycle is complete. Future work would require a newly defined dataset and evaluation contract rather than continued adaptation to this completed record.
