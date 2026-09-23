# Frontier Research Update

**Owner:** Alvaro Mendizabal  
**Purpose:** employer-facing summary of the current private research frontier  
**Publication boundary:** results and engineering decisions only; latest training code, exact hyperparameters, model weights, checkpoints, private prediction matrices, and AWS logs are withheld

## Why the project was reopened

The first research cycle produced a complete historical assessment and a curated portfolio. After closeout, the project was reopened privately to test materially different model families documented in high-performing public competition writeups.

The goal is not to copy public repositories. The workflow independently reconstructs the mechanisms that are documented, records where source details are unavailable, compares each candidate on the same dates and targets as the retained system, and blends components only when measured complementarity justifies it.

The previously evaluated historical assessment remains frozen. New research cannot turn that population back into an untouched test.

## Current verified frontier

| Research family | Public-method idea tested | Verified project result | Decision |
|---|---|---:|---|
| Retained mixed-horizon stack | Target-routed constituent features with tree ensembles and stacking | **0.289365** on 355 later development dates vs incumbent **0.276025** | Strongest retained later-period candidate; uncertainty still crossed zero |
| Expanded one-day stack | Broader application of the same target-stack idea | **0.385559** vs **0.403381** incumbent on its 180-date matched population | Rejected |
| Recursive sequence forecast | Forecast underlying series recursively, then derive target returns | Substantially below the incumbent in the controlled screening experiment | Rejected |
| Direct MLP/RNN reconstruction | Joint raw-input feed-forward and short-sequence models with ranking-aware loss ablation | Neural combination **0.105102** across the 355 later dates | Rejected |
| Three-way complementarity blend | Incumbent + retained panel + direct neural reconstruction | **0.287436** vs retained panel **0.289365** | Rejected; diversity did not translate into improvement |
| Feature-token Transformer | Joint feature tokens plus point-in-time released-target history | **Scientific score pending** | Current private frontier |

Scores above come from matched local development populations, not the Kaggle hidden leaderboard. Different rows should not be numerically ranked against one another when their evaluation populations differ.

## Engineering work behind the frontier

The newer work adds several production-style research controls beyond the original modeling study:

- **Exact data and prediction replay.** Saved predictions are keyed by explicit date identities; positional repair is prohibited.
- **Independent source lineage.** Public methods are reconstructed in project-owned code with source-described behavior separated from local reconstruction choices.
- **Checkpointed experiments.** Completed stages are reused rather than retrained after notebook, terminal, or managed-job interruption.
- **Managed GPU execution.** Transformer training uses deterministic SageMaker job identities, private S3 checkpoints, and network-isolated training containers.
- **Job reconciliation.** A rerun reuses a compatible in-progress or completed job, advances past terminal failed attempts, and prevents duplicate managed jobs.
- **Capacity-aware routing.** GPU selection can advance across approved G6e training instance classes only for genuine pre-allocation capacity failures; code/model failures do not trigger more expensive retries.
- **Failure classification.** Infrastructure and schema failures before the first fit are reported separately from scientific model outcomes.
- **Promotion discipline.** A new model must improve matched later-period evaluations rather than win only on the screen used to choose it.

## What has not been claimed

This project does **not** claim:

- an official Kaggle MITSUI leaderboard score;
- a competition win or medal;
- complete reconstruction of first- or second-place private systems;
- that every useful feature family has been exhausted;
- that a model is better because it is more complex;
- live trading profitability or production deployment.

The current Transformer experiment is not represented as a successful model until it actually completes fitting and produces matched-period evidence.

## Employer-facing takeaway

This project demonstrates more than model fitting. It shows the ability to design a leakage-aware research system, reproduce external ideas without conflating attribution with implementation, diagnose misleading gains, preserve negative results, manage cloud training safely, and make promotion decisions from matched evidence rather than intuition.

The public repository stays concise. The complete frontier implementation and operational state remain in the private AWS research workspace.
