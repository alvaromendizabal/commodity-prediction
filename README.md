# Commodity Forecasting | ML Engineering & Research Portfolio

**Alvaro Mendizabal** · [GitHub profile](https://github.com/alvaromendizabal)

End-to-end research on return ranking across **424 commodity-related targets and four forecast horizons**. The project combines point-in-time data engineering, horizon-aware label availability, purged temporal validation, domain-informed feature systems, public-solution reconstruction, GPU research, restartable AWS execution, and explicit model-promotion gates.

**Portfolio status: complete and employer-facing. Frontier status: active.** The earlier historical-assessment cycle is recorded and frozen; current frontier work is evaluated separately rather than retuned against that population.

**Start here:** [Frontier notebook](notebooks/28_frontier_research_ledger.ipynb) · [Technical case study](docs/PORTFOLIO_CASE_STUDY.md) · [Frontier update](docs/FRONTIER_RESEARCH_UPDATE.md) · [Reproducibility guide](docs/FRONTIER_REPRODUCIBILITY.md) · [Research ledger](reports/frontier_research_ledger.json)

## Engineering highlights

| Area | What I built and evaluated |
|---|---|
| Information timing | Horizon-specific label-release rules, causal feature construction, purged chronological folds, explicit evaluation boundaries |
| Feature engineering | Market paths, risk state, released-label priors, target-pair routing, short-horizon dynamics, market groups, engineered financial feature banks |
| Modeling | Pooled nonlinear models, tree stacks, recursive forecasting, MLP/RNN models, feature-token Transformers, online refits, group-wise ensembles, active Attention/Residual/AutoEncoder frontier |
| Validation | Fold-0-only selection where required, frozen later-period transfer, paired block uncertainty, stop/promotion gates, negative-result retention |
| Reliability | Resumable checkpoints, source/package identities, raw-input hashes, prediction replay, compatible-checkpoint recovery after engineering repairs |
| Cloud/GPU | SageMaker and Studio execution, quota/capacity diagnosis, deterministic study identity, direct L40S training, failure classification |
| Reproducibility | Public protocols, tests, integrated competitive source, executed notebooks, machine-readable results, exact private handoff hashes |

## Current verified evidence

Different rows can use different historical populations. Numbers should only be compared when their populations are explicitly matched.

| Evaluation / experiment | Result | Decision |
|---|---:|---|
| 535-date development reference | `current_market` **0.309709** | Historical development reference |
| 355-date later replication | incumbent **0.276025**; retained panel **0.289365** | +0.013340; paired 95% interval **[-0.009061, +0.035750]** |
| Direct MLP/RNN reconstruction | direct neural **0.105102**; blend **0.287436** | Rejected |
| Feature-token Transformer | locked blend **0.253231** vs panel **0.289365** | Rejected |
| Zero-fit online adaptation | selected online system **0.280713** vs panel **0.289365** | Rejected |
| Lightweight true online refit | blend **0.297114** vs panel **0.289365** | Numeric gain, **not promoted**: later-fold deltas +0.023589 / -0.009509; interval crosses zero |
| Group-wise LightGBM/Ridge reconstruction | fold-0 **0.557014**; later pooled **0.273250** vs panel **0.289365** | Rejected after failed temporal transfer |
| Full Attention/Residual/AutoEncoder online ensemble | 51 compatible fits preserved before an engineering-only batching repair | **Active frontier; scientific score pending** |
| Locked historical assessment, 247 dates | incumbent **0.190453**; training-only mean **0.212085** | No demonstrated advantage over control |

The local metric is mean daily cross-sectional rank correlation divided by its population standard deviation, without annualization.

These are **local historical evaluation scores, not official Kaggle leaderboard scores or trading returns**. The historical private-leaderboard winner's **0.63834** was measured on a different hidden population and is not directly comparable to the local development numbers above.

## Why negative results are part of the portfolio

The group-wise study is a useful example. Its frozen fold-0 blend jumped from **0.418181 to 0.557014**, the largest screening gain in the current frontier. On the two later periods, however, its deltas were approximately flat and negative, and its pooled score fell below the retained panel.

The project rejected that candidate instead of tuning against the later outcomes. The same promotion discipline prevented the Transformer and the numerically positive online-refit blend from replacing the retained panel.

## Reproducibility

The repository now integrates the third-place-inspired competitive reconstruction source, protocols, tests, and executed notebooks into `main`. It also publishes the current frontier evidence ledger, active-method documentation, testable experiment contracts, and cryptographic identities for the newer private AWS handoff packages.

Exact numeric reproduction additionally requires the authorized competition files and compatible ML/GPU dependencies; those data, private predictions, fitted weights/checkpoints, credentials, and operational secrets are not redistributed.

See [Frontier reproducibility](docs/FRONTIER_REPRODUCIBILITY.md).

## Active frontier

The current active branch independently reconstructs the public 15th-place combination of a broad engineered feature bank, fold-0 top-800 selection, Attention/Residual/AutoEncoder networks, ranking-aware hybrid loss, seven-day online refits, and validation/recency weighting.

A prior execution completed **51 compatible neural fits** before a 1,025-row training generation exposed a singleton BatchNorm minibatch. The repaired runner preserves every row by merging only a one-row final minibatch into the preceding batch and reuses old checkpoints only after parent study/package identity and model/scaler hashes validate.

That is an engineering repair, not a scientific result. No final full-ensemble score is claimed until the evaluation completes.

## Scope

This is an employer-facing research repository, not a live trading product. It does not claim a competition win, medal, current official MITSUI leaderboard score, trading profitability, or production deployment.
