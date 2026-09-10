# Commodity Prediction

Domain-driven feature engineering for multi-horizon commodity return prediction, with point-in-time market features, purged temporal validation, interpretable notebooks, and reproducible AWS experiments.

**Current phase: target-aware feature research. Feature gate: open.** This project studies the MITSUI&CO. Commodity Prediction Challenge as an offline forecasting problem. The latest controlled study found that a simple historical-mean benchmark remains stronger than the fitted alternatives. Final optimization and the one-time final test remain gated. Kaggle submissions are outside the project scope.

## Start with the notebooks

| Notebook | Question answered |
| --- | --- |
| [00 · Data audit](notebooks/00_data_audit.ipynb) | What is predicted, when are labels available, and which dates are reserved? |
| [01 · Exploratory analysis](notebooks/01_eda.ipynb) | How do coverage, missingness, and label availability shape the features? |
| [02 · Feature research](notebooks/02_feature_research.ipynb) | Do engineered representations beat credible controls, and which findings survive uncertainty? |

All three canonical notebooks execute in fresh AWS kernels. Twelve Plotly figures have static GitHub previews. The [AWS execution record](reports/aws_execution.json) records model replay, verified checkpoints, and notebook completion.

## Completed controlled study

- **1,961 ordered dates · 557 inputs · 424 targets · four horizons.**
- **15,098 candidates in 12 families:** 9,418 reused and 5,680 added regime, interaction, and released-label-history candidates.
- **69 new fold/representation experiments:** 23 declared variants across three expanding folds. All **30 initial experiments** remain preserved.
- Fixed diagnostic models: Ridge alpha 100 and a small histogram tree control. Output-specific screens retain at most **24 features per model**.
- Training-only preprocessing and screening, economic asset/pair routing, structural forecasts, 11 conditional family ablations, group permutation, temporal selection stability, and paired block-bootstrap sensitivity.

All scores below use the same **535 development validation dates**.

| Representation | Pooled official metric | Fold 1 | Fold 2 | Fold 3 |
| --- | ---: | ---: | ---: | ---: |
| Frozen target training means | **0.217739** | 0.162550 | 0.182488 | 0.296861 |
| Aligned one-date-return Ridge | 0.208660 | 0.188461 | 0.151816 | 0.282768 |
| Structural one-date-return Ridge | 0.207232 | 0.204698 | 0.150980 | 0.267481 |
| Extended aligned tree model | 0.139765 | 0.228130 | 0.037254 | 0.164609 |
| Extended aligned Ridge | 0.077233 | 0.164648 | −0.023056 | 0.105566 |
| Extended structural Ridge | 0.037435 | 0.140462 | 0.024037 | −0.064517 |

The strongest fitted model trails the historical-mean control by **−0.009080**; its conditional paired 95% interval is **[−0.0580, +0.0366]**. Adding the three new families to aligned Ridge changes the metric by **+0.007697**, with interval **[−0.0793, +0.0943]**. No positive matched feature gain survives the simultaneous bounds over the declared comparisons. Wider feature coverage has not established a dependable improvement.

Released-label history has the largest positive conditional ablation point estimate (**+0.029935**), but its interval includes zero. Removing pair features helps in the conditional interval; that finding does not survive simultaneous comparison bounds. Long regimes, structural expansion, and sign-stable screening do not justify promotion from these results.

![Controlled representation comparison](reports/figures/ablation_scores.png)

The official metric is **mean daily cross-sectional Spearman correlation / population standard deviation**. These are historical research estimates, not leaderboard results or realized trading Sharpe ratios. Intervals use 2,000 paired resamples within folds, with 10-/20-/40-date blocks. They condition on fitted models and do not capture all adaptive research or refitting uncertainty. See the [full study](reports/feature_study.json) and [protocol](docs/research-protocol.md).

## Exact screening counts

In the final fold, extended aligned screening evaluates **61,598 feature/output assignments**, retains **10,176**, and rejects **51,422**. Retained assignments use **4,241 unique features**; every output retains 24 features. Rejections comprise 27,447 unstable-sign/insufficient-history assignments, 18,687 feature-budget exclusions, 4,906 duplicates, 268 constants, and 114 correlated candidates. Assignment counts differ from the 15,098 generated unique candidates. Budget rejection does not prove absence of predictive information.

## Time boundary and evidence integrity

A target at origin t spans t+1 through t+h+1 and becomes available at t+h+1. Five dates are purged before each validation block. A terminal embargo gives validation lengths **180, 180, and 175**, ending at origin 1703; all validation outcomes are available by 1708. Imputation, normalization, screening, and fitting use only each fold's training data. Historical-label features obey their explicit release delays.

The initial 540-date evaluation had already inspected forward outcomes through date 1713. Rescoring cannot undo that inspection. Origins **1709–1713 are a permanent buffer**; only **1714–1960, or 247 untouched origins**, enter the eventual final test. [The final-evaluation restriction](configs/final_evaluation.json) is enforced by publication checks. Initial predictions are rescored on the common 535-date window without refitting; their earlier 540-date scores remain historical records.

The 69 new experiments completed in **20.7 minutes**. An identical second run reused all 73 feature/model/permutation stages, performed **zero fits and zero preprocessing**, and regenerated the summary in **9.2 seconds**. There are **105 verified manifests** across both studies. AWS verification replays every new saved model and compares its predictions with the sealed originals.

**52 automated tests** cover leakage, release timing, terminal embargo, metric parity, screening, structural identities, serialization, full-study resumability, tampering, atomic writes, heartbeat logs, and idempotent S3 synchronization. Ruff, formatting, type checks, and publication verification are CI gates.

## Reproduce

Python 3.12 and `uv.lock` define the environment. Obtain the [competition data](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) through your authorized access and extract it to `data/raw/`.

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python scripts/quality.py
uv run --frozen commodity research
uv run --frozen python -m commodity_prediction.studies.run
uv run --frozen python scripts/make_notebooks.py
uv run --frozen python -m ipykernel install --user --name commodity
uv run --frozen kaleido_get_chrome
uv run --frozen python scripts/execute_notebooks.py
```

Use a Linux/Jupyter environment that permits local kernel communication and Chrome rendering. Notebook execution applies a traceable publication correction to the frozen summary's legacy reservation wording; numerical results and model lineage remain unchanged. Raw data, labels, predictions, and fitted artifacts are excluded from Git.

## AWS research environment

Private SageMaker JupyterLab space **`commodity-prediction-dev`**, in **`us-west-2`**, with **`ml.m5.xlarge` (4 vCPU, 16 GiB RAM)**, **50 GB persistent EBS**, and a **60-minute idle timeout**. CPU compute is **$0.23 per active hour**, verified against AWS Pricing on 2026-09-09; storage and transfer are additional. Compute is stopped between work sessions while EBS and S3 artifacts persist.

An encrypted, versioned private S3 bucket stores source data and verified checkpoints. [AWS settings](configs/aws.json) and [checksum-pinned snapshots](configs/bootstrap.json) make startup resumable. Completed source/data/model fingerprints are preserved; new studies carry independent dependency fingerprints. Corrupt or stale checkpoints fail verification.

## Next research gate

The strongest current signal comes from cross-sectional target priors. Next, separate those priors from useful time-varying residual forecasts, test target/horizon normalization and robust pooled representations, and assess surviving interactions with nested temporal selection. External data require a defensible calendar mapping, source vintages, and appropriate rights; date IDs must not receive guessed calendar dates. The feature gate remains open until major plausible avenues and diminishing returns have adequate evidence.

## Data and sources

Competition data is **Competition Use Only** and is not redistributed here. Only source code and aggregate evidence are public. Other deployment or commercial use requires appropriate data rights.

[Dataset](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) · [Official metric](https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric) · [Target calculation](https://www.kaggle.com/code/sohier/mitsui-target-calculation-example/) · [Rules](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/rules)
