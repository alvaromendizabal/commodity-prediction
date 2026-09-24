# Commodity forecasting: an engineering and research case study

**Project by Alvaro Mendizabal** · [GitHub](https://github.com/alvaromendizabal)

## Problem

The task is to rank short-horizon returns across 424 commodity-related targets spanning four forecast horizons. The hard part is not simply increasing model capacity: the system must respect delayed outcomes, heterogeneous market histories, a rank-based objective, and substantial temporal instability.

The local research metric is mean daily cross-sectional rank correlation divided by its population standard deviation.

## Ownership

The project covers problem framing, data contracts, point-in-time feature construction, chronological validation, pooled and target-specific models, public-solution reconstruction, GPU execution, checkpoint recovery, uncertainty analysis, and notebook communication.

AWS is the private execution workspace; GitHub is the durable source/evidence layer. Restricted competition data, private predictions, fitted weights/checkpoints, and credentials are not committed.

## Retained reference

The established 535-date development reference is **0.309709**. A third-place-inspired mixed-horizon panel was frozen and checked on two later development periods:

| Population | Incumbent | Panel | Difference |
|---|---:|---:|---:|
| 180 later dates | 0.167042 | 0.177350 | +0.010308 |
| 175 later dates | 0.390619 | 0.411945 | +0.021326 |
| Pooled 355 later dates | 0.276025 | **0.289365** | +0.013340 |

The pooled paired 95% interval was **[-0.009061, +0.035750]**. The panel remains the frontier reference, while the evidence is described as conditional rather than definitive.

## Why screening is not enough

A group-wise LightGBM/Ridge reconstruction produced the largest fold-0 screening gain in the frontier: **0.557014** versus **0.418181**.

The selection was frozen before the later periods. Its later-fold deltas were **+0.000139** and **-0.026877**, and its pooled score fell to **0.273250**, below the **0.289365** panel. The candidate was rejected instead of being retuned on those later outcomes.

That experiment is one of the strongest demonstrations in the repository of why research discipline matters as much as raw modeling complexity.

## A positive aggregate result that was not promoted

The lightweight seven-day online-refit system alone was weak, but a frozen 15% blend with the retained panel reached **0.297114**, a pooled improvement of **+0.007749**.

Its two later-fold deltas were **+0.023589** and **-0.009509**, and the paired interval was **[-0.023449, +0.036580]**. The effect was not consistent enough to replace the retained panel.

## Transformer reconstruction

The feature-token Transformer completed 24 fit stages and 427 epochs on a Studio GPU. The fold-0-selected blend later pooled to **0.253231**, below the **0.289365** panel, so the family was rejected.

The experiment is preserved because it demonstrates released-label timing, shared neural representation learning, deterministic GPU orchestration, and a complete negative result.

## Active full-15th-place-inspired frontier

The current active branch combines a broad engineered feature bank, top-800 fold-0 selector, Attention/Residual/AutoEncoder networks, ranking-aware hybrid loss, seven-day online refits, and validation/recency weighting.

A prior execution completed **51 compatible fits** before a 1,025-row generation exposed a singleton BatchNorm minibatch. The repair preserves every row by merging only a one-row final minibatch into the preceding batch and reuses old checkpoints only after parent study/package identity and model/scaler hashes validate.

This is an engineering repair, not a scientific result. The final ensemble score remains pending.

## Historical assessment

A separate recorded historical assessment spans 247 dates, IDs 1714-1960. The frozen incumbent scored **0.190453** versus **0.212085** for a training-only historical-mean control, with a paired 95% interval of **[-0.133651, +0.080336]**.

That population is already evaluated and is not reused to select frontier systems.

## What the portfolio establishes

The project demonstrates an integrated research system: point-in-time features, explicit leakage tests, multiple model families, chronological transfer checks, uncertainty, cloud recovery, artifact identity, and willingness to preserve negative results.

It does **not** claim that local metrics equal the competition leaderboard. No official MITSUI leaderboard score is currently verified for this repository.
