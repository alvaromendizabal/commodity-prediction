# Frontier Research Update

**Owner:** Alvaro Mendizabal  
**Portfolio status:** complete  
**Frontier status:** active  
**Evidence date:** 2026-09-23

## Objective

The frontier program tests materially different modeling mechanisms rather than repeatedly tuning one baseline. Public-method ideas are independently reimplemented, source-described behavior is separated from local reconstruction choices, and candidates are evaluated on matched chronological populations.

The already-recorded historical assessment remains frozen and is not reused for frontier selection.

## Current frontier

| Research family | Verified evidence | Decision |
|---|---:|---|
| Retained mixed-horizon panel | **0.289365** on 355 later dates vs incumbent **0.276025** | Retained reference; original paired interval crossed zero |
| Direct MLP/RNN | direct neural **0.105102**; three-way blend **0.287436** | Rejected |
| Feature-token Transformer | locked blend **0.253231** | Rejected |
| Zero-fit causal online adaptation | selected online system **0.280713** | Rejected |
| Lightweight true online refit | frozen blend **0.297114** | Not promoted: later-fold deltas +0.023589 / -0.009509; CI **[-0.023449, +0.036580]** |
| Group-wise LightGBM/Ridge | fold-0 **0.557014**; later **0.273250** | Rejected after failed later transfer |
| Full Attention/Residual/AutoEncoder online ensemble | 51 compatible fits preserved before engineering repair | Active; scientific score pending |

## Research controls added by the frontier

- explicit horizon-specific released-label timing;
- zero-fit prediction reconciliation before new compute;
- direct Studio GPU execution and resumable fit-level checkpoints;
- deterministic study/package identities and raw-input hashes;
- exact date-ID alignment rather than positional repair;
- fold-0-only selection where required;
- frozen later-period transfer tests;
- paired block uncertainty;
- explicit stop and promotion gates;
- compatibility rules for reusing checkpoints after engineering-only repairs;
- notebook gates that require executed Plotly evidence and static fallbacks;
- failure manifests that distinguish pre-fit, scientific, and presentation failures.

## Transfer failure as evidence

The group-wise reconstruction had the strongest screen result: **0.557014** versus **0.418181** on fold 0. Once frozen, it failed to transfer and pooled to **0.273250**, below the retained **0.289365** panel.

That negative result is intentionally preserved. The project does not treat model complexity or a screen-only gain as evidence of generalization.

## Active full-15 branch

The active system combines a broad engineered financial feature bank, fold-0 top-800 selector, Attention/Residual/AutoEncoder networks, ranking-aware hybrid loss, seven-day online refits, and validation/recency weighting.

Its first real run completed 51 compatible fits before an engineering edge case produced a one-row BatchNorm minibatch. The repair changes only batching for the singleton case, preserves every training row, uses a new study lineage, and imports old fits only when parent identity and hashes validate.

No scientific score is claimed yet.

## Submission status

An official-file schema audit for the late-submission notebook passed. The local gateway smoke test was skipped because the project interpreter did not already contain Polars; no package was installed just to force that check.

There is still **no verified official MITSUI leaderboard result for this repository**. Local development scores are not presented as hidden-leaderboard equivalents.
