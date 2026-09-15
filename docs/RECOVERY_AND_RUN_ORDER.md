# Commodity research release: corrected 16–17, prepared 18–19

## Start here — one error, one controlled recovery

The uploaded notebook 16 records `PREFLIGHT_PASSED` with 62 earlier tests, followed by exact baseline replays for all three folds. The supervisor then raised `TypeError: emit() got multiple values for argument 'stage'`. Its heartbeat expanded a worker dictionary containing `stage` into a function that already received a positional stage name.

The corrected logger uses `worker_stage`, `worker_task`, `worker_completed_tasks`, and `worker_fit_attempts_total`. Regression tests explicitly enter the heartbeat branch with a saved state file. Earlier tests did not cover that branch.

This is a software/logging error, not a negative feature experiment and not evidence of a damaged AWS space. The notebook does NOT prove whether the worker wrote a feature/model checkpoint just before the supervisor stopped it. The recovery installer reads the saved state and manifests before deciding what may resume.

**Prepared and statically reviewed only. No new tests, notebooks, model fits, account operations or Git writes have been executed by the assistant.**

## Expected workspace

- Existing space: `commodity-prediction-dev`, Oregon (`us-west-2`).
- Domain: `QuickSetupDomain-20260902T115323` / `d-njhxv1erusdc`.
- Profile: `default-20260902T115323`.
- CPU: existing `ml.m5.xlarge`, 50 GB, **No Script**.
- Project: `/home/sagemaker-user/projects/commodity-prediction-manual`.
- Python: `/home/sagemaker-user/projects/commodity-prediction-current/.venv/bin/python`.
- Notebook kernel: `Commodity - manual (verified)`.
- Pinned research source: `d142a4cb57a5c4b2880f9341619e13a735b1cddc`. Remote HEAD has NOT been checked in this preparation.

Do not delete/recreate the space, reset Git, reinstall packages, delete earlier folders, rerun old studies, or select a lifecycle configuration.

## 1. Open the original space

From only ChatGPT open: open the AWS Console in another browser tab, sign in normally, choose Oregon, open SageMaker AI → Studio, select the domain/profile above, Open Studio → JupyterLab → `commodity-prediction-dev`. Run the existing space if stopped, keeping **No Script**, then Open JupyterLab. Starting it starts paid compute.

Save notebook 16 with Ctrl+S. Shut down its kernel (Kernel → Shut Down Kernel) before replacing the imported helper. Stop any other experiment kernels; do not delete their files.

## 2. Upload and run the new installer

Download `commodity_research_release.py` from ChatGPT. In JupyterLab's left file browser return to the home/top-level folder; click Upload Files and select the script. This is NOT CloudShell.

Open + → Terminal and run:

```bash
python3 -u "$HOME/commodity_research_release.py"
```

The installer performs no model loading/fitting or tests. It:

1. Accepts only the exact original helper/test revision and the diagnosed failure.
2. Checks analytical function text is unchanged for rounds 16–17.
3. Backs up source, test receipts, failed reports, worker logs and notebooks.
4. Retains any sealed stage whose manifest and recorded hashes match.
5. Quarantines an incomplete **feature-only** directory without deleting it. An incomplete or unrecorded model fit stops for review instead of being refitted automatically.
6. Preserves original study lineages, completed fit counts and elapsed time through an explicit source-compatibility record. The changed execution source is separately recorded.
7. Installs corrected canonical files and prepared rounds 18–19. Existing notebooks 16–17 are preserved, including outputs and edits.
8. Archives old test evidence; it is not relabeled as a pass for the correction.

Expected: `RESEARCH_RELEASE_INSTALLED`, then preserved fit counts and remaining time. A second identical installation prints `RELEASE_ALREADY_INSTALLED`; it does not reauthorize another failed run.

If `STOPPED`, do not proceed. Return the final error and `logs/manual_prior_research/recovery/release_installation.json` if created. A different round-17 error was not supplied and will not be guessed away.

## 3. Execute the corrected preflight yourself

```bash
cd "$HOME/projects/commodity-prediction-manual"
"$HOME/projects/commodity-prediction-current/.venv/bin/python" -u \
  scripts/commodity_prior_research.py --preflight
```

Only continue after `PREFLIGHT_PASSED`. This executes the prepared original and regression tests, including the actual heartbeat branch and one tiny synthetic repository-model fit. **Zero private-data forecasting fits** occur in preflight. The 120-second preflight ceiling contains an 80-second test limit.

Failures: keep `logs/manual_prior_research/preflight.json`, `tests.json`, `tests.log`; stop compute and return them. Do not edit expected hashes or remove tests.

## 4. Resume the same notebook 16

Open `projects → commodity-prediction-manual → notebooks → 16_prior_dynamics_replication.ipynb`.
Choose `Commodity - manual (verified)`, restart its kernel, then Run → Run All Cells. The same original notebook can be used: it imports the corrected helper.

- Same 24 prior-dynamics inputs and all four original panels.
- Up to eight total new replication fits across folds 0 and 2, minus any already completed ones.
- Four saved middle-period models and the original baselines are reused, not retrained.
- Only the original 360-second study allowance **minus recorded time already spent** remains.
- Sealed stages are verified on reuse. A new error does not get an automatic retry.
- Ten Plotly figures remain explicit inline notebook outputs.

Expected final `RESULT: NOTEBOOK_COMPLETE`; the decision can be `REVIEW_REPLICATED_PRIOR_DYNAMICS` or `NO_STABLE_PRIOR_DYNAMICS_GAIN`. Both mean successful scientific completion.

Save with Ctrl+S, close and reopen without execution, and confirm that inline charts remain visible. Download the round-16 report now. Do NOT run other-fold sweeps or change any model settings.

## 5. Complete notebook 17 separately

After round 16 completes without a software error, open `17_event_history_ablation.ipynb`, select the verified kernel and Run All. A negative scientific decision in round 16 does not block round 17; it uses the original baseline, not the candidate winner.

- Original 24 event-history representations, four panels, at most four fits.
- Own 240-second cumulative budget; any existing matching record is preserved.
- Last 8/21/63 legally released observed outcomes, not last K rows.
- First development period 1169–1348; saved control 0.4033810874.
- Success ends `RESULT: NOTEBOOK_COMPLETE`, with either review or stop-panel decision.

Save/reopen and download the report. **This is the immediate two-round recovery milestone. Return its results before executing rounds 18–19.** Both new rounds are included in full, not merely proposed, but the repaired pipeline's first real results must be inspected before another block of work.

## 6. Prepared next two rounds — after the recovery milestone is reviewed

Run the additional preflight once:

```bash
cd "$HOME/projects/commodity-prediction-manual"
"$HOME/projects/commodity-prediction-current/.venv/bin/python" -u \
  scripts/commodity_next_research.py --preflight
```

Require `NEXT_PREFLIGHT_PASSED`. It runs numerical, timing, formula, metadata, Copy-on-Write, control and heartbeat tests; it fits no forecasting model. Time ceiling 100 seconds, including an 80-second test subprocess.

Then run **18_released_sequence_ablation.ipynb** alone. Save, inspect and download. Next run **19_prior_error_memory_ablation.ipynb** alone only if the first completed without software error. They are scientifically independent and fixed in advance; neither uses the other's result for selection.

Each has 24 candidates, four panels, at most four fits, a 240-second worker ceiling, ten inline Plotly charts, exact baseline replay, and private manifest-verified checkpoints. They use the **last** development period 1529–1703 (175 origins); its saved control is **0.3906188649**. That is NOT the pooled 0.309709 and is NOT untouched holdout data.

Prepared code does not make any feature predictive. Runtime tests establish correctness, not a score gain. Model choices stay fixed in these comparisons; model/representation interactions remain an open research consideration.

## Resource envelope

Keep CPU at ml.m5.xlarge. Shut down unused kernels to avoid retaining earlier feature arrays. Require at least 2 GiB free disk. The worker-plus-direct-descendants RSS guard is 14 GiB; it is not an account-wide memory reservation and cannot include unrelated kernels.

- Installer: at most 120 seconds; no fitting.
- Corrected preflight: at most 120 seconds, tiny synthetic test fit only.
- Round16: remaining part of original 360 seconds.
- Round17: remaining part of original 240 seconds.
- Later next-round preflight: 100 seconds.
- Rounds18 and19: 240 seconds each.
- Each HTML/ZIP export: separately capped at 90 seconds; export failures do not justify model refits.

These are code ceilings, not measured runtimes for this release. For the immediate 16–17 milestone, budget roughly 20–30 minutes powered on for startup/checks/results/downloads; estimated compute cost = hourly instance rate × 0.33–0.50, plus storage/other applicable charges. Billing rate has not been inspected. Subsequent 18–19 is a separate session; do not pay for idle time between reviews.

## Exact return package

From `logs/manual_prior_research/`: `preflight.json`, `tests.json`, `tests.log` on failure. From `recovery/`: `release_installation.json`.

| Round | Folder under logs | Report |
|---|---|---|
| 16 | manual_prior_replication | commodity_prior_replication_report.json |
| 17 | manual_event_history | commodity_event_history_report.json |
| 18 | manual_sequence_state | commodity_sequence_state_report.json |
| 19 | manual_prior_error_memory | commodity_prior_error_memory_report.json |

Return the reports and executed notebooks for the rounds actually completed. Keep the `<slug>_dashboard.html` files for local interactive viewing. Download `<slug>_checkpoints.zip` privately; do not attach models, label-derived arrays, or raw data here and do not publish those ZIPs. A ZIP on EBS is not an off-disk backup.

## Stop compute

After downloads, return to Studio → JupyterLab → commodity-prediction-dev → **Stop space**. Confirm Stopped. Never choose Delete space. No script in this release starts or stops AWS resources. Do not remove the older `commodity-prediction-current/.venv` directory.

## Publication is separate from execution

Read `PRIVATE_RESEARCH_PUBLIC_PORTFOLIO.md` before any Git commit. The full research code belongs in a private repository. The public portfolio contains only a deliberately curated case study and aggregate evidence. Do not publish this research source ZIP, full notebooks 03–19, worker logs, model checkpoints or raw/derived datasets publicly. GitHub has not been changed by the assistant.
