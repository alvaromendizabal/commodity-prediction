# Project instructions

- Keep the canonical notebook/code filenames. Edit in place; do not create fix/fixed/repair variants.
- The project is an offline research and portfolio workflow. Kaggle submissions are out of scope.
- Feature engineering is a completion gate. It remains open until major plausible families have been tested for leakage, screened on training data, ablated, and evaluated for stability and diminishing returns.
- Keep the final 252 dates untouched for selection. Use the purged chronological folds in configs/research.json. Targets for horizon h span t+1 through t+h+1 and are released at t+h+1.
- Current feature comparisons additionally apply configs/feature_study.json's five-date terminal embargo: 535 validation dates, ending at date 1703. Rescore preserved initial predictions on this same window; do not directly compare their old 540-date metrics.
- The original 540-date evaluation already inspected forward outcomes through date 1713. Follow configs/final_evaluation.json: reserve origins 1709–1713 as a permanent buffer and use only the 247 untouched origins 1714–1960 for the eventual one-time final test. Do not claim all 252 original reserved outcomes are pristine.
- Keep the original core-module fingerprint and completed study checkpoints intact. Follow-up study modules carry their own dependency fingerprint. Target-specific screening counts are feature/output assignments, distinct from unique feature counts.
- Never use random splits, full-data preprocessing, backward filling, hidden notebook state, or the overlapping mock test set as validation.
- Target-derived features must use explicit historical release times. Do not apply ordinary cross-fold target encoding to this temporal panel.
- Use the official daily Spearman correlation Sharpe metric with population standard deviation. Historical CV is not a leaderboard score or realized trading performance.
- Keep raw data, labels, model predictions, and large checkpoint files in private storage under the Kaggle data-use terms. Publish aggregate results only.
- Implement typed, modular Python; explicit meaningful tests; UTC timestamps; stage and total elapsed time; heartbeats; artifact fingerprints; atomic checkpoints; verified resumability.
- Edit, test, execute, inspect, resolve failures, verify, then commit. Do not hand the user untested work when it can be run here.
- Preserve valid completed experiments and reuse them. Never silently accept stale source/data/model lineage.
- Notebooks are a primary employer-facing artifact: problem, data, EDA, domain reasoning, feature research, ablations, interpretation, and honest conclusions. Use polished Plotly figures with GitHub-compatible static fallbacks.
- Keep GitHub primarily Python/Jupyter, document meaningful commits/PRs, and verify checks before merging.
- Use connected AWS/GitHub access and minimize manual user steps. Keep compute bounded and stop idle resources while preserving EBS and S3 artifacts.
- Report exact candidate/retained/rejected counts, official-metric results, uncertainty, failed hypotheses, notebook/test status, cloud verification, and remaining research gaps.
- Do not declare state-of-the-art performance or project completion without evidence. The user's desired research-grade standard is the goal, not a self-assigned badge.

## Execution discipline and competitive objective

- Follow research → hypothesis → implementation → tests → bounded experiment → inspection → decision → checkpoint → report → continue. Diagnose an unsuccessful stage before another large experiment.
- Before spending substantial compute, verify schemas, required artifacts, small-scale correctness, checkpoint resume, the question answered, and explicit continue/stop criteria. Use smoke → sample → single fold → representative validation → full evaluation when justified.
- Give every substantial stage a wall-time budget, heartbeat and progress counters. Stop invalid, duplicate, clearly weaker or low-information directions promptly. Preserve every valid intermediate result and never refit it merely because publication failed.
- After each bounded milestone, report attempted/completed work, passes, failures, exact metrics, saved artifacts, GitHub status, the conclusion, the next action and why its expected information justifies compute.
- Feature research remains the priority. Compare plausible families with primary domain literature, academic methods, leading competition solutions, point-in-time external sources and actual expert information needs. Generic feature volume is insufficient.
- For every important family, document mechanism and information timing, test leakage explicitly, screen within training, measure matched additions/removals and temporal stability, and retain only with supporting evidence. Distinguish tested negative evidence from untested prerequisites or screening exclusions.
- Treat historical leading performance as a research target. Investigate the feature, validation, model and ensemble gaps separately. Historical CV on different dates does not establish a likely leaderboard win; no submission or unsupported record claim is permitted.
- Standing user authorization covers necessary project work in connected services. On 2026-09-10 the user explicitly authorized these 24 risk-state models, validation predictions and checkpoint manifests in s3://sagemaker-commodity-prediction-560403859723-us-west-2/ and notebook publication in the existing commodity-prediction-dev AWS space. Do not request this authorization again.
- Continue through manageable, measurable increments. Keep the final evaluation gate closed while major plausible high-value feature avenues remain unresolved. An employer-facing 9.9/10 standard is a delivery goal, never a self-certified claim.
