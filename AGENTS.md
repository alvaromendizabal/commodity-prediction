# Project instructions

- Keep the canonical notebook/code filenames. Edit in place; do not create fix/fixed/repair variants.
- The project is an offline research and portfolio workflow. Kaggle submissions are out of scope.
- Feature engineering is a completion gate. It remains open until major plausible families have been tested for leakage, screened on training data, ablated, and evaluated for stability and diminishing returns.
- Keep the final 252 dates untouched for selection. Use the purged chronological folds in configs/research.json. Targets for horizon h span t+1 through t+h+1 and are released at t+h+1.
- Current feature comparisons additionally apply configs/feature_study.json's five-date terminal embargo: 535 validation dates, ending at date 1703. Rescore preserved initial predictions on this same window; do not directly compare their old 540-date metrics.
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
