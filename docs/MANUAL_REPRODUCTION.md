# Manual reproduction

Use the pinned project environment and legally acquired Kaggle data.
The manual study notebooks orchestrate scripts/commodity_*.py and retain their
individual source/data/configuration lineage checks. Do not run every research
notebook simply to view this repository. Historical results are available in
reports/manual_research and existing notebook outputs.

For rounds 18 and 19, first run the installed command:

```bash
cd "$HOME/projects/commodity-prediction-manual"
"$HOME/projects/commodity-prediction-current/.venv/bin/python" -u \
  scripts/commodity_next_research.py --preflight
```

Require NEXT_PREFLIGHT_PASSED, then run the corresponding notebook in the
verified kernel. The preflight is not the older rounds-16/17 preflight.
A missing preflight is a stopped dependency gate, not a model result.
Fresh external reproduction requires the preceding source/data/checkpoints;
no claim is made that cloning alone reproduces historical private artifacts.
The final holdout must remain separate; repeated development is exploratory.
