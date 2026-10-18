"""One-time final-project selection; removed before merging."""
from pathlib import Path
import ast,json,re,shutil,subprocess
R=Path.cwd()
if subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()!='4dd453681c4813ecd9d36a5f76d7ee7f8421149d':
    raise RuntimeError('Main advanced; reconcile before curation')
def remove(n):
 p=R/n
 if p.is_dir():shutil.rmtree(p)
 elif p.exists():p.unlink()
def write(n,t):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
def select(d,names):
 for p in (R/d).iterdir():
  if p.name not in names:remove(p.relative_to(R))
p=R/'src/commodity_prediction/studies/evaluation.py'
t=p.read_text(); nodes=ast.parse(t).body
functions={'evaluate','block_indices','metric_draws','compare_predictions'}
write('src/commodity_prediction/studies/evaluation.py','"""Final forecast comparisons and fold-respecting paired uncertainty."""\nfrom __future__ import annotations\nimport numpy as np\nimport pandas as pd\nfrom commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations\n\n'+'\n\n'.join(ast.get_source_segment(t,n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in functions)+'\n')
select('src/commodity_prediction',{'__init__.py','metrics.py','runtime.py','studies'})
select('src/commodity_prediction/studies',{'__init__.py','evaluation.py'})
select('scripts',{'quality.py','verify_portfolio.py','execute_public_closeout.py'})
select('tests',{'test_portfolio_evidence.py','test_delayed_label_demo.py'})
select('notebooks',{'28_final_research_closeout.ipynb'})
select('reports',{'portfolio_summary.json','final_research_ledger.json','execution_recovery.json','figures'})
select('reports/figures',{'portfolio_final.svg','portfolio_temporal.svg'})
select('docs',{'assets','PORTFOLIO_CASE_STUDY.md','REPRODUCIBILITY.md','PUBLICATION_SCOPE.md','REPOSITORY_GUIDE.md','DEMO_REPLAY.md','ERROR_ANALYSIS.md'})
remove('AGENTS.md');remove('configs')
write('tests/test_forecast_contracts.py','''"""Actual retained metric, paired uncertainty and checkpoint contracts."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from commodity_prediction.metrics import daily_rank_correlations, correlation_sharpe
from commodity_prediction.studies.evaluation import block_indices, compare_predictions


def test_daily_metric_matches_independent_rank_correlations():
    y=pd.DataFrame([[1,2,2,4],[4,1,np.nan,3],[2,4,1,3]])
    p=pd.DataFrame([[2,1,1,4],[4,3,2,1],[1,3,4,2]])
    values=daily_rank_correlations(y,p)
    expected=[spearmanr(a[np.isfinite(a)],b[np.isfinite(a)]).statistic for a,b in zip(y.to_numpy(),p.to_numpy(),strict=True)]
    np.testing.assert_allclose(values,expected,atol=1e-15)
    assert correlation_sharpe(values)==pytest.approx(np.mean(expected)/np.std(expected,ddof=0))


@pytest.mark.parametrize('fault',['row_order','column_order','nonfinite','too_few','constant'])
def test_invalid_metric_inputs_fail(fault):
    y=pd.DataFrame([[1.,2.,3.],[3.,1.,2.]])
    p=y.copy()
    if fault=='row_order':p=p.iloc[::-1]
    elif fault=='column_order':p=p[p.columns[::-1]]
    elif fault=='nonfinite':p.iloc[0,0]=np.nan
    elif fault=='too_few':y.iloc[0,1:]=np.nan
    else:p.iloc[0,:]=1
    with pytest.raises((ValueError,ZeroDivisionError)):daily_rank_correlations(y,p)


def test_bootstrap_respects_every_fitted_fold_and_seed():
    indices=block_indices([7,11],3,60,42)
    np.testing.assert_array_equal(indices,block_indices([7,11],3,60,42))
    assert indices.shape==(60,18)
    assert (indices[:,:7]<7).all()
    assert ((indices[:,7:]>=7)&(indices[:,7:]<18)).all()


def test_paired_comparison_preserves_declared_populations():
    x=np.sin(np.arange(40)*1.4)*0.2+0.1
    scores={'reference':x,'candidate':x+0.02}
    rows=compare_predictions(scores,[17,23],[('candidate','reference')],{'bootstrap_block_dates':[2,4],'bootstrap_repetitions':60,'seed':42})
    assert len(rows)==2
    for row in rows:
        assert row['delta']==pytest.approx(correlation_sharpe(x+0.02)-correlation_sharpe(x))
        assert row['simultaneous_95_interval'][0]<=row['delta']<=row['simultaneous_95_interval'][1]


def test_checkpoint_reuse_requires_content_and_lineage(tmp_path):
    from commodity_prediction.runtime import atomic_json, seal_checkpoint, verify_checkpoint
    atomic_json(tmp_path / 'result.json', {'value': 1})
    seal_checkpoint(tmp_path, 'frozen-input', ['result.json'])
    assert verify_checkpoint(tmp_path, 'frozen-input')
    with pytest.raises(ValueError, match='Stale'):
        verify_checkpoint(tmp_path, 'other-input')
    (tmp_path / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        verify_checkpoint(tmp_path, 'frozen-input')
''')
write('README.md','''# Commodity Forecasting · Point-in-time evaluation

**Alvaro Mendizabal · Time-series ML · Delayed-label systems · Statistical evaluation**

![Point-in-time forecasting, availability controls and interactive comparison](docs/assets/public-demo-hero.svg)

[![Quality](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/commodity-prediction/actions/workflows/quality.yml)

I built a forecasting evaluation system for **424 targets across four horizons**, with training-only selection, exact label-release accounting, paired temporal comparisons and reproducible diagnostics. The final research reference uses target-routed trees with retained covariance and persistence/retrieval components. Private fitting and routing specifications are not distributed.

[**Open Forecasting Lab ↗**](https://alvaro-forecasting-lab.tartmacaw2.chatgpt.site) · [Case study](docs/PORTFOLIO_CASE_STUDY.md) · [Reproduce the public checks](docs/REPRODUCIBILITY.md)

## The result is a decision, not a performance claim

| Population | Recorded comparison | Decision |
|---|---|---|
| Later temporal replication, 355 dates | Candidate **0.289365** vs reference **0.276025** | Positive estimate; interval crossed zero; not promoted |
| Locked historical assessment, 247 dates | Reference **0.190453** vs training-only mean **0.212085** | No demonstrated advantage; no post-assessment reselection |
| Delayed-label replay | **134 scored dates**, including a **73-date** terminal segment | Selection remained training-only |

The metric is mean daily cross-sectional rank correlation divided by its population standard deviation. It is **not annualized Sharpe, realized trading profit or a live investment strategy**. The original negative assessment and inconclusive comparison remain intact. [Original record](reports/portfolio_summary.json)

![Locked historical comparison](reports/figures/portfolio_final.svg)

## Maintained engineering

```mermaid
flowchart LR
 A[Forecast origin] --> B[Only released labels]
 B --> C[Training-only control or frozen prediction]
 C --> D[Exact date and target alignment]
 D --> E[Daily rank correlations]
 E --> F[Fold-respecting paired uncertainty]
 F --> G[Recorded retain or reject decision]
```

| Concern | Actual public implementation | What it demonstrates |
|---|---|---|
| Delayed information | [Release calendar](examples/delayed_label_demo.py) | Future labels cannot enter an earlier snapshot |
| Metric fidelity | [Rank metric](src/commodity_prediction/metrics.py) | Ordered identities, missing targets, tied ranks and population standard deviation |
| Fair comparison | [Evaluation and paired blocks](src/commodity_prediction/studies/evaluation.py) | Fixed target populations and blocks contained within fitted folds |
| Interactive fitted control | [Browser engine](public-demo/engine.js) | Rolling, train-only fitting on authored synthetic series |
| Independent replay | [Replay verifier](tools/replay_public_demo.mjs) | Recompute actual browser outputs and reject modified exports |

## Run from a clone

No private data or cloud account is required:

```bash
python3 -S scripts/verify_portfolio.py
python3 -S examples/delayed_label_demo.py
python3 -S review/run.py --output /tmp/commodity-review --with-node
python3 -S review/run.py --output /tmp/commodity-review --with-node --check
```

Use Python 3.12; Node.js 22 or 24 enables actual browser and export-replay checks. With the locked Python environment, run `uv sync --frozen --extra dev`, then `uv run --frozen --extra dev python scripts/quality.py` and `uv run --frozen python scripts/execute_public_closeout.py`.

The [canonical executed review](notebooks/28_final_research_closeout.ipynb) replays public aggregate evidence. Forecasting Lab fits a reference on synthetic data and exposes label eligibility at every origin; it does not execute the private tree ensemble or reproduce its historical score.

## Publication boundary

The maintained tree contains final evaluation arithmetic, release-time contracts, the fitted public control, replay tests and compact original evidence. Superseded feature campaigns, unused model implementations, operator launchers and duplicate notebooks are removed. The compact historical selection ledger is retained for accountability, not advertised as a catalog of runnable model families.

Raw competition inputs, model weights, current feature identities and private routing recipes are excluded. No production deployment, unbiased future alpha or investment suitability is claimed. [Scope](docs/PUBLICATION_SCOPE.md) · [Source map](docs/REPOSITORY_GUIDE.md)
''')
write('docs/REPOSITORY_GUIDE.md','''# Maintained project map

`src/commodity_prediction/metrics.py` contains the exact ordered rank metric. `studies/evaluation.py` contains final comparisons and fold-respecting paired uncertainty. `runtime.py` contains atomic checkpoints and verified reuse. `examples/delayed_label_demo.py` is the explicit information-availability contract.

`public-demo/` is the actual fitted synthetic browser reference. `tools/` independently tests and replays it. `scripts/verify_portfolio.py` checks original aggregate evidence without pretending to possess private prediction rows. `scripts/execute_public_closeout.py` re-executes the single canonical review notebook.

`reports/portfolio_summary.json` and `reports/final_research_ledger.json` preserve the original final decision and historical selection evidence. They are not a list of current public model implementations. Superseded feature packages, private-runtime launchers and redundant notebook galleries have been retired.
''')
write('docs/PORTFOLIO_CASE_STUDY.md','''# Case study · Forecast only with information available at the time

I built a point-in-time forecasting workflow spanning 424 targets and four horizons. The important engineering problem was not simply fitting a powerful model: delayed labels, mismatched date populations and repeated selection could manufacture apparent improvements.

## Final approach and decision

The final private reference combined target-routed trees with retained covariance and persistence/retrieval components. On a later 355-date comparison, the candidate's positive estimate was not supported by a strictly positive paired interval. On the locked 247-date historical assessment, the reference scored 0.190453 versus 0.212085 for a training-only mean. I recorded that negative result without reopening the assessment for reselection. These are historical correlation-ratio metrics, not trading returns. [Original evidence](../reports/portfolio_summary.json)

## Public engineering that can be exercised

The retained rank metric validates exact date and target ordering, tied ranks, missing labels and nonfinite inputs. The paired uncertainty implementation keeps resampling inside fitted-model boundaries. The release-calendar example rejects future-label reads and proves that changing unavailable labels cannot change an earlier snapshot.

The browser control really refits on eligible synthetic history. Changing the forecast horizon, label delay or training window changes the fitted result, not a prerecorded chart. Exported forecasts are independently recomputed and checked for tampering. [Reproduction](REPRODUCIBILITY.md)

## Scope and ownership

My work covers the data-availability protocol, feature/model evaluation, artifact recovery, comparison diagnostics and public review experience. Private fitted models, exact routing specifications and competition data are not distributed. The public release validates component behavior and the recorded conclusion; it does not claim to reproduce private competitive performance or establish future profitability.
''')
for name in ['docs/REPRODUCIBILITY.md','docs/PUBLICATION_SCOPE.md','docs/ERROR_ANALYSIS.md','docs/DEMO_REPLAY.md']:
 p=R/name
 if p.exists():
  t=p.read_text()
  t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:m.group(0) if m.group(2).startswith(('http','#','mailto:')) or (p.parent/m.group(2).split('#')[0]).exists() else m.group(1),t)
  p.write_text(t)
p=R/'pyproject.toml';t=p.read_text()
t=re.sub(r'^  "(?:plotly|pyarrow|boto3|threadpoolctl|kaleido)[^\n]*\n','',t,flags=re.M)
t=re.sub(r'\[project.scripts\][\s\S]*?(?=\[tool.hatch)', '',t)
p.write_text(t)
p=R/'review/project.json';v=json.loads(p.read_text())
for c in v['cards']:
 if c.get('tests') in {'tests/test_feature_study.py','tests/test_study_storage.py'}:c['tests']='tests/test_forecast_contracts.py'
 if c.get('implementation')=='src/commodity_prediction/studies/storage.py':c['implementation']='src/commodity_prediction/runtime.py'
v['signals'][0]={'value':'424 targets','label':'Four-horizon evaluation contract'}
v['source_scopes'] += [{'directory':'src','suffixes':['.py']},{'directory':'tests','suffixes':['.py']}]
write('review/project.json',json.dumps(v,indent=2)+'\n')
p=R/'docs/REPRODUCIBILITY.md';t=p.read_text().replace('Earlier notebooks have their own historical data and checkpoint requirements: manual reproduction. Reviewing their saved outputs does not require launching the studies again.', 'The superseded notebook gallery is not part of the maintained release.');p.write_text(t)
text='\n'.join(p.read_text(errors='ignore') for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and p.suffix in {'.md','.html','.css','.js','.mjs','.py','.json'})
for p in (R/'docs/assets').rglob('*'):
 if p.is_file() and p.name not in text:p.unlink()
print('Removed retired experiment implementations; preserved final evidence and actual public controls.')
