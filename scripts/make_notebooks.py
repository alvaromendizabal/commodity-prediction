"""Create canonical notebook sources; execution is a separate required stage."""

from pathlib import Path

import nbformat as nbf
from compact_notebook import cells as compact_cells
from domain_notebook import cells as domain_cells
from domain_notebook import tree_cells
from feature_notebook import cells as feature_cells
from robustness_notebook import cells as robustness_cells


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = root / "notebooks"
    notebooks.mkdir(exist_ok=True)
    setup = """from pathlib import Path
import sys
if not Path("pyproject.toml").exists():
    sys.path.insert(0, str(Path.cwd().parent))
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Markdown
from scripts.notebook_support import project_root, checked_reports, checked_study, checked_domain, checked_attribution, show_figure
root = project_root()
audit, research = checked_reports(root)
study = checked_study(root)
domain = checked_domain(root)
attribution = checked_attribution(root)
study_config = json.loads((root / "configs/feature_study.json").read_text())
config = json.loads((root / "configs/research.json").read_text())
display(Markdown(f"**Verified domain attribution:** `{attribution['lineage'][:16]}` · **Feature gate:** open · Previous studies preserved"))"""
    definitions = {
        "00_data_audit.ipynb": [
            (
                "md",
                "# Commodity forecasting | Data and prediction contract\n\nA research project built around 424 multi-horizon return targets from LME, JPX, US equities, and FX. This notebook establishes what can be known at prediction time and which data is reserved for final evaluation.\n\n**Research question:** can economically motivated representations improve stable cross-sectional return ranking?\n\n[Dataset](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) · [Official target construction](https://www.kaggle.com/code/sohier/mitsui-target-calculation-example/) · Next: `01_eda.ipynb`.",
            ),
            ("code", setup),
            (
                "code",
                """from commodity_prediction.data import load_data
from commodity_prediction.studies.run import study_folds
x, y, pairs = load_data(root)
folds, development_stop = study_folds(len(x), config, study_config)
inventory = pd.DataFrame({"Measure": ["Observed dates", "Input columns", "Return targets", "Development dates", "Nominally reserved dates", "Untouched final-test dates"],
                          "Count": [len(x), x.shape[1], y.shape[1], development_stop, len(x) - development_stop, 247]})
display(inventory.set_index("Measure"))""",
            ),
            (
                "md",
                "## The target starts after the current date\n\nFor an asset with price $P$ and horizon $h$, the target at date $t$ is $\\log(P_{t+h+1}/P_{t+1})$. For a pair, subtract the second asset's return. Current-row market observations are available before prediction. Labels become available at $t+h+1$.\n\nThis is why a one-day target needs a two-date availability delay. Shifting targets by only the stated horizon would leak information.",
            ),
            (
                "code",
                """release = pd.DataFrame({"Horizon": [1, 2, 3, 4], "Label release delay": [2, 3, 4, 5], "Targets": [int((pairs.lag == h).sum()) for h in range(1, 5)]})
display(release.set_index("Horizon"))
display(Markdown(f"Target reconstruction compared **{audit['target_reconstruction_compared_values']:,}** observable development labels. Maximum absolute difference: **{audit['target_reconstruction_max_absolute_error']:.2e}**, below the 1e-5 rounding tolerance."))""",
            ),
            (
                "md",
                "## Protect 247 final-test dates and the boundary buffer\n\nThree expanding walk-forward folds use 180, 180, and 175 validation dates. A five-date terminal embargo ensures every validation target is fully observable before the reserved interval begins. Five dates are purged before every validation block, so all fitting labels were released strictly before its first prediction. All 252 nominally reserved origins remain outside development. The initial evaluation already inspected overlapping forward outcomes through date 1713, so origins 1709–1713 are a permanent buffer. Only origins 1714–1960 (247 dates) qualify for the eventual untouched final test. The downloadable mock test file overlaps training and is not a valid holdout.",
            ),
            (
                "code",
                """fig = go.Figure()
for fold in folds:
    label = f"Fold {fold.number + 1}"
    for part, start, length, color in [
        ("Train", 0, fold.train_stop, "#1F6C99"),
        ("Purge", fold.train_stop, fold.validation_start - fold.train_stop, "#EDAF43"),
        ("Validation", fold.validation_start, fold.validation_stop - fold.validation_start, "#27A394"),
        ("Boundary buffer", development_stop, 5, "#D76C64"),
        ("Untouched final test", development_stop + 5, len(x) - development_stop - 5, "#CAD3DE")]:
        fig.add_trace(go.Bar(x=[length], y=[label], base=start, orientation="h", name=part,
                             marker_color=color, showlegend=fold.number == 0))
fig.update_layout(barmode="overlay", title="Validation respects prediction-time information", xaxis_title="Date index", legend={"orientation": "h", "y": -0.22})
show_figure(fig, root, "validation_protocol", 450)""",
            ),
            (
                "md",
                "## Reproducibility and data handling\n\nThe public repository contains source, aggregate evidence, and executed notebooks. Raw market rows, labels, and model predictions remain in private storage under the competition's data-use terms. A data/source/configuration fingerprint connects every downstream result. Missing, changed, or corrupt checkpoints cause a clear failure rather than silent reuse.\n\n**Status:** data contract verified. Feature research is in progress; final evaluation has not run.",
            ),
        ],
        "01_eda.ipynb": [
            (
                "md",
                "# Commodity forecasting | Availability and market structure\n\nBefore selecting features, examine market coverage and the distinct data families. All analysis in this notebook uses the development interval. Raw values and row-level labels are not displayed.\n\nPrevious: `00_data_audit.ipynb` · Next: `02_feature_research.ipynb`.",
            ),
            ("code", setup),
            (
                "code",
                """from commodity_prediction.data import load_data
x, y, pairs = load_data(root)
x = x.iloc[:audit["development_dates"]]
y = y.iloc[:audit["development_dates"]]
origins = pd.Series([c.split("_")[0] for c in x.columns]).value_counts().rename_axis("Market").reset_index(name="Columns")
fig = px.bar(origins, x="Market", y="Columns", text="Columns", title="Input coverage differs substantially across markets")
fig.update_traces(marker_color="#1F6C99", textposition="outside")
show_figure(fig, root, "market_coverage", 470)""",
            ),
            (
                "md",
                "## Missingness is a market-state signal\n\nTrading holidays, exchange time zones, listing history, and absent observations create structured gaps. Backward filling would import future information. Candidate features therefore preserve missingness, record observation age, and use training-only median imputation during model fitting.",
            ),
            (
                "code",
                """missing = x.isna().mean().mul(100).rename("Missing percent").rename_axis("Feature").reset_index()
fig = px.histogram(missing, x="Missing percent", nbins=25, title="Most inputs are well covered; a sparse tail needs screening")
fig.update_traces(marker_color="#27A394")
fig.update_layout(yaxis_title="Input columns")
show_figure(fig, root, "missingness_distribution", 470)
display(pd.DataFrame({"Screening consideration": ["Inputs over 40% missing in development", "Entirely constant observed inputs", "Development dates"],
                      "Count": [int((x.isna().mean() > 0.4).sum()), int((x.nunique() < 2).sum()), len(x)]}).set_index("Screening consideration"))""",
            ),
            (
                "code",
                """availability = []
for origin in sorted(origins.Market):
    cols = [c for c in x.columns if c.startswith(origin + "_")]
    daily_missing = x[cols].isna().mean(axis=1)
    by_block = daily_missing.groupby(np.arange(len(x)) // 63).mean().mul(100)
    availability.extend({"Market": origin, "63-date block": int(i), "Missing percent": float(v)} for i, v in by_block.items())
matrix = pd.DataFrame(availability).pivot(index="Market", columns="63-date block", values="Missing percent")
fig = px.imshow(matrix, color_continuous_scale="Blues", aspect="auto", title="Availability changes over the historical interval", labels={"color": "Missing %"})
show_figure(fig, root, "availability_over_time", 440)""",
            ),
            (
                "md",
                "## Target availability must remain explicit\n\nMissing labels are excluded from each target's fit. The primary metric ranks all observed targets within a date. Horizon-specific diagnostics exclude dates with fewer than two observed targets and report their coverage; they do not change the primary metric.",
            ),
            (
                "code",
                """coverage = pd.DataFrame({"Horizon": [str(h) for h in range(1, 5)], "Observed labels (%)": [100 * y[pairs.loc[pairs.lag == h, "target"]].notna().mean().mean() for h in range(1, 5)]})
fig = px.bar(coverage, x="Horizon", y="Observed labels (%)", text_auto=".1f", title="Every horizon contains missing labels")
fig.update_traces(marker_color="#1F6C99")
fig.update_yaxes(range=[0, 105])
show_figure(fig, root, "target_coverage", 460)""",
            ),
            (
                "md",
                "## Implications for feature research\n\n1. Use returns and volatility-normalized quantities to compare instruments with different units.\n2. Preserve missing-market indicators and elapsed observation age.\n3. Test relative market strength and pair spreads against a simple return reference.\n4. Fit every imputer, scaler, and supervised screener inside the training interval.\n5. Require improvements to survive time splits and feature-group ablations.\n\nThis EDA motivates hypotheses; it does not prove predictive value.",
            ),
        ],
    }
    definitions["01_eda.ipynb"][-1:-1] = [
        (
            "md",
            "## Risk scales and shared market movements\n\nTarget assets have different return scales. These descriptive development-only plots motivate volatility normalization and common-factor residuals; their full-window estimates are never used as model preprocessing. Basket correlations average currently observed assets within each market, so changing coverage and asynchronous closes can affect them. Correlation does not establish transmission or causality.",
        ),
        (
            "code",
            """target_assets = sorted({asset for pair in pairs.pair for asset in pair.split(" - ")})
asset_returns = np.log(x[target_assets].where(x[target_assets] > 0)).diff()
risk = asset_returns.std(ddof=0).mul(100).rename("Observed return standard deviation (%)").rename_axis("Asset").reset_index()
risk["Market"] = risk.Asset.str.split("_").str[0]
fig = px.box(risk,x="Market",y="Observed return standard deviation (%)",color="Market",points="all",title="Target assets differ in return risk and measurement scale")
fig.update_layout(showlegend=False)
show_figure(fig,root,"domain_eda_risk_scales",500)
baskets = pd.DataFrame({market:asset_returns[[a for a in target_assets if a.startswith(market+"_")]].mean(axis=1) for market in sorted(risk.Market.unique())})
fig = px.imshow(baskets.corr(),text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu",aspect="auto",title="Observed market baskets share some return variation",labels={"color":"Correlation"})
show_figure(fig,root,"domain_eda_market_correlations",510)
display(pd.Series({"Distinct target assets":len(target_assets),"Directed target-pair strings":pairs.pair.nunique(),"Unordered target-asset sets":len({tuple(sorted(p.split(" - "))) for p in pairs.pair})},name="Prediction graph").to_frame())""",
        ),
    ]
    definitions["02_feature_research.ipynb"] = feature_cells(setup) + domain_cells() + tree_cells()
    definitions["02_feature_research.ipynb"] += robustness_cells()
    definitions["02_feature_research.ipynb"] += compact_cells()
    for filename, cells in definitions.items():
        notebook = nbf.v4.new_notebook()
        notebook.metadata = {
            "kernelspec": {
                "display_name": "Commodity Research (Python 3.12)",
                "language": "python",
                "name": "commodity",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
        notebook.cells = [
            nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
            for kind, text in cells
        ]
        nbf.write(notebook, notebooks / filename)


if __name__ == "__main__":
    main()
