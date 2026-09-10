"""Create canonical notebook sources; execution is a separate required stage."""

from pathlib import Path

import nbformat as nbf


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
from scripts.notebook_support import project_root, checked_reports, show_figure
root = project_root()
audit, research = checked_reports(root)
config = json.loads((root / "configs/research.json").read_text())
display(Markdown(f"**Verified experiment:** `{research['lineage'][:16]}` · **Feature gate:** open"))"""
    definitions = {
        "00_data_audit.ipynb": [
            (
                "md",
                "# Commodity forecasting | Data and prediction contract\n\nA research project built around 424 multi-horizon return targets from LME, JPX, US equities, and FX. This notebook establishes what can be known at prediction time and which data is reserved for final evaluation.\n\n**Research question:** can economically motivated representations improve stable cross-sectional return ranking?\n\n[Dataset](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) · [Official target construction](https://www.kaggle.com/code/sohier/mitsui-target-calculation-example/) · Next: `01_eda.ipynb`.",
            ),
            ("code", setup),
            (
                "code",
                """from commodity_prediction.data import load_data, make_folds
x, y, pairs = load_data(root)
folds, development_stop = make_folds(len(x), config)
inventory = pd.DataFrame({"Measure": ["Observed dates", "Input columns", "Return targets", "Development dates", "Reserved holdout dates"],
                          "Count": [len(x), x.shape[1], y.shape[1], development_stop, len(x) - development_stop]})
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
                "## Reserve the final 252 dates\n\nThree expanding walk-forward folds use 180 validation dates each. Five dates are purged before every validation block, so all fitting labels were released strictly before its first prediction. The final 252 dates are excluded from feature construction, EDA, screening, and model selection. The downloadable mock test file overlaps training and is not a valid holdout.",
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
        ("Reserved holdout", development_stop, len(x) - development_stop, "#CAD3DE")]:
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
        "02_feature_research.ipynb": [
            (
                "md",
                "# Commodity forecasting | Feature research\n\n**Completion gate: OPEN.** This notebook reports the first controlled feature-family experiment. It does not claim that feature engineering is exhausted or that the final model is selected.\n\nThe initial comparison holds the algorithm and regularization fixed: Ridge regression with $\\alpha=100$. Only the representation changes. This isolates a first estimate of feature value before stronger model optimization.",
            ),
            ("code", setup),
            (
                "md",
                "## Candidate families and hypotheses\n\nReturns and momentum capture persistence; reversion measures departures from recent levels; volatility separates movement size from direction. Liquidity and OHLC structure describe trading activity and intraday shape. Market-relative features describe shared context, while target-aligned pair spreads use the competition's economic relationships. All inputs are observable at or before the prediction row.",
            ),
            (
                "code",
                """family_counts = pd.Series(research["family_counts"]).sort_values().rename_axis("Family").reset_index(name="Candidates")
fig = px.bar(family_counts, x="Candidates", y="Family", orientation="h", text="Candidates", title=f"{research['candidate_count']:,} domain-motivated candidates across nine families")
fig.update_traces(marker_color="#1F6C99", textposition="outside")
show_figure(fig, root, "feature_families", 570)""",
            ),
            (
                "md",
                "## Fit the screener only on training dates\n\nFor each fold and each ablation: reject over-40%-missing and constant features; remove exact duplicates; rank features by mean absolute target correlation using only observed training labels; remove highly correlated selected candidates; cap the diagnostic model at 96 features. Training medians, means, and scales are reused unchanged for validation.\n\nThe count is a candidate search budget, not a claim that thousands of independent signals exist. Final retained features remain undecided.",
            ),
            (
                "code",
                """lineage = research["lineage"]
features_dir = root / "artifacts" / lineage / "features"
from commodity_prediction.runtime import verify_checkpoint
assert verify_checkpoint(features_dir, lineage)
families = json.loads((features_dir / "families.json").read_text())
last_fold = research["folds_completed"] - 1
screen = pd.read_csv(root / "artifacts" / lineage / f"fold_{last_fold}" / "all_families" / "screening.csv")
decision = screen.assign(reason=screen.reason.str.split(":").str[0]).groupby("reason").size().sort_values(ascending=False)
display(decision.to_frame("Features"))
assert len(screen) == research["candidate_count"]
assert int((screen.status == "retained").sum()) <= config["max_features"]""",
            ),
            (
                "md",
                "## Official metric and additive ablations\n\nThe official metric is the mean daily cross-sectional Spearman rank correlation divided by its **population** standard deviation. Higher is better; no annualization factor is applied. The code is parity-tested against an independent Spearman calculation, including ties and missing labels.\n\n[Official metric](https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric). Each family is added separately to the same one-date-return reference; an additional experiment combines all families. These are historical validation scores, not Kaggle submissions.",
            ),
            (
                "code",
                """comparison = pd.DataFrame(research["comparison"])
display(comparison[["variant", "official_metric", "delta_from_reference", "fold_metrics", "retained_per_fold"]].round(4))
ordered = comparison.sort_values("official_metric")
fig = px.bar(ordered, x="official_metric", y="variant", orientation="h", text_auto=".3f", title="Which representations improve the fixed diagnostic model?")
fig.update_traces(marker_color="#27A394", textposition="outside")
fig.update_layout(xaxis_title="Pooled walk-forward correlation Sharpe", yaxis_title="Feature experiment")
fig.update_xaxes(range=[min(0, float(ordered.official_metric.min())) - 0.025, max(0, float(ordered.official_metric.max())) + 0.025])
show_figure(fig, root, "ablation_scores", 600)""",
            ),
            (
                "code",
                """rows = []
for result in research["results"]:
    rows.append({"Fold": f"Fold {result['fold'] + 1}", "Variant": result["variant"], "Metric": result["official_metric"]})
matrix = pd.DataFrame(rows).pivot(index="Variant", columns="Fold", values="Metric")
fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                aspect="auto", title="Average gains can hide unstable periods", labels={"color": "Metric"})
show_figure(fig, root, "fold_stability", 590)""",
            ),
            (
                "code",
                """delta = comparison.loc[comparison.variant != "reference"].sort_values("delta_from_reference").copy()
lower = delta.conditional_delta_95_interval.map(lambda v: v[0])
upper = delta.conditional_delta_95_interval.map(lambda v: v[1])
fig = go.Figure(go.Scatter(x=delta.delta_from_reference, y=delta.variant, mode="markers",
    marker={"size": 11, "color": "#1F6C99"},
    error_x={"type": "data", "symmetric": False, "array": upper - delta.delta_from_reference,
             "arrayminus": delta.delta_from_reference - lower}))
fig.add_vline(x=0, line_dash="dash", line_color="#9EAFBF")
fig.update_layout(title="Uncertainty around the initial feature gains", xaxis_title="Metric change versus reference · conditional 95% interval")
show_figure(fig, root, "ablation_uncertainty", 570)""",
            ),
            (
                "md",
                "The paired circular block bootstrap uses 20-date blocks and 500 resamples. It conditions on already fitted models, does not refit the screener, and is not adjusted for testing multiple families. It is an initial uncertainty diagnostic rather than a definitive significance claim.",
            ),
            (
                "code",
                """best = research["comparison"][0]
display(Markdown(f"**Initial leader:** `{best['variant']}` · pooled official metric **{best['official_metric']:.4f}** · change versus reference **{best['delta_from_reference']:+.4f}**.\\n\\nCompleted **{research['experiments_completed']} fold/representation experiments** across all 424 targets. The final 252 dates remain unused for model selection."))
display(pd.DataFrame({"Open research work": ["Conditional drop-family ablations and group permutation importance", "Nonlinear model controls without broad tuning", "Longer-window regimes, nonlinear interactions, and stability selection", "Target/pair-specific screening versus shared screening", "Release-aware pooling or target encoding only where it adds information", "External point-in-time data assessment and source vintages", "Block-bootstrap sensitivity and model-selection uncertainty", "Locked final feature gate, then final model comparison and holdout"]}))""",
            ),
            (
                "md",
                "## Decision\n\nKeep the feature gate open. This experiment establishes a reproducible reference and empirical family comparisons. It does not yet establish diminishing returns, a production model, a trading strategy, or an employer-facing project completion score. Follow-up experiments must preserve this lineage, reuse verified stages, and document negative results as carefully as positive ones.",
            ),
        ],
    }
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
