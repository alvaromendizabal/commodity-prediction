"""Append the verified fitted rank-prior evidence to the canonical feature notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

MARKER = "rank_prior_fit_evidence"


def publication_cells(root: Path) -> list[nbformat.NotebookNode]:
    report = json.loads((root / "reports/rank_prior_fit_execution.json").read_text())
    if report["status"] != "completed" or report["holdout_evaluated"]:
        raise ValueError("Rank-prior fitted evidence is not publishable")
    scores = report["scores"]
    leader = report["decision"]["leader"]
    leader_score = report["decision"]["highest_reproduced_fitted_development_score"]
    markdown = nbformat.v4.new_markdown_cell(
        "## Metric-aligned rank prior: useful diagnostic, negative fitted transfer\n\n"
        "A training-only diagonal target-rank prior passed its predeclared no-fit information-value gate, "
        "but the matched fitted transfer test did **not** improve either frozen representation. "
        f"`admitted_tail_rank` scored **{scores['admitted_tail_rank']['official_metric']:.6f}** versus "
        f"**{scores['admitted_tail']['official_metric']:.6f}** for its control; `current_market_rank` scored "
        f"**{scores['current_market_rank']['official_metric']:.6f}** versus **{scores['current_market']['official_metric']:.6f}**. "
        f"The fitted leader therefore remains `{leader}` at **{leader_score:.6f}**.\n\n"
        "All six rank-augmented models were trained only on their fold prefixes and replay exactly from saved checkpoints. "
        "The final 247 origins remain untouched. The rank-prior family is stopped rather than tuned further on the same development folds."
    )
    markdown.metadata[MARKER] = True
    code = nbformat.v4.new_code_cell(
        """rank_fit = json.loads((root / "reports/rank_prior_fit_execution.json").read_text())
rank_rows = []
for name in ["admitted_tail", "admitted_tail_rank", "current_market", "current_market_rank"]:
    row = rank_fit["scores"][name]
    rank_rows.append({
        "Representation": name,
        "Official metric": row["official_metric"],
        "Fold 1": row["fold_scores"][0],
        "Fold 2": row["fold_scores"][1],
        "Fold 3": row["fold_scores"][2],
    })
rank_table = pd.DataFrame(rank_rows)
display(rank_table.style.format({c: "{:.6f}" for c in rank_table.columns if c != "Representation"}))
fig = px.bar(
    rank_table,
    x="Representation",
    y="Official metric",
    text_auto=".4f",
    title="Diagonal target-rank prior does not improve the matched fitted controls",
)
fig.add_hline(y=rank_fit["scores"]["current_market"]["official_metric"], line_dash="dash", annotation_text="Current fitted leader")
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Official development metric")
show_figure(fig, root, "rank_prior_fit_scores", 500)
for key, label in [
    ("admitted_tail_rank_vs_admitted_tail", "Admitted tail + rank vs control"),
    ("current_market_rank_vs_current_market", "Current market + rank vs control"),
]:
    comparison = rank_fit["matched_uncertainty"][key]
    interval = comparison["20"]["conditional_95_interval"]
    display(Markdown(
        f"**{label}:** delta **{comparison['delta']:+.6f}** · "
        f"20-date conditional 95% interval **[{interval[0]:+.6f}, {interval[1]:+.6f}]**"
    ))
display(Markdown(
    "**Decision:** stop the fitted diagonal-rank family. Preserve the positive no-fit diagnostic and the negative fitted transfer result as separate evidence; feature engineering remains open."
))"""
    )
    code.metadata[MARKER] = True
    return [markdown, code]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/02_feature_research.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not cell.metadata.get(MARKER)]
    notebook.cells.extend(publication_cells(root))
    report = json.loads((root / "reports/rank_prior_fit_execution.json").read_text())
    notebook.metadata["rank_prior_fit_lineage"] = report["lineage"]
    nbformat.validate(notebook)
    temporary = path.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(
        json.dumps(
            {
                "notebook": str(path.relative_to(root)),
                "lineage": report["lineage"],
                "evidence_cells": sum(bool(cell.metadata.get(MARKER)) for cell in notebook.cells),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
