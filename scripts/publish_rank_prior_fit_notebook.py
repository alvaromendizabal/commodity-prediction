"""Append the verified fitted rank-prior evidence to the canonical feature notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

MARKER = "rank_prior_fit_evidence"


def write_static_svg(root: Path, scores: dict) -> Path:
    """Write a deterministic browser-independent static view of the four fitted scores."""
    names = ["admitted_tail", "admitted_tail_rank", "current_market", "current_market_rank"]
    values = [float(scores[name]["official_metric"]) for name in names]
    width, height = 1120, 500
    left, right, top, bottom = 110, 50, 85, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    lo = min(values) - 0.004
    hi = max(values) + 0.004
    span = hi - lo
    bar_w = 150
    gap = (plot_w - len(values) * bar_w) / (len(values) + 1)
    leader = float(scores["current_market"]["official_metric"])
    y_leader = top + plot_h * (hi - leader) / span
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFBFD"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#20334D}.title{font-size:23px;font-weight:600}.label{font-size:14px}.value{font-size:15px;font-weight:600}.axis{font-size:13px}</style>",
        '<text x="40" y="42" class="title">Diagonal target-rank prior does not improve the matched fitted controls</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#9EAFBF" stroke-width="1"/>',
        f'<line x1="{left}" y1="{y_leader:.2f}" x2="{width - right}" y2="{y_leader:.2f}" stroke="#8070A6" stroke-width="2" stroke-dasharray="7 5"/>',
        f'<text x="{width - right - 180}" y="{y_leader - 8:.2f}" class="axis">Current fitted leader</text>',
    ]
    palette = ["#1F6C99", "#27A394", "#EDAF43", "#D76C64"]
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        x = left + gap * (index + 1) + bar_w * index
        y = top + plot_h * (hi - value) / span
        h = top + plot_h - y
        display_name = name.replace("_", " ")
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" rx="4" fill="{palette[index]}"/>',
                f'<text x="{x + bar_w / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" class="value">{value:.4f}</text>',
                f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 30}" text-anchor="middle" class="label">{display_name}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="25" y="{top + plot_h / 2}" transform="rotate(-90 25 {top + plot_h / 2})" text-anchor="middle" class="axis">Official development metric</text>',
            '<text x="40" y="475" class="axis">Source: verified 535-date development evaluation; final 247 origins untouched.</text>',
            "</svg>",
        ]
    )
    output = root / "reports/figures/rank_prior_fit_scores.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    if output.stat().st_size <= 1000:
        raise ValueError("Static rank-prior SVG is unexpectedly small")
    return output


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
    write_static_svg(root, report["scores"])
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
                "static_svg": "reports/figures/rank_prior_fit_scores.svg",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
