"""Shared presentation defaults and strict lineage checks for canonical notebooks."""

import base64
import json
import os
from pathlib import Path

from IPython.display import display

from commodity_prediction.runtime import digest, fingerprint


def project_root() -> Path:
    root = Path(os.environ.get("COMMODITY_ROOT", Path.cwd())).resolve()
    if not (root / "pyproject.toml").exists():
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError("Open this notebook within the commodity-prediction repository")
    return root


def checked_reports(root: Path) -> tuple[dict, dict]:
    audit = json.loads((root / "reports/data_audit.json").read_text())
    report = json.loads((root / "reports/research.json").read_text())
    lineage = json.loads((root / "reports/lineage.json").read_text())
    if audit["lineage"] != report["lineage"] or fingerprint(lineage) != report["lineage"]:
        raise ValueError("Reports do not share the same verified lineage")
    if lineage["config"] != json.loads((root / "configs/research.json").read_text()):
        raise ValueError("Research configuration has changed; rerun the experiment")
    for name, sha in lineage["files"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Stale source/data artifact: {name}")
    return audit, report


def checked_study(root: Path) -> dict:
    from commodity_prediction.studies.run import study_lineage

    report = json.loads((root / "reports/feature_study.json").read_text())
    evidence = json.loads((root / "reports/feature_study_lineage.json").read_text())
    config = json.loads((root / "configs/feature_study.json").read_text())
    actual, _ = study_lineage(root, config)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Feature study results have stale source, data, or configuration")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected feature-gate or holdout state")
    return report


def checked_domain(root: Path) -> dict:
    from commodity_prediction.domain.run import domain_lineage

    report = json.loads((root / "reports/domain_study.json").read_text())
    evidence = json.loads((root / "reports/domain_lineage.json").read_text())
    config = json.loads((root / "configs/domain_study.json").read_text())
    actual, _ = domain_lineage(root, config)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Domain study results have stale source, data, or configuration")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected domain research boundary")
    return report


def checked_attribution(root: Path) -> dict:
    from commodity_prediction.domain.attribution.run import attribution_lineage

    report = json.loads((root / "reports/tree_attribution.json").read_text())
    evidence = json.loads((root / "reports/tree_attribution_lineage.json").read_text())
    actual, _ = attribution_lineage(root)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Tree attribution results have stale dependencies")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected attribution research boundary")
    return report


def checked_robustness(root: Path) -> dict:
    from commodity_prediction.domain.robustness.reporting.run import analysis_lineage

    report = json.loads((root / "reports/domain_robustness.json").read_text())
    evidence = json.loads((root / "reports/domain_robustness_lineage.json").read_text())
    actual, _ = analysis_lineage(root)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Feature robustness results have stale dependencies")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected robustness research boundary")
    return report


def checked_compact(root: Path) -> dict:
    from commodity_prediction.domain.compact.run import compact_lineage

    report = json.loads((root / "reports/compact_study.json").read_text())
    evidence = json.loads((root / "reports/compact_lineage.json").read_text())
    actual, _ = compact_lineage(root)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Compact feature results have stale dependencies")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected compact research boundary")
    return report


def checked_risk_state(root: Path) -> dict:
    from commodity_prediction.domain.risk_state.run import study_lineage

    report = json.loads((root / "reports/risk_state_study.json").read_text())
    evidence = json.loads((root / "reports/risk_state_lineage.json").read_text())
    actual, _ = study_lineage(root)
    if report["lineage"] != actual or fingerprint(evidence) != actual:
        raise ValueError("Risk-state feature results have stale dependencies")
    if report["feature_gate"] != "open" or report["holdout_evaluated"]:
        raise ValueError("Unexpected risk-state research boundary")
    return report


def show_figure(fig, root: Path, name: str, height: int = 540) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        width=1120,
        font={"family": "Arial", "size": 15, "color": "#20334D"},
        title={"font": {"size": 23}, "x": 0.035},
        margin={"l": 85, "r": 45, "t": 90, "b": 80},
        paper_bgcolor="#FAFBFD",
        plot_bgcolor="#FAFBFD",
        colorway=["#1F6C99", "#27A394", "#EDAF43", "#D76C64", "#8070A6"],
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E3E9F0", zerolinecolor="#9EAFBF")
    output = root / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    png = fig.to_image(format="png", scale=1.5)
    (output / f"{name}.png").write_bytes(png)
    fig.write_html(output / f"{name}.html", include_plotlyjs="cdn")
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(fig.to_json()),
            "image/png": base64.b64encode(png).decode(),
            "text/plain": str(fig.layout.title.text),
        },
        raw=True,
    )
