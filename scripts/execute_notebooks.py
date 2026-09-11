"""Execute canonical notebooks with a fresh kernel, preserving existing filenames."""

import argparse
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from review_feature_study import review
from traitlets.config import Config

from commodity_prediction.runtime import RunLog


def _optional_lineage(root: Path, report_name: str) -> str | None:
    path = root / "reports" / report_name
    if not path.exists() or path.stat().st_size == 0:
        return None
    report = json.loads(path.read_text())
    lineage = report.get("lineage")
    return str(lineage) if lineage else None


def execute(path: Path, root: Path, log: RunLog) -> None:
    with log.stage(path.name):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="commodity",
            config=Config({"KernelManager": {"transport": "ipc"}}),
            resources={"metadata": {"path": str(root)}},
        ).execute()
        notebook.metadata["execution_engine"] = "nbclient_ipc_kernel"
        required = {
            "study_lineage": "feature_study.json",
            "domain_lineage": "domain_study.json",
            "attribution_lineage": "tree_attribution.json",
            "robustness_lineage": "domain_robustness.json",
            "compact_lineage": "compact_study.json",
            "risk_state_lineage": "risk_state_study.json",
        }
        for metadata_name, report_name in required.items():
            report = json.loads((root / "reports" / report_name).read_text())
            notebook.metadata[metadata_name] = report["lineage"]
        optional = {
            "released_context_lineage": "released_context_study.json",
            "market_path_lineage": "market_path_study.json",
            "rank_prior_probe_lineage": "rank_prior_probe_execution.json",
            "rank_prior_fit_lineage": "rank_prior_fit_execution.json",
        }
        for metadata_name, report_name in optional.items():
            lineage = _optional_lineage(root, report_name)
            if lineage:
                notebook.metadata[metadata_name] = lineage
        temporary = path.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Canonical notebook filename to execute; repeat to select more than one.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    review(root)
    os.environ["COMMODITY_ROOT"] = str(root)
    log = RunLog(root / "logs/notebooks.jsonl")
    available = {path.name: path for path in (root / "notebooks").glob("*.ipynb")}
    requested = args.notebook or sorted(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown canonical notebook(s): {unknown}")
    for name in requested:
        execute(available[name], root, log)


if __name__ == "__main__":
    main()
