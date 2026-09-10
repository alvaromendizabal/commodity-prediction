"""Execute canonical notebooks with a fresh kernel, preserving existing filenames."""

import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from review_feature_study import review
from traitlets.config import Config

from commodity_prediction.runtime import RunLog


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    review(root)
    os.environ["COMMODITY_ROOT"] = str(root)
    log = RunLog(root / "logs/notebooks.jsonl")
    for path in sorted((root / "notebooks").glob("*.ipynb")):
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
            report = json.loads((root / "reports/feature_study.json").read_text())
            notebook.metadata["study_lineage"] = report["lineage"]
            domain = json.loads((root / "reports/domain_study.json").read_text())
            notebook.metadata["domain_lineage"] = domain["lineage"]
            attribution = json.loads((root / "reports/tree_attribution.json").read_text())
            notebook.metadata["attribution_lineage"] = attribution["lineage"]
            robustness = json.loads((root / "reports/domain_robustness.json").read_text())
            notebook.metadata["robustness_lineage"] = robustness["lineage"]
            compact = json.loads((root / "reports/compact_study.json").read_text())
            notebook.metadata["compact_lineage"] = compact["lineage"]
            temporary = path.with_suffix(".ipynb.tmp")
            nbformat.write(notebook, temporary)
            temporary.replace(path)


if __name__ == "__main__":
    main()
