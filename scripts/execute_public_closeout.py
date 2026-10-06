"""Execute only the public aggregate closeout and persist a review copy."""

from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def validate_saved_outputs(notebook: nbformat.NotebookNode, families: list[dict[str, str]]) -> None:
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    if len(cells) != 5:
        raise ValueError("Expected five aggregate-analysis cells")
    for cell in cells:
        if type(cell.execution_count) is not int or cell.execution_count < 1 or not cell.outputs:
            raise ValueError(f"Missing execution evidence: {cell.id}")
        if any(output.output_type == "error" for output in cell.outputs):
            raise ValueError(f"Notebook error: {cell.id}")
    family_cell = next(cell for cell in cells if cell.id == "family-ledger")
    text = "".join(output.get("text", "") for output in family_cell.outputs)
    if any(row["family"] not in text for row in families):
        raise ValueError("Saved family ledger output is incomplete")


def main() -> None:
    source = ROOT / "notebooks/28_final_research_closeout.ipynb"
    output = ROOT / "artifacts/28_final_research_closeout.reproduced.ipynb"
    notebook = nbformat.read(source, as_version=4)
    families = json.loads((ROOT / "reports/final_research_ledger.json").read_text())["families"]
    NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    notebook.metadata["public_closeout_execution"] = {
        "engine": "nbclient Jupyter kernel",
        "environment": "GitHub Actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
        "scope": "Public aggregate analysis; no model training or private data",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    validate_saved_outputs(nbformat.read(output, as_version=4), families)
    print("PASS: five executed cells and complete family ledger persisted after save/reopen")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
