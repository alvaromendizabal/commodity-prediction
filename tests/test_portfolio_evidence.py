"""Negative checks for the public evidence contract, independent of private data."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_portfolio", ROOT / "scripts/verify_portfolio.py"
)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def evidence(tmp_path):
    for relative in (
        "reports/portfolio_summary.json",
        "reports/final_research_ledger.json",
        "notebooks/28_final_research_closeout.ipynb",
        "scripts/verify_portfolio.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def change(root, relative, mutation):
    path = root / relative
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))


def test_committed_evidence_passes():
    checks = verifier.verify()
    assert checks["saved_executed_cells"] == 5
    assert checks["research_families"] == 20


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.4", True, 0.4])
def test_invalid_or_inconsistent_metrics_fail(evidence, value):
    change(
        evidence,
        "reports/portfolio_summary.json",
        lambda data: data["temporal_replication"].update(candidate=value),
    )
    with pytest.raises(ValueError):
        verifier.verify(evidence)


def test_future_label_selection_fails(evidence):
    change(
        evidence,
        "reports/portfolio_summary.json",
        lambda data: data["later_research"]["post_competition_exact_replay"].update(
            test_labels_used_for_selection=True
        ),
    )
    with pytest.raises(ValueError, match="test-label selection"):
        verifier.verify(evidence)


@pytest.mark.parametrize("field,value", [("execution_count", None), ("outputs", [])])
def test_missing_execution_evidence_fails(evidence, field, value):
    def mutation(data):
        code_cell = next(cell for cell in data["cells"] if cell["cell_type"] == "code")
        code_cell[field] = value

    change(evidence, "notebooks/28_final_research_closeout.ipynb", mutation)
    with pytest.raises(ValueError, match="closeout cell"):
        verifier.verify(evidence)


def test_error_output_fails(evidence):
    def mutation(data):
        code_cell = next(cell for cell in data["cells"] if cell["cell_type"] == "code")
        code_cell["outputs"] = [{"output_type": "error", "ename": "ValueError"}]

    change(evidence, "notebooks/28_final_research_closeout.ipynb", mutation)
    with pytest.raises(ValueError, match="contains an error"):
        verifier.verify(evidence)


def test_duplicate_family_fails(evidence):
    change(
        evidence,
        "reports/final_research_ledger.json",
        lambda data: data["families"].append(data["families"][0]),
    )
    with pytest.raises(ValueError, match="duplicate families"):
        verifier.verify(evidence)


def test_checks_remain_active_under_python_optimization(evidence):
    change(
        evidence,
        "reports/portfolio_summary.json",
        lambda data: data.update(project_status="unfinished"),
    )
    result = subprocess.run(
        [sys.executable, "-O", str(evidence / "scripts/verify_portfolio.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "project must be complete" in result.stderr
    assert "PORTFOLIO_VERIFY=PASS" not in result.stdout


def load_executor():
    spec = importlib.util.spec_from_file_location(
        "execute_public_closeout", ROOT / "scripts/execute_public_closeout.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_notebook_outputs():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/28_final_research_closeout.ipynb", as_version=4)
    families = json.loads((ROOT / "reports/final_research_ledger.json").read_text())["families"]
    cell = next(cell for cell in notebook.cells if cell.id == "family-ledger")
    # Authored unit-test output; never saved as a public execution artifact.
    cell.outputs = [
        nbformat.v4.new_output(
            "stream", name="stdout", text="\n".join(r["family"] for r in families)
        )
    ]
    return notebook, families


def test_executor_validates_complete_synthetic_output(synthetic_notebook_outputs):
    notebook, families = synthetic_notebook_outputs
    load_executor().validate_saved_outputs(notebook, families)


def test_executor_rejects_truncated_family_output(synthetic_notebook_outputs):
    notebook, families = synthetic_notebook_outputs
    cell = next(cell for cell in notebook.cells if cell.id == "family-ledger")
    cell.outputs[0].text = "counts only; no family rows\n"
    with pytest.raises(ValueError, match="family ledger output is incomplete"):
        load_executor().validate_saved_outputs(notebook, families)


def test_executor_rejects_unexecuted_cell(synthetic_notebook_outputs):
    notebook, families = synthetic_notebook_outputs
    cell = next(cell for cell in notebook.cells if cell.id == "family-ledger")
    cell.execution_count = None
    with pytest.raises(ValueError, match="Missing execution evidence"):
        load_executor().validate_saved_outputs(notebook, families)
