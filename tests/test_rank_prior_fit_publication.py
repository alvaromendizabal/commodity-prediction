"""Publication-contract tests for the fitted rank-prior notebook evidence."""

import json
from pathlib import Path

from scripts.publish_rank_prior_fit_notebook import MARKER, publication_cells, write_static_svg


def test_rank_prior_publication_cells_are_bounded_and_decisive():
    root = Path(__file__).resolve().parents[1]
    cells = publication_cells(root)
    assert len(cells) == 2
    assert all(cell.metadata.get(MARKER) is True for cell in cells)
    text = "\n".join(cell.source for cell in cells)
    assert "0.301772" in text
    assert "0.307373" in text
    assert "0.309709" in text
    assert "stop" in text.lower()
    assert "final 247 origins remain untouched" in text.lower()


def test_static_svg_is_deterministic_and_browser_independent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/rank_prior_fit_execution.json").read_text())
    (tmp_path / "reports/figures").mkdir(parents=True)
    first = write_static_svg(tmp_path, report["scores"])
    content = first.read_text()
    first_bytes = first.read_bytes()
    second = write_static_svg(tmp_path, report["scores"])
    assert second.read_bytes() == first_bytes
    assert len(first_bytes) > 1000
    assert "0.3097" in content
    assert "current market rank" in content
    assert "final 247 origins untouched" in content.lower()


def test_rank_prior_execution_receipt_keeps_final_test_gated():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/rank_prior_fit_execution.json").read_text())
    assert report["status"] == "completed"
    assert report["feature_gate"] == "open"
    assert report["holdout_evaluated"] is False
    assert report["final_test_evaluations"] == 0
    assert report["new_fitted_models"] == 6
    assert report["model_checkpoints_replayed"] == 6
    assert report["maximum_prediction_replay_error"] == 0.0
    assert report["decision"]["advance_fitted_diagonal_rank"] is False
