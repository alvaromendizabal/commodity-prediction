#!/usr/bin/env python3
"""User-run local publication preparation. NEVER pushes, commits, or reads accounts.

Private research snapshot and public aggregate showcase are deliberately separate.
The public output is rebuilt from explicit scalar fields, never a redacted copy
of a full research notebook or report. Secret-pattern checks are incomplete and
cannot replace human inspection or a professional license/security review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-manual")
OUT = Path("/home/sagemaker-user/commodity-publication/release-16-19")
PIN = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
DIRS = ("src", "scripts", "tests", "configs", "docs", "notebooks", "reports")
ROOT_FILES = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
)
EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".json",
    ".toml",
    ".lock",
    ".txt",
    ".yml",
    ".yaml",
    ".png",
    ".svg",
    ".html",
    ".rst",
}
DENY_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "data",
    "artifacts",
    ".venv",
    "node_modules",
    "readonly_recovery",
    "recovery",
}
SECRET_PATTERNS = [
    r"AKIA[A-Z0-9]{16}",
    r"ASIA[A-Z0-9]{16}",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"https://[^\s/@:]+:[^\s/@]+@",
]
REPORTS = [
    (
        "manual_validation",
        "commodity_validation_report.json",
        "Normalization",
        "Three development periods",
    ),
    (
        "manual_session_ablation",
        "commodity_session_ablation_report.json",
        "Session representation",
        "First development period",
    ),
    (
        "manual_close_network",
        "commodity_close_network_report.json",
        "Cross-asset networks",
        "First development period",
    ),
    (
        "manual_feature_diagnosis",
        "commodity_feature_diagnosis_report.json",
        "Network removal diagnosis",
        "First development period",
    ),
    (
        "manual_rank_state",
        "commodity_rank_state_report.json",
        "Rank-state representation",
        "First development period",
    ),
    (
        "manual_target_context",
        "commodity_target_context_report.json",
        "Instrument context",
        "Middle development period",
    ),
    (
        "manual_response_encoding",
        "commodity_response_encoding_report.json",
        "Delayed response representation",
        "Middle development period",
    ),
    (
        "manual_innovation_ablation",
        "commodity_innovation_ablation_report.json",
        "Innovation representation",
        "Middle development period",
    ),
    (
        "manual_prior_dynamics",
        "commodity_prior_dynamics_report.json",
        "Historical dynamics screen",
        "Middle development period",
    ),
    (
        "manual_prior_replication",
        "commodity_prior_replication_report.json",
        "Historical dynamics replication",
        "Three development periods",
    ),
    (
        "manual_event_history",
        "commodity_event_history_report.json",
        "Observed-event histories",
        "First development period",
    ),
    (
        "manual_sequence_state",
        "commodity_sequence_state_report.json",
        "Released sequence structure",
        "Last development period",
    ),
    (
        "manual_prior_error_memory",
        "commodity_prior_error_memory_report.json",
        "Historical prior errors",
        "Last development period",
    ),
]
PUBLIC_FILES = {
    "README.md",
    "RIGHTS.md",
    ".gitignore",
    "docs/methodology.md",
    "docs/limitations.md",
    "evidence/summary.json",
    "notebooks/00_research_review.ipynb",
    ".github/workflows/portfolio.yml",
}


class Stop(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise Stop("Symlink or oversized source file: " + str(path))
    if any(x in DENY_DIRS for x in path.relative_to(ROOT).parts):
        raise Stop("Restricted source path selected.")
    return path


def secret_check(path):
    if path.suffix not in {".png"}:
        txt = path.read_text(errors="replace")
        for pat in SECRET_PATTERNS:
            if re.search(pat, txt):
                raise Stop(
                    "Potential credential material in "
                    + str(path.relative_to(ROOT))
                    + "; no matching value is printed."
                )


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def summarize(report, label, scope, source_hash):
    """Copy only explicit safe, bounded scalars. Never serialise the original report."""
    rows = []
    status = str(report.get("status", "UNKNOWN"))
    complete = status in {"ROUND_COMPLETE", "NOTEBOOK_COMPLETE"} or (
        status.startswith("NOTEBOOK_AND_") and status.endswith("READY")
    )
    if complete:
        source = report.get("comparison_rows", [])
        if not isinstance(source, list) or len(source) > 32:
            raise Stop("Unexpected comparison table.")
        for i, row in enumerate(source):
            score = row.get("official_metric")
            delta = row.get("delta_vs_current_market", row.get("matched_delta"))
            if not finite(score):
                continue
            rows.append(
                {
                    "panel": f"Panel {i + 1:02d}",
                    "official_metric": float(score),
                    "delta_vs_matched_control": float(delta) if finite(delta) else None,
                }
            )
    fit = report.get("new_training_fits")
    fit = fit if type(fit) is int and 0 <= fit <= 1000 else None
    return dict(
        study=label,
        evaluation_scope=scope,
        status="Completed" if complete else "Not completed / not verified",
        comparisons=rows,
        new_fits_in_this_report=fit if complete else None,
        report_sha256=source_hash,
        provenance="Owner-produced execution report; aggregate extraction is not an independent model replay.",
    )


def public_notebook():
    cells = []

    def md(t):
        cells.append(dict(cell_type="markdown", metadata={}, source=t.splitlines(True)))

    def code(t):
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                execution_count=None,
                outputs=[],
                source=t.splitlines(True),
            )
        )

    md(
        "# Commodity return forecasting — research review\n\nPublic presentation of selected aggregate evidence. Full feature implementation and model artifacts are private.\nScores from different evaluation periods are not interchangeable. No competition-winning claim is made.\n"
    )
    code(
        "from pathlib import Path\nimport json\nimport plotly.graph_objects as go\nfrom IPython.display import display\nimport pandas as pd\nROOT=Path.cwd() if (Path.cwd()/'evidence/summary.json').exists() else Path.cwd().parent\nevidence=json.loads((ROOT/'evidence/summary.json').read_text())\nprint('Evidence scope:', evidence['scope'])\n"
    )
    md(
        "## Reference evaluation\nThe two reference values below are recorded on the same 535-origin development evaluation. These are not live leaderboard scores.\n"
    )
    code(
        "f=go.Figure(go.Bar(x=['Stored historical means','Current reference model'],y=[0.217739151200211,0.3097087232124053]))\nf.update_layout(title='Recorded reference performance — 535 development origins',xaxis_title='Reference',yaxis_title='Mean daily Spearman / population standard deviation')\nf.show(renderer='plotly_mimetype')\n"
    )
    for scope, title in [
        ("First development period", "First-period feature comparisons"),
        ("Middle development period", "Middle-period feature comparisons"),
        ("Last development period", "Last-period feature comparisons"),
        ("Three development periods", "Pooled development comparisons"),
    ]:
        md(
            "## "
            + title
            + "\nAll panels from completed included reports are shown, including negative results. Panel identifiers are presentation labels, not public implementation specifications.\n"
        )
        code(
            "scope="
            + repr(scope)
            + "\nrows=[dict(study=s['study'],**r) for s in evidence['studies'] if s['evaluation_scope']==scope for r in s['comparisons']]\nif rows:\n    display(pd.DataFrame(rows))\n    f=go.Figure()\n    for name in dict.fromkeys(r['study'] for r in rows):\n        group=[r for r in rows if r['study']==name]\n        f.add_bar(x=[r['panel'] for r in group],y=[r['delta_vs_matched_control'] for r in group],name=name)\n    f.add_hline(y=0,line_dash='dot')\n    f.update_layout(title=scope+' — matched changes, not cross-period comparisons',xaxis_title='Declared panel',yaxis_title='Metric change versus its own saved control')\n    f.show(renderer='plotly_mimetype')\nelse:\n    print('No completed evidence available for this evaluation scope.')\n"
        )
    md(
        "## Scope and limitations\nRepeated development-set inspection makes this exploratory research. Figures do not establish causal feature importance or correct the full adaptive search. The final evaluation remains gated. Save this notebook after running; reopen without execution to confirm inline outputs persist.\n"
    )
    return dict(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        nbformat=4,
        nbformat_minor=5,
    )


def public_files(summary):
    README = """# Commodity Return Forecasting

A feature-first investigation of short-horizon, multi-market return ranking.

## Research question
Can domain-informed, point-in-time representations improve the ranking of 424 return targets without contaminating temporal evaluation?

## Evidence at a glance

| Reference | Recorded development metric | Evaluation |
|---|---:|---|
| Stored historical target means | 0.217739 | 535 development origins |
| Current reference model | 0.309709 | Same 535 origins |

These are **development scores, not Kaggle leaderboard scores**. The historical-dynamics screen produced a small middle-period improvement that required temporal replication. Its replication status is recorded below rather than assumed.

## Read the research

- [Aggregate evidence](evidence/summary.json) includes all panels from the included completed comparisons, not only the best results.
- [Research notebook](notebooks/00_research_review.ipynb) provides scope-separated Plotly comparisons. Run it in JupyterLab, save it, and reopen it to retain interactive outputs. GitHub's static notebook view may not display interactive JavaScript.
- [Methodology](docs/methodology.md) explains temporal splits, feature timing, matched comparisons and checkpoint controls.
- [Limitations](docs/limitations.md) separates exploratory validation, final assessment and competitive claims.

## Engineering approach
Immutable source data; explicit availability boundaries; fixed control models; feature-family additions and removals; saved predictions; checksum manifests; bounded, checkpointed computation; and failures documented separately from negative predictive results.

## Implementation access
This is a **public case study**, not an open-source reproduction kit. Full feature code, experiment configurations, fitted models, outcome-derived arrays and private execution records are maintained separately. The only public code is aggregate-data presentation code. Any public material can be copied; withholding the research implementation is the access-control boundary.

## Current evidence status
"""
    for s in summary["studies"]:
        README += "\n- " + s["study"] + ": " + s["status"] + " (" + s["evaluation_scope"] + ")."
    README += "\n\nPrepared from owner-provided execution records. No independent third-party verification, completed final-test evaluation, or leaderboard-winning result is asserted.\n"
    methodology = """# Methodology and evidence boundaries

The prediction outputs are return targets across heterogeneous markets. This project treats each proposed representation as a hypothesis and compares it with a saved model using the same evaluation origins and metric.

The official-style development calculation is the mean of daily cross-sectional Spearman correlations divided by their population standard deviation. It is not annualized, and pooled performance is not an average of fold ratios.

Temporal data boundaries and horizon-specific outcome-release delays govern feature availability. Data-dependent transformations are fitted within training or updated sequentially using only previously available information. The stored model controls are replayed instead of retrained for each comparison.

Feature-family addition, removal, representation controls and temporal replication provide the primary evidence. Permutation and importance diagnostics are not causal proof. Conditional time-block intervals reflect dependence under the stated resampling scheme, but not the full repeated-development-search process.

The public evidence contains bounded scalar aggregates and source-report hashes only. Research source, raw data, targets, per-origin predictions, trained weights and infrastructure details are omitted intentionally. Public/private releases are linked by separate manifests; the repositories are not meant to contain identical files.
"""
    limitations = """# Limitations

- The development periods have been inspected repeatedly. A local gain is exploratory until it survives appropriate further evaluation.
- Public and private leaderboard results, development scores and final evaluation are different settings.
- The current official strongest comparable leaderboard result has not been freshly verified in this publication.
- Feature count is not an information count, and passing code checks is not evidence of improved prediction.
- A negative panel result does not falsify its entire underlying economic hypothesis.
- The public showcase does not provide full reproducibility by itself: proprietary implementation and data-dependent artifacts are private.
- No public-source license or visibility setting can recall copies already obtained by other people.
"""
    rights = """# Rights and disclosure

Original newly authored showcase text and presentation code: copyright 2026 Alvaro Mendizabal. All rights reserved except applicable law, platform terms, and any separately granted permission.

This notice does not override third-party notices or revoke permissions granted under an earlier license. Review the repository's actual historical licenses before changing distribution terms. No research implementation is licensed or disclosed by this showcase merely because it is discussed here.

Public GitHub repositories remain viewable and forkable under GitHub's terms. This notice is not a technical copying barrier. Obtain legal advice for any specific relicensing or enforcement decision.
"""
    workflow = """name: Portfolio evidence checks
on: [push, pull_request]
permissions:
  contents: read
jobs:
  evidence:
    runs-on: ubuntu-latest
    timeout-minutes: 3
    steps:
      - uses: actions/checkout@v4
      - name: Check aggregate evidence schema and notebook state
        run: |
          python - <<'PY_CHECK'
          import json, pathlib, subprocess
          allowed={'README.md','RIGHTS.md','.gitignore','docs/methodology.md','docs/limitations.md','evidence/summary.json','notebooks/00_research_review.ipynb','.github/workflows/portfolio.yml'}
          tracked=set(subprocess.check_output(['git','ls-files','-z']).decode().rstrip('\\0').split('\\0'))
          assert tracked == allowed, 'Unexpected public files; review disclosure boundary'
          r=json.loads(pathlib.Path('evidence/summary.json').read_text())
          assert r['schema']=='commodity-public-aggregates-v1'
          assert r['not_a_leaderboard_result'] is True
          for s in r['studies']:
              assert set(s)=={'study','evaluation_scope','status','comparisons','new_fits_in_this_report','report_sha256','provenance'}
              for x in s['comparisons']:
                  assert set(x)=={'panel','official_metric','delta_vs_matched_control'}
          n=json.loads(pathlib.Path('notebooks/00_research_review.ipynb').read_text())
          assert n['nbformat']==4
          assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
          print('Public evidence schema passed; this is not a private-model verification.')
          PY_CHECK
"""
    return {
        "README.md": README.encode(),
        "RIGHTS.md": rights.encode(),
        ".gitignore": b".ipynb_checkpoints/\n__pycache__/\n.DS_Store\n",
        "docs/methodology.md": methodology.encode(),
        "docs/limitations.md": limitations.encode(),
        "evidence/summary.json": (
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode(),
        "notebooks/00_research_review.ipynb": (
            json.dumps(public_notebook(), indent=1) + "\n"
        ).encode(),
        ".github/workflows/portfolio.yml": workflow.encode(),
    }


def prepare():
    if ROOT.is_symlink() or not ROOT.is_dir():
        raise Stop("Missing expected manual checkout.")
    if OUT.exists():
        raise Stop(
            "This release export already exists. Preserve it; do not overwrite reviewed publication bytes."
        )
    # Read-only Git inventory. No fetch, reset, branch change, staging or remote operations.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    if head != PIN:
        raise Stop("Research source revision changed; review before publication.")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout
    sources = []
    for folder in DIRS:
        base = ROOT / folder
        if not base.exists():
            continue
        for dirpath, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [
                d for d in dirs if d not in DENY_DIRS and not (Path(dirpath) / d).is_symlink()
            ]
            for name in files:
                path = Path(dirpath) / name
                if path.suffix in EXTENSIONS and not path.name.startswith("."):
                    sources.append(safe_file(path))
    sources += [safe_file(ROOT / name) for name in ROOT_FILES if (ROOT / name).exists()]
    for folder in (ROOT / "logs").glob("manual*"):
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in folder.glob("*.json"):
            if path.stat().st_size < 24 * 1024**2:
                sources.append(safe_file(path))
    sources = sorted(set(sources))
    if len(sources) > 5000 or sum(x.stat().st_size for x in sources) > 600 * 1024**2:
        raise Stop("Source snapshot exceeds bounded export size.")
    for source in sources:
        secret_check(source)
    studies = []
    for folder, file, label, scope in REPORTS:
        path = ROOT / "logs" / folder / file
        if path.is_file() and not path.is_symlink():
            studies.append(summarize(json.loads(path.read_text()), label, scope, digest(path)))
        else:
            studies.append(
                dict(
                    study=label,
                    evaluation_scope=scope,
                    status="Not supplied / not verified",
                    comparisons=[],
                    new_fits_in_this_report=None,
                    report_sha256=None,
                    provenance="No complete report found.",
                )
            )
    summary = dict(
        schema="commodity-public-aggregates-v1",
        scope="Exploratory temporal development research",
        not_a_leaderboard_result=True,
        strongest_official_score_verified=False,
        raw_data_included=False,
        research_implementation_included=False,
        studies=studies,
    )
    public = public_files(summary)
    if set(public) != PUBLIC_FILES:
        raise Stop("Public allowlist changed.")
    OUT.mkdir(parents=True)
    private = OUT / "private_source"
    show = OUT / "public_showcase"
    hashes = {}
    for source in sources:
        rel = source.relative_to(ROOT)
        dest = private / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if digest(source) != digest(dest):
            raise Stop("Private snapshot copy mismatch.")
        hashes[str(rel)] = digest(dest)
    for rel, body in public.items():
        dest = show / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    (OUT / "private_paths.nul").write_bytes(b"\0".join(x.encode() for x in hashes) + b"\0")
    (OUT / "public_paths.nul").write_bytes(b"\0".join(x.encode() for x in sorted(public)) + b"\0")
    receipt = dict(
        status="LOCAL_PUBLICATION_CANDIDATE",
        source_commit=head,
        private_files=hashes,
        public_files={r: digest(show / r) for r in public},
        secret_scan="Basic patterns only; human review required",
        git_status_at_export=status,
        private_snapshot_scope="Active manual checkout source and small manual receipts; not a raw/model/.venv backup and not every older worktree.",
        generated_utc=datetime.now(UTC).isoformat(),
        tests_executed=0,
        git_writes=0,
        account_operations=0,
    )
    (OUT / "publication_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    for folder, name in [(private, "private_source.zip"), (show, "public_showcase.zip")]:
        with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for f in folder.rglob("*"):
                if f.is_file():
                    z.write(f, str(f.relative_to(folder)))
        with zipfile.ZipFile(OUT / name) as z:
            for rel in z.namelist():
                if hashlib.sha256(z.read(rel)).hexdigest() != digest(folder / rel):
                    raise Stop("Publication archive verification failed.")
    print("RESULT: LOCAL_PUBLICATION_CANDIDATE")
    print("DIRECTORY:", OUT)
    print(
        "No GitHub changes. Review every public file and the private/source boundary before any commit."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()

    def expiry(*_):
        raise Stop("180-second export deadline. Partial local export preserved; no blind retry.")

    signal.signal(signal.SIGALRM, expiry)
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        prepare()
    except BaseException as e:
        print("RESULT: STOPPED")
        print("ERROR:", type(e).__name__ + ": " + str(e))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
