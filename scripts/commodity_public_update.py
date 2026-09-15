#!/usr/bin/env python3
"""Prepare a reviewed update of the EXISTING PUBLIC commodity-prediction repo.

User-run only. No model loading/fitting, notebook execution, dependency install,
remote request, commit, push, merge, deletion, rename or visibility operation.
Local Git reads are used to compare the pinned checkout with an explicit overlay.
--apply writes only a separate existing review worktree, never the research tree.
--stage explicitly stages only the reviewed manifest there. No automatic upload.
This is prepared source; its tests have not been run by its authoring assistant.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import io
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-manual")
REVIEW = Path("/home/sagemaker-user/projects/commodity-prediction-review")
STORE = Path.home() / "commodity-public-update"
PIN = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
REPO = "alvaromendizabal/commodity-prediction"
MAX_FILE = 25 * 1024**2
MAX_TOTAL = 250 * 1024**2
PLOT = "application/vnd.plotly.v1+json"
SOURCE_DIRS = {"src", "scripts", "tests", "configs", "docs", "notebooks", ".github"}
ROOT_FILES = {
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
    "COPYING",
    "requirements.txt",
    "Makefile",
}
TEXT_SUFFIXES = {
    ".py",
    ".ipynb",
    ".json",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
    ".sql",
    ".lock",
}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "data",
    "artifacts",
    "logs",
    "recovery",
    "exports",
}
DONE = {
    "DEPENDENCIES_READY",
    "NOTEBOOK_AND_FEATURE_AUDIT_READY",
    "NOTEBOOK_AND_FIRST_FOLD_REVIEW_READY",
    "NOTEBOOK_AND_VALIDATION_REVIEW_READY",
    "NOTEBOOK_AND_SESSION_FEATURES_READY",
    "NOTEBOOK_AND_SESSION_ABLATION_READY",
    "NOTEBOOK_AND_CLOSE_NETWORK_READY",
    "NOTEBOOK_AND_DIAGNOSIS_READY",
    "NOTEBOOK_AND_RANK_STATE_READY",
    "NOTEBOOK_AND_TARGET_CONTEXT_READY",
    "NOTEBOOK_AND_RESPONSE_READY",
    "NOTEBOOK_AND_INFORMATION_AUDIT_READY",
    "NOTEBOOK_AND_FEATURE_ROUND_READY",
    "ROUND_COMPLETE",
    "NOTEBOOK_COMPLETE",
}
STUDIES = [
    (3, "readiness", "commodity_manual_readiness.json", "Runtime and feature readiness", 0),
    (4, "first_fold", "commodity_first_fold_report.json", "Normalization first-period screen", 180),
    (5, "validation", "commodity_validation_report.json", "Normalization temporal validation", 535),
    (
        6,
        "session_features",
        "commodity_session_features_report.json",
        "Session candidate laboratory",
        0,
    ),
    (
        7,
        "session_ablation",
        "commodity_session_ablation_report.json",
        "Session feature ablations",
        180,
    ),
    (8, "close_network", "commodity_close_network_report.json", "Cross-asset peer features", 180),
    (
        9,
        "feature_diagnosis",
        "commodity_feature_diagnosis_report.json",
        "Network removal and rank laboratory",
        180,
    ),
    (10, "rank_state", "commodity_rank_state_report.json", "Released historical-rank states", 180),
    (11, "target_context", "commodity_target_context_report.json", "Instrument context", 180),
    (
        12,
        "response_encoding",
        "commodity_response_encoding_report.json",
        "Delayed response encoding",
        180,
    ),
    (
        13,
        "information_audit",
        "commodity_information_audit_report.json",
        "Saved-model information audit",
        535,
    ),
    (
        14,
        "innovation_ablation",
        "commodity_innovation_ablation_report.json",
        "Ordinary and robust innovations",
        180,
    ),
    (15, "prior_dynamics", "commodity_prior_dynamics_report.json", "Prior dynamics screening", 180),
    (
        16,
        "prior_replication",
        "commodity_prior_replication_report.json",
        "Prior dynamics temporal replication",
        535,
    ),
    (17, "event_history", "commodity_event_history_report.json", "Observed-event histories", 180),
    (
        18,
        "sequence_state",
        "commodity_sequence_state_report.json",
        "Released sequence structure",
        175,
    ),
    (
        19,
        "prior_error_memory",
        "commodity_prior_error_memory_report.json",
        "As-of prior-error memory",
        175,
    ),
]
# Patterns identify obvious credentials, not every possible secret. Matches are
# never printed. Human review remains mandatory, including Plotly payloads.
SECRET_PATTERNS = [
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r'(?i)(?:X-Amz-Signature|X-Amz-Credential)=[^&\s"<>]{12,}'),
    re.compile(r"(?i)https://[^/\s:@]+:[^/\s@]+@"),
]


class Stop(RuntimeError):
    """A review gate. Preserve state rather than bypassing it."""


def utc():
    return datetime.now(UTC).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe(root, rel):
    rel = str(rel)
    p = PurePosixPath(rel)
    if (
        not p.parts
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in rel
        or any(ord(c) < 32 for c in rel)
    ):
        raise Stop("Unsafe relative path.")
    out = Path(root)
    if out.is_symlink():
        raise Stop("Symlinked destination root.")
    for item in p.parts:
        out /= item
        if out.is_symlink():
            raise Stop("Symlinked path; review manually: " + rel)
    return out


def read(path, limit=MAX_FILE):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Stop("Expected a bounded regular file: " + str(path))
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise Stop("File changed during the snapshot: " + str(path))
    return data


def atomic(path, data):
    path = Path(path)
    safe(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def bounded(seconds):
    def alarm(*_):
        raise Stop("Publication deadline reached; preserve the written plan and logs.")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def git(root, *args, optional=False):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, env=env, timeout=25, check=False
    )
    if r.returncode:
        if optional:
            return None
        # Do not echo stderr, which could contain a credential-bearing URL.
        raise Stop("Local Git command failed: " + " ".join(args[:2]))
    return r.stdout


def assert_repo(root):
    url = git(root, "remote", "get-url", "origin").decode().strip()
    allowed = {
        "https://github.com/" + REPO,
        "https://github.com/" + REPO + ".git",
        "git@github.com:" + REPO + ".git",
        "ssh://git@github.com/" + REPO + ".git",
    }
    if url not in allowed:
        raise Stop(
            "Origin does not name the existing commodity-prediction repository. Do not rename it or change visibility."
        )


def scan(data, path):
    text = data.decode("utf-8")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise Stop("Possible credential detected; no upload is permitted: " + str(path))
    return text


def eligible(rel):
    p = PurePosixPath(rel)
    if any(x in EXCLUDED_PARTS for x in p.parts):
        return False
    if p.name.startswith(".env") or p.name in {"kaggle.json", "credentials", "token.json"}:
        return False
    if rel in ROOT_FILES:
        return True
    return len(p.parts) > 1 and p.parts[0] in SOURCE_DIRS and p.suffix in TEXT_SUFFIXES


def nb_text(x):
    return "".join(x) if isinstance(x, list) else str(x or "")


def sanitize_notebook(body):
    """Keep code/markdown and inline Plotly; remove console/HTML tables/errors.

    The original working notebook is never edited. A disclosure cell explicitly
    records removals and the original error count. A failed notebook stays failed.
    Plotly figure values remain public; review them before publishing.
    """
    nb = json.loads(body)
    if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
        raise Stop("Unsupported notebook structure.")
    counts = dict(
        code_cells=0,
        executed_code_cells=0,
        original_error_outputs=0,
        original_plotly_outputs=0,
        kept_plotly_outputs=0,
        removed_outputs=0,
    )
    for c in nb["cells"]:
        c["metadata"] = {
            k: v for k, v in c.get("metadata", {}).items() if k in {"tags", "slideshow"}
        }
        c.pop("attachments", None)
        if c.get("cell_type") != "code":
            if c.get("cell_type") == "markdown":
                c["source"] = nb_text(c.get("source")).replace(
                    "Private feature research.", "Feature research."
                )
            continue
        counts["code_cells"] += 1
        counts["executed_code_cells"] += c.get("execution_count") is not None
        kept = []
        for o in c.get("outputs", []):
            counts["original_error_outputs"] += o.get("output_type") == "error"
            data = o.get("data", {})
            if PLOT in data:
                counts["original_plotly_outputs"] += 1
                cleaned = {PLOT: data[PLOT]}
                # MIME data remain inline. No HTML tables, widgets or stdout.
                obj = {"output_type": "display_data", "metadata": {}, "data": cleaned}
                kept.append(obj)
                counts["kept_plotly_outputs"] += 1
            else:
                counts["removed_outputs"] += 1
        c["outputs"] = kept
    note = (
        "## Publication provenance\n\nThis is a publication copy, not a new execution. "
        f"Original code cells: {counts['code_cells']}; cells with execution counts: {counts['executed_code_cells']}; "
        f"original error outputs: {counts['original_error_outputs']}; inline Plotly outputs retained: {counts['kept_plotly_outputs']}. "
        "Console logs, table/HTML output, attachments and tracebacks were omitted from this copy. "
        "The original remains in the research workspace. An execution count alone is not proof of success. "
        "Refer to the linked result ledger for completed/failed/pending study status. "
        "Interactive figure data are disclosed with this public notebook.\n"
    )
    nb["cells"].insert(
        1, {"cell_type": "markdown", "id": "publication-provenance", "metadata": {}, "source": note}
    )
    nb["metadata"] = {
        k: v for k, v in nb.get("metadata", {}).items() if k in {"kernelspec", "language_info"}
    }
    nb["metadata"]["publication_copy"] = counts
    return encode(nb), counts


def number(v):
    return (
        v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    )


def label(v, fallback="unknown"):
    return (
        v if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_ .:+/()=-]{1,140}", v) else fallback
    )


def scalar_study(body, number_id, title, origins):
    r = json.loads(body)
    if not isinstance(r, dict):
        raise Stop("Research report must be a JSON object.")
    state = label(r.get("status"))
    complete = state in DONE
    out = dict(
        notebook=number_id,
        title=title,
        status=state,
        completed=complete,
        evaluation_origins=origins,
        report_sha256=sha(body),
        comparisons=[],
    )
    for k in (
        "new_training_fits",
        "control_refits",
        "maximum_prediction_replay_error",
        "cumulative_supervised_seconds",
        "supervised_seconds",
        "elapsed_seconds",
        "plotly_figures",
    ):
        if number(r.get(k)) is not None:
            out[k] = r[k]
    for k in ("source_commit", "lineage"):
        if isinstance(r.get(k), str) and re.fullmatch("[0-9a-f]{40,64}", r[k]):
            out[k] = r[k]
    out["decision"] = label(r.get("decision"), "not recorded")
    if complete:
        for row in r.get("comparison_rows", []):
            score = number(row.get("official_metric"))
            if score is None:
                continue
            one = {"variant": label(row.get("variant")), "official_metric": score}
            for name in (
                "delta_vs_current_market",
                "matched_delta",
                "positive_folds",
                "passes_review_gate",
                "all_inputs_admitted",
                "all_intended_inputs_admitted",
            ):
                v = row.get(name)
                if isinstance(v, bool) or number(v) is not None:
                    one[name] = v
            out["comparisons"].append(one)
    # No daily series, target arrays, feature matrices, infrastructure or raw logs.
    return out


def studies(root):
    result = []
    fingerprints = {}
    for n, slug, filename, title, origins in STUDIES:
        rel = "logs/manual_" + slug + "/" + filename
        path = safe(root, rel)
        if not path.exists():
            result.append(
                dict(
                    notebook=n,
                    title=title,
                    status="REPORT_NOT_AVAILABLE",
                    completed=False,
                    evaluation_origins=origins,
                    comparisons=[],
                )
            )
            continue
        raw = read(path)
        fingerprints[rel] = sha(raw)
        entry = scalar_study(raw, n, title, origins)
        entry["evidence_path"] = f"reports/manual_research/study_{n:02d}.json"
        result.append(entry)
    return result, fingerprints


def make_readme(records):
    table = ["| Notebook | Investigation | Evidence status | Origins |", "|---|---|---|---:|"]
    for r in records:
        table.append(
            f"| {r['notebook']:02d} | {r['title']} | {r['status']} | {r['evaluation_origins']} |"
        )
    return (
        """# Commodity Prediction Research

A reproducible, feature-first investigation of short-horizon, multi-asset return ranking.

**Start here:** [research overview notebook](notebooks/portfolio_overview.ipynb) ·
[experiment ledger](docs/MANUAL_RESEARCH_RESULTS.md) ·
[reproduction guide](docs/MANUAL_REPRODUCTION.md) ·
[aggregate evidence](reports/manual_research/index.json)

## Research question

Which representations of market observations and legally released target history improve
short-horizon cross-sectional forecasts, and do those improvements transfer across time?
The project studies target/pair structure, market behavior, released priors, temporal
representations and controlled feature-family additions/removals. Negative findings are
retained rather than hidden.

## Evaluation boundary

The established `current_market` reference scores **0.309709 over 535 development origins**.
Its three period scores are **0.403381**, **0.167042**, and **0.390619**, respectively.
This is a historical local development reference, **not a Kaggle leaderboard score or a
claim that no later contender has a higher point estimate**. Consult the period-specific
ledger below for newer comparisons. Do not rank runs evaluated on different dates together.

The metric is the mean daily cross-sectional Spearman correlation divided by its population
standard deviation, without annualization. Feature values follow prediction-time availability;
label-derived features respect horizon-specific release delays. The final evaluation remains
separate from repeated exploratory development. A positive point estimate is not automatic
model promotion or proof of a competition record.

![Historical development reference](reports/manual_research/development_reference.svg)

GitHub displays notebooks statically; use JupyterLab or a compatible notebook
viewer for interactive Plotly controls. The SVG preview above needs no JavaScript.

## What is public

This repository includes research implementation, tests, declared configurations, and
notebooks with inline Plotly output when execution evidence exists. Raw/derived dataset
files, fitted weights, credentials, environments and infrastructure logs are not distributed
here. There is **one public repository**: no separate private code repository is required.
Existing license and third-party notices are preserved.

## Experiment index

"""
        + "\n".join(table)
        + """

`REPORT_NOT_AVAILABLE` means no matching report was available in the active workspace at
export time; it does not mean the experiment did not occur. Pending or failed work is never
listed as a completed score. Export reads saved evidence without fitting models or executing
research notebooks. See [disclosure notes](docs/PUBLICATION_SCOPE.md).

## Engineering and reproducibility

Experiments use bounded workers, training-partition preprocessing, explicit stage manifests,
checkpoint checksums, saved-prediction replay and per-study fit/time accounting. These are
reported safeguards, not a substitute for independent reproduction. Prepared tests are not
called passing tests without their execution receipts. Inline figures remain in notebooks;
self-contained HTML is supplemental. Reproduction is manual and requires licensed data.

Earlier source, methodological notes and historical publication evidence remain in this
repository. The original README is retained under `docs/history/` when this update is first
prepared. No Git history rewrite, repository rename or visibility change is part of this release.

**Current phase: feature research remains open.** The goal is the strongest valid comparable
performance, not an unsupported leaderboard claim.
"""
    ).encode()


def portfolio_notebook():
    cells = []

    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text))

    def code(text):
        cells.append(
            dict(cell_type="code", metadata={}, source=text, execution_count=None, outputs=[])
        )

    md(
        "# Commodity prediction · evidence review\n\nPresentation of saved aggregate results only. "
        "No model loading, fitting, raw-data access, or research-notebook execution. "
        "Run this notebook in the **review worktree** after the publication overlay is applied. "
        "Do not compare scores from different evaluation periods. Save and reopen to confirm inline outputs persist."
    )
    code(
        "from pathlib import Path\nimport json\nimport pandas as pd\nimport plotly.graph_objects as go\nimport plotly.io as pio\nfrom IPython.display import display\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nindex=json.loads((ROOT/'reports/manual_research/index.json').read_text())\nstudies=index['studies']\npio.renderers.default='plotly_mimetype'\nrows=[dict(notebook=s['notebook'],title=s['title'],origins=s['evaluation_origins'],**r) for s in studies if s['completed'] for r in s['comparisons']]\ndf=pd.DataFrame(rows)\ndisplay(pd.DataFrame([{k:s[k] for k in ('notebook','title','status','evaluation_origins')} for s in studies]))\ndef show(fig,title,x,y):\n    fig.update_layout(title=title,xaxis_title=x,yaxis_title=y,height=560,margin=dict(l=80,r=30,t=90,b=130))\n    fig.show()\n"
    )
    md(
        "## Historical reference across periods\nThe same fixed comparison, on three different chronological intervals. Its variability motivates replication."
    )
    code(
        "f=go.Figure(go.Bar(x=['1169–1348','1349–1528','1529–1703'],y=[.40338108742296147,.16704165319061334,.39061886486364056]))\nshow(f,'1 · Historical reference by period','Development origins','Daily-correlation mean / population standard deviation')"
    )
    md(
        "## Matched improvements\nCompare deltas within the evaluation used by each study. This chart does not pool incompatible predictions."
    )
    code(
        "f=go.Figure()\nif not df.empty:\n    for n,g in df.groupby('notebook',sort=True):\n        delta=g.get('delta_vs_current_market',g.get('matched_delta',pd.Series(index=g.index,dtype=float)))\n        if 'matched_delta' in g: delta=delta.fillna(g['matched_delta'])\n        f.add_bar(x=[f'{n:02d}: '+v for v in g['variant']],y=delta,name=f'Notebook {n:02d}')\nf.add_hline(y=0)\nshow(f,'2 · Matched development changes — different periods remain separate','Experiment / panel','Within-study metric difference')"
    )
    md(
        "## Candidate replication\nThis section uses report 16 only. An absent report produces an explicit gap rather than a fabricated result."
    )
    code(
        "r16=next(s for s in studies if s['notebook']==16)\nf=go.Figure()\nif r16['completed']:\n    f.add_bar(x=[r['variant'] for r in r16['comparisons']],y=[r['official_metric'] for r in r16['comparisons']])\n    f.add_hline(y=.3097087232124053,line_dash='dash')\nelse:\n    f.add_annotation(text='Completed replication report not available in this release',showarrow=False)\nshow(f,'3 · Prior-dynamics replication, 535 origins only','Declared representation','Pooled development metric')"
    )
    md(
        "## Execution accounting\nFit counts describe the reported experiment cost; more fits are not evidence of better science."
    )
    code(
        "r=[s for s in studies if s['completed'] and 'new_training_fits' in s]\nf=go.Figure(go.Bar(x=[str(s['notebook']) for s in r],y=[s['new_training_fits'] for s in r]))\nshow(f,'4 · Reported new fits by milestone','Notebook','Reported new fits')"
    )
    md(
        "## Bounded analytical time\nThese are saved report timings, not an estimate of all cloud charges or all work ever performed."
    )
    code(
        "r=[(s,next((s[k] for k in ('cumulative_supervised_seconds','supervised_seconds','elapsed_seconds') if k in s),None)) for s in studies if s['completed']]\nr=[(s,t) for s,t in r if t is not None]\nf=go.Figure(go.Bar(x=[str(s['notebook']) for s,t in r],y=[t for s,t in r]))\nshow(f,'5 · Recorded analytical/supervised time','Notebook','Seconds (report timing scope)')"
    )
    md(
        "## Completion and evidence gaps\nOnly available reports supply completion status. An unexecuted template is not an executed result."
    )
    code(
        "counts=pd.Series([s['status'] for s in studies]).value_counts()\nf=go.Figure(go.Bar(x=list(counts.index),y=counts.tolist()))\nshow(f,'6 · Evidence status across the manual research sequence','Recorded status','Number of milestones')\nprint('RESULT: PORTFOLIO_OVERVIEW_READY')\nprint('Save this notebook, reopen without running, and verify all six figures are visible.')"
    )
    return encode(
        dict(
            nbformat=4,
            nbformat_minor=5,
            metadata={
                "kernelspec": {
                    "display_name": "Commodity - manual (verified)",
                    "language": "python",
                    "name": "commodity-manual",
                }
            },
            cells=[dict(id=f"portfolio-{i:02d}", **c) for i, c in enumerate(cells)],
        )
    )


def reference_svg():
    """Static GitHub preview of the documented reference, not a new experiment."""
    values = [0.40338108742296147, 0.16704165319061334, 0.39061886486364056]
    labels = ["1169–1348", "1349–1528", "1529–1703"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="300" viewBox="0 0 960 300" role="img" aria-label="Historical development reference by period">',
        '<rect width="960" height="300" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033">',
        '<text x="28" y="37" font-size="24">Historical development reference</text>',
        '<text x="28" y="63" font-size="14">Daily Spearman mean / population standard deviation · not a leaderboard score</text>',
    ]
    for i, (v, label_) in enumerate(zip(values, labels, strict=False)):
        y = 94 + i * 56
        parts.append(f'<text x="28" y="{y + 22}" font-size="17">{html.escape(label_)}</text>')
        parts.append(
            f'<rect x="165" y="{y}" width="{v / 0.5 * 600:.3f}" height="30" fill="#285D7A"/>'
        )
        parts.append(
            f'<text x="{175 + v / 0.5 * 600:.3f}" y="{y + 22}" font-size="16">{v:.6f}</text>'
        )
    parts.extend(
        [
            '<text x="28" y="283" font-size="13">Pooled reference over 535 origins: 0.309709. Separate contender results remain in the experiment ledger.</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts).encode()


def current_release():
    p = safe(STORE, "current.json")
    r = json.loads(read(p))
    rel = r["release_directory"]
    if not re.fullmatch(r"release-[0-9TZ-]+", rel):
        raise Stop("Invalid publication directory reference.")
    return safe(STORE, rel)


def prepared_gate():
    p = safe(STORE, "self_test.json")
    r = json.loads(read(p))
    if r.get("status") != "TESTS_PASSED" or r.get("helper_sha256") != sha(read(Path(__file__))):
        raise Stop("Run this helper with --self-test before preparing publication.")


def prepare(root):
    prepared_gate()
    root = Path(root)
    assert_repo(root)
    if git(root, "rev-parse", "HEAD").decode().strip() != PIN:
        raise Stop("Active research source pin differs; preserve local work and return the status.")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = safe(STORE, "release-" + stamp)
    dest.mkdir(parents=True, exist_ok=False)
    tracked = git(root, "ls-files", "-z").decode().split("\0")
    names = set(x for x in tracked if x)
    for top in SOURCE_DIRS:
        start = safe(root, top)
        if not start.exists():
            continue
        for path, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if d not in EXCLUDED_PARTS and not Path(path, d).is_symlink()
            )
            for name in files:
                names.add(Path(path, name).relative_to(root).as_posix())
    records, source_hashes = studies(root)
    entries = []
    omitted = []
    files = {}
    notebook_changes = []
    total = 0
    for rel in sorted(names):
        if not eligible(rel):
            continue
        if rel == "docs/PRIVATE_RESEARCH_PUBLIC_PORTFOLIO.md":
            omitted.append(
                dict(
                    path=rel,
                    reason="Superseded private/public split instructions; preserved in the research workspace, not included as current guidance.",
                )
            )
            continue
        path = safe(root, rel)
        if path.suffix == ".ipynb" and re.search(r"\(\d+\)", path.name):
            omitted.append(
                dict(
                    path=rel,
                    reason="Duplicate download-style notebook name; canonical notebook retained.",
                )
            )
            continue
        if not path.exists():
            omitted.append(
                dict(path=rel, reason="Deleted locally; automatic deletion is not authorized.")
            )
            continue
        body = read(path)
        total += len(body)
        if total > MAX_TOTAL or len(names) > 5000:
            raise Stop("Snapshot exceeds bounded publication size.")
        before = git(root, "show", PIN + ":" + rel, optional=True)
        if before == body:
            continue
        # Outputs and machine reports are handled by the aggregate ledger; no
        # arbitrary ignored logs/reports or data-like arrays are copied.
        if rel.endswith(".ipynb"):
            body, change = sanitize_notebook(body)
            notebook_changes.append(dict(path=rel, **change))
        text = scan(body, rel)
        if rel.endswith(".py"):
            ast.parse(text, filename=rel)  # static syntax only; never import source
        if rel.endswith(".json"):
            json.loads(text)
        files[rel] = body
        entries.append(
            dict(
                path=rel,
                source_sha256=sha(read(path)),
                base_sha256=None if before is None else sha(before),
                published_sha256=sha(body),
                bytes=len(body),
                kind="source_update",
            )
        )
        source_hashes[rel] = sha(read(path))
    root_readme = safe(root, "README.md")
    generated = {
        "README.md": make_readme(records),
        "AGENTS.md": b"""# Research operating rules\n\nThis is the existing public commodity-prediction repository. Keep its name and\nvisibility unchanged. The owner executes all project code, tests, notebooks,\ncloud operations, Git commits/pushes/merges and submissions. An assistant may\nresearch public sources, inspect supplied results, prepare implementations/tests\nand explain instructions; it must not operate project accounts or execute\nproject work without new explicit authorization.\n\nKeep experiments bounded, feature-first, leakage-safe, reproducible and\ncheckpointed. Never replace failed evidence with an invented pass. Preserve\nraw data, environments, historical source lineages and user changes. Compare\nscores only on matched evaluation scopes. Source and notebook outputs are\npublic after review; do not publish credentials, restricted data or model\ncheckpoints. See docs/PUBLICATION_SCOPE.md and docs/MANUAL_REPRODUCTION.md.\n\nDo not silently reformat fingerprinted research modules or weaken existing\nquality/publication gates. The active execution pin is not a publication branch.\nPreserve earlier instructions as historical evidence, not as an override of\nthese current manual-execution and public-repository rules.\n""",
        "reports/manual_research/index.json": encode(
            dict(
                repository=REPO,
                created_utc=utc(),
                research_source_pin=PIN,
                studies=records,
                evidence_scope="Saved user-run reports; exporter does not replay checkpoints or retrain. No leaderboard equivalence.",
            )
        ),
        "notebooks/portfolio_overview.ipynb": portfolio_notebook(),
        "reports/manual_research/development_reference.svg": reference_svg(),
        "docs/MANUAL_REPRODUCTION.md": b"""# Manual reproduction\n\nUse the pinned project environment and legally acquired Kaggle data.\nThe manual study notebooks orchestrate scripts/commodity_*.py and retain their\nindividual source/data/configuration lineage checks. Do not run every research\nnotebook simply to view this repository. Historical results are available in\nreports/manual_research and existing notebook outputs.\n\nFor rounds 18 and 19, first run the installed command:\n\n```bash\ncd "$HOME/projects/commodity-prediction-manual"\n"$HOME/projects/commodity-prediction-current/.venv/bin/python" -u \\\n  scripts/commodity_next_research.py --preflight\n```\n\nRequire NEXT_PREFLIGHT_PASSED, then run the corresponding notebook in the\nverified kernel. The preflight is not the older rounds-16/17 preflight.\nA missing preflight is a stopped dependency gate, not a model result.\nFresh external reproduction requires the preceding source/data/checkpoints;\nno claim is made that cloning alone reproduces historical private artifacts.\nThe final holdout must remain separate; repeated development is exploratory.\n""",
        "docs/PUBLICATION_SCOPE.md": b"""# Public repository scope\n\nThis update keeps the same public repository and publishes research code, tests,\nconfigurations and notebooks, not just a showcase hiding the implementation.\nThe export covers the active manual checkout, not every historical workspace.\nRaw and derived datasets, model weights, credentials, environments and machine\nlogs are excluded; this does not require a private GitHub repository.\n\nNotebook publication copies retain code, markdown and inline Plotly outputs.\nConsole output, HTML tables, image/other outputs, attachments and tracebacks are\nomitted, and a visible provenance note records the original error count.\nOriginal working notebooks are not changed. Retained Plotly payloads must also\nbe reviewed for data rights and disclosure. Automated scanning is incomplete.\n\nPrepared, failed and completed studies are distinguished using saved receipts.\nChecksums establish file identity, not the correctness of scientific claims.\nNo research code is reformatted, no lineage pin is silently relaxed, no old\nworkflow is removed and no license is replaced. Existing remote material not\nin this overlay is preserved. A normal update does not erase earlier history.\nThe owner must review all staged source and outputs before public upload.\n""",
    }
    if root_readme.exists():
        generated["docs/history/README_before_manual_publication.md"] = read(root_readme)
        source_hashes["README.md"] = sha(read(root_readme))
    agents = safe(root, "AGENTS.md")
    if agents.exists():
        generated["docs/history/AGENTS_before_manual_publication.md"] = read(agents)
        source_hashes["AGENTS.md"] = sha(read(agents))
    for r in records:
        if "evidence_path" in r:
            generated[r["evidence_path"]] = encode(r)
    ledger = [
        "# Manual research results",
        "",
        "Scores remain grouped by their own evaluation period; no cross-period ranking is implied.",
        "",
    ]
    for r in records:
        ledger += [
            f"## {r['notebook']:02d} — {r['title']}",
            f"Status: **{r['status']}**. Evaluation origins: **{r['evaluation_origins']}**.",
            "",
        ]
        if r["comparisons"]:
            ledger += ["| Representation | Metric | Matched baseline change |", "|---|---:|---:|"]
            for z in r["comparisons"]:
                delta = z.get("delta_vs_current_market", z.get("matched_delta"))
                ledger.append(
                    f"| {z['variant']} | {z['official_metric']:.6f} | {delta:+.6f} |"
                    if delta is not None
                    else f"| {z['variant']} | {z['official_metric']:.6f} | Not reported |"
                )
            ledger.append("")
        ledger += [
            f"Decision: `{r.get('decision', 'not available')}`. Source report SHA-256: `{r.get('report_sha256', 'not available')}`.",
            "",
        ]
    generated["docs/MANUAL_RESEARCH_RESULTS.md"] = ("\n".join(ledger) + "\n").encode()
    generated["scripts/commodity_public_update.py"] = read(Path(__file__))
    for rel, body in generated.items():
        scan(body, rel)
        files[rel] = body
        entries = [e for e in entries if e["path"] != rel]
        base = git(root, "show", PIN + ":" + rel, optional=True)
        entries.append(
            dict(
                path=rel,
                source_sha256=None,
                base_sha256=None if base is None else sha(base),
                published_sha256=sha(body),
                bytes=len(body),
                kind="generated_publication",
            )
        )
    for rel, body in files.items():
        atomic(safe(dest / "overlay", rel), body)
    for rel, expected in source_hashes.items():
        if sha(read(safe(root, rel))) != expected:
            raise Stop("Source/report changed during publication preparation: " + rel)
    manifest = dict(
        status="PUBLIC_UPDATE_CANDIDATE",
        created_utc=utc(),
        repository=REPO,
        source_root=str(root),
        source_pin=PIN,
        remote_checked=False,
        entries=sorted(entries, key=lambda x: x["path"]),
        source_hashes=source_hashes,
        notebook_disclosure_changes=notebook_changes,
        local_deletions_not_applied=omitted,
        study16_report_present=any(s["notebook"] == 16 and "report_sha256" in s for s in records),
        full_active_worktree_mirror=False,
        tests_executed_by_preparation=False,
        limits={"seconds": 180, "single_file_bytes": MAX_FILE, "source_bytes": MAX_TOTAL},
        caveats=[
            "All modified/new eligible source files from active checkout; not every older working copy.",
            "Retained figure payloads and existing public history still require human disclosure review.",
            "Existing checks/workflows/licenses are preserved. Their success is not asserted.",
            "Research checkout remains pinned; post-merge synchronization verified separately.",
        ],
    )
    atomic(dest / "manifest.json", encode(manifest))
    atomic(dest / "paths.nul", b"".join(e["path"].encode() + b"\0" for e in manifest["entries"]))
    atomic(dest / "review.txt", ("\n".join(e["path"] for e in manifest["entries"]) + "\n").encode())
    atomic(STORE / "current.json", encode(dict(release_directory=dest.name)))
    print("RESULT: PUBLIC_UPDATE_CANDIDATE")
    print("REVIEW:", dest / "review.txt")
    print("MANIFEST:", dest / "manifest.json")
    print("Open the overlay in JupyterLab and review every proposed public file before applying.")
    return manifest


def load_manifest():
    d = current_release()
    m = json.loads(read(d / "manifest.json"))
    if m.get("repository") != REPO or m.get("source_pin") != PIN:
        raise Stop("Manifest repository or source pin changed.")
    if len({e["path"] for e in m["entries"]}) != len(m["entries"]):
        raise Stop("Duplicate manifest paths.")
    for e in m["entries"]:
        rel = e["path"]
        approved_aggregate = str(rel).startswith("reports/manual_research/") and PurePosixPath(
            rel
        ).suffix in {".json", ".svg"}
        if not eligible(rel) and not approved_aggregate:
            raise Stop("Manifest contains a non-publication path: " + str(rel))
        body = read(safe(d / "overlay", rel))
        if sha(body) != e["published_sha256"] or len(body) != e["bytes"]:
            raise Stop("Publication overlay changed after preparation: " + e["path"])
    return d, m


def apply(worktree, ack):
    if not ack:
        raise Stop("Read the overlay and pass --reviewed-public-code explicitly.")
    d, m = load_manifest()
    w = Path(worktree)
    if w.resolve() == Path(m["source_root"]).resolve():
        raise Stop("Never apply the public overlay to the pinned research checkout.")
    assert_repo(w)
    branch = git(w, "branch", "--show-current").decode().strip()
    if branch != "chore/public-research-update-20260913":
        raise Stop("Apply only on the declared review branch, never main.")
    head = git(w, "rev-parse", "HEAD").decode().strip()
    if git(w, "merge-base", "--is-ancestor", PIN, "HEAD", optional=True) is None:
        raise Stop("Review branch does not descend from the research pin.")
    if git(w, "status", "--porcelain").strip():
        raise Stop(
            "Review worktree has local changes. Preserve and inspect them; no reset is performed."
        )
    # Three-way, path-by-path comparison detects remote edits after the pin.
    conflicts = []
    for e in m["entries"]:
        target = safe(w, e["path"])
        current = sha(read(target)) if target.exists() else None
        if current not in {e["base_sha256"], e["published_sha256"]}:
            conflicts.append(e["path"])
    if conflicts:
        atomic(d / "conflicts.json", encode(conflicts))
        raise Stop(
            "Remote/current files changed independently. Review conflicts.json before applying."
        )
    for rel, expected in m["source_hashes"].items():
        if sha(read(safe(m["source_root"], rel))) != expected:
            raise Stop(
                "Research source or evidence advanced after prepare. Make a new candidate before publication."
            )
    for e in m["entries"]:
        atomic(safe(w, e["path"]), read(safe(d / "overlay", e["path"])))
    atomic(
        d / "applied.json",
        encode(dict(worktree=str(w.resolve()), base_head=head, applied_utc=utc())),
    )
    print("RESULT: REVIEW_WORKTREE_UPDATED")
    print("No index staging, commit, push or merge has occurred.")


def verify_overlay_files(worktree, allow_overview=False):
    d, m = load_manifest()
    w = Path(worktree)
    assert_repo(w)
    for e in m["entries"]:
        path = safe(w, e["path"])
        body = read(path)
        if sha(body) == e["published_sha256"]:
            continue
        if allow_overview and e["path"] == "notebooks/portfolio_overview.ipynb":
            original = json.loads(read(safe(d / "overlay", e["path"])))
            changed = json.loads(body)
            original_sources = [
                (c["cell_type"], nb_text(c.get("source"))) for c in original["cells"]
            ]
            actual_sources = [
                (c["cell_type"], nb_text(c.get("source"))) for c in changed.get("cells", [])
            ]
            if original_sources != actual_sources:
                raise Stop("Portfolio notebook source changed; it needs a new review.")
            errors = [
                o
                for c in changed["cells"]
                for o in c.get("outputs", [])
                if o.get("output_type") == "error"
            ]
            plots = sum(
                PLOT in o.get("data", {}) for c in changed["cells"] for o in c.get("outputs", [])
            )
            cells = [c for c in changed["cells"] if c["cell_type"] == "code"]
            if errors or plots != 6 or any(c.get("execution_count") is None for c in cells):
                raise Stop(
                    "Save the executed portfolio overview with six Plotly figures and no errors."
                )
            scan(body, e["path"])
        else:
            raise Stop("Worktree differs from reviewed overlay: " + e["path"])
    return d, m


def stage(worktree, ack):
    if not ack:
        raise Stop("Explicit public review confirmation is required.")
    d, m = verify_overlay_files(worktree, allow_overview=True)
    w = Path(worktree)
    if (
        w.resolve() == Path(m["source_root"]).resolve()
        or git(w, "branch", "--show-current").decode().strip()
        != "chore/public-research-update-20260913"
    ):
        raise Stop(
            "Stage only in the declared review worktree branch, never the research checkout or main."
        )
    expected = {e["path"] for e in m["entries"]}
    already = set(
        x for x in git(w, "diff", "--cached", "--name-only", "-z").decode().split("\0") if x
    )
    if already - expected:
        raise Stop("Unexpected files are already staged. Review them; no index reset is performed.")
    # Literal paths, never an unrestricted directory add. Force applies only to
    # this explicit manifest, because aggregate reports can be gitignored.
    paths = d / "paths.nul"
    git(
        w,
        "--literal-pathspecs",
        "add",
        "-f",
        "--pathspec-from-file=" + str(paths),
        "--pathspec-file-nul",
    )
    staged = set(
        x for x in git(w, "diff", "--cached", "--name-only", "-z").decode().split("\0") if x
    )
    if not staged or staged - expected:
        raise Stop("Empty or unexpected staged update.")
    deleted = git(w, "diff", "--cached", "--diff-filter=D", "--name-only").strip()
    if deleted:
        raise Stop("Unexpected staged deletion.")
    receipt = []
    for rel in sorted(staged):
        body = git(w, "show", ":" + rel)
        scan(body, rel)
        if sha(body) != sha(read(safe(w, rel))):
            raise Stop("Staged content differs from working copy: " + rel)
        receipt.append(dict(path=rel, sha256=sha(body), bytes=len(body)))
    atomic(
        d / "staged.json", encode(dict(status="STAGED_FOR_HUMAN_REVIEW", files=receipt, utc=utc()))
    )
    print("RESULT: STAGED_FOR_HUMAN_REVIEW")
    print("Run git diff --cached --stat and git diff --cached before committing.")


def verify_merge(root, worktree):
    d, m = load_manifest()
    staged = json.loads(read(d / "staged.json"))
    w = Path(worktree)
    assert_repo(w)
    tip = git(w, "rev-parse", "origin/main").decode().strip()
    results = []
    for e in staged["files"]:
        raw = git(w, "show", "origin/main:" + e["path"], optional=True)
        if raw is None or sha(raw) != e["sha256"]:
            raise Stop("Fetched main does not contain the exact staged publication: " + e["path"])
        results.append(e["path"])
    unchanged = all(sha(read(safe(root, rel))) == h for rel, h in m["source_hashes"].items())
    receipt = dict(
        status="FETCHED_MAIN_PUBLICATION_MATCHES",
        checked_utc=utc(),
        main_commit=tip,
        files_verified=len(results),
        research_source_unchanged=unchanged,
        research_head=git(root, "rev-parse", "HEAD").decode().strip(),
        account_api_calls=0,
        remote_fetch_performed_by_helper=False,
        scope="Exact staged files checked against origin/main fetched by the user. Active research checkout remains pinned; not a byte-for-byte full workspace mirror.",
    )
    atomic(d / "merge_verification.json", encode(receipt))
    print(json.dumps(receipt, indent=2))


def collect(root):
    out = safe(STORE, "return_reports.zip")
    names = []
    for n in (16, 17, 18, 19):
        row = next(z for z in STUDIES if z[0] == n)
        rel = "logs/manual_" + row[1] + "/" + row[2]
        if safe(root, rel).exists():
            names.append(rel)
    names += [
        x
        for x in (
            "logs/manual_next_research/preflight.json",
            "logs/manual_next_research/tests.json",
            "logs/manual_next_research/tests.log",
        )
        if safe(root, x).exists()
    ]
    metadata = dict(
        created_utc=utc(),
        included=names,
        not_included="No raw data, models, feature arrays or credentials are intentionally included. Review before sharing.",
    )
    with tempfile.NamedTemporaryFile(dir=STORE, suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("contents.json", encode(metadata))
            for rel in names:
                body = read(safe(root, rel))
                scan(body, rel)
                z.writestr(Path(rel).name, body)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    print("RETURN_PACKAGE:", out)


class PublicationTests(unittest.TestCase):
    """Synthetic local tests only; never a private-data experiment."""

    def test_parent_rejected(self):
        with self.assertRaises(Stop):
            safe("/tmp", "../x")

    def test_absolute_rejected(self):
        with self.assertRaises(Stop):
            safe("/tmp", "/x")

    def test_backslash_rejected(self):
        with self.assertRaises(Stop):
            safe("/tmp", "a\\b")

    def test_newline_rejected(self):
        with self.assertRaises(Stop):
            safe("/tmp", "a\nb")

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "link").symlink_to("/tmp")
            with self.assertRaises(Stop):
                safe(d, "link/x")

    def test_regular_source_allowed(self):
        self.assertTrue(eligible("src/x.py"))

    def test_tests_allowed(self):
        self.assertTrue(eligible("tests/test_x.py"))

    def test_config_allowed(self):
        self.assertTrue(eligible("configs/round.json"))

    def test_raw_excluded(self):
        self.assertFalse(eligible("data/raw/train.csv"))

    def test_models_excluded(self):
        self.assertFalse(eligible("artifacts/model.joblib"))

    def test_environment_excluded(self):
        self.assertFalse(eligible(".venv/file.py"))

    def test_credentials_excluded(self):
        self.assertFalse(eligible("configs/kaggle.json"))

    def test_machine_logs_excluded(self):
        self.assertFalse(eligible("logs/result.json"))

    def test_license_allowed(self):
        self.assertTrue(eligible("LICENSE"))

    def test_unexpected_binary_excluded(self):
        self.assertFalse(eligible("scripts/archive.zip"))

    def test_secret_detected(self):
        token = ("ghp" + "_" + "A" * 36).encode()
        with self.assertRaises(Stop):
            scan(token, "fixture")

    def test_source_regex_not_a_token(self):
        scan(b"pattern = r'gh[pousr]_[A-Za-z0-9]{30,}'", "fixture")

    def test_failed_report_has_no_result_claim(self):
        r = scalar_study(
            encode(
                dict(status="STOPPED", comparison_rows=[dict(variant="x", official_metric=0.9)])
            ),
            18,
            "Example",
            175,
        )
        self.assertEqual(r["comparisons"], [])
        self.assertFalse(r["completed"])

    def test_complete_report_scalar_whitelist(self):
        r = scalar_study(
            encode(
                dict(
                    status="NOTEBOOK_COMPLETE",
                    comparison_rows=[dict(variant="x", official_metric=0.3)],
                    raw_labels=[1, 2],
                )
            ),
            18,
            "Example",
            175,
        )
        self.assertNotIn("raw_labels", r)
        self.assertEqual(r["comparisons"][0]["official_metric"], 0.3)

    def test_nonfinite_not_allowed(self):
        self.assertIsNone(number(float("inf")))

    def test_bool_not_a_score(self):
        self.assertIsNone(number(True))

    def test_all_negative_rows_retained(self):
        r = scalar_study(
            encode(
                dict(
                    status="NOTEBOOK_COMPLETE",
                    comparison_rows=[
                        dict(variant="x", official_metric=-0.2),
                        dict(variant="y", official_metric=-0.1),
                    ],
                )
            ),
            17,
            "Example",
            180,
        )
        self.assertEqual(len(r["comparisons"]), 2)

    def test_notebook_preserves_plotly_and_source(self):
        source = "fig.show()"
        nb = dict(
            nbformat=4,
            nbformat_minor=5,
            metadata={},
            cells=[
                dict(
                    cell_type="code",
                    source=source,
                    execution_count=1,
                    metadata={},
                    outputs=[
                        dict(
                            output_type="display_data",
                            data={PLOT: {"data": [], "layout": {}}},
                            metadata={},
                        )
                    ],
                )
            ],
        )
        b, c = sanitize_notebook(encode(nb))
        a = json.loads(b)
        self.assertEqual(a["cells"][0]["source"], source)
        self.assertEqual(c["kept_plotly_outputs"], 1)

    def test_error_removal_is_disclosed(self):
        nb = dict(
            nbformat=4,
            metadata={},
            cells=[
                dict(
                    cell_type="code",
                    source="x",
                    execution_count=1,
                    metadata={},
                    outputs=[dict(output_type="error", ename="Error", evalue="x", traceback=[])],
                )
            ],
        )
        b, c = sanitize_notebook(encode(nb))
        self.assertEqual(c["original_error_outputs"], 1)
        self.assertIn("original error outputs: 1", b.decode())

    def test_html_table_not_published(self):
        nb = dict(
            nbformat=4,
            metadata={},
            cells=[
                dict(
                    cell_type="code",
                    source="x",
                    execution_count=1,
                    metadata={},
                    outputs=[
                        dict(output_type="display_data", data={"text/html": "<table>raw</table>"})
                    ],
                )
            ],
        )
        b, c = sanitize_notebook(encode(nb))
        self.assertNotIn("<table>", b.decode())
        self.assertEqual(c["removed_outputs"], 1)

    def test_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "report.json")
            atomic(p, encode({"a": 1}))
            self.assertEqual(json.loads(read(p)), {"a": 1})

    def test_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x")
            p.write_bytes(b"abc")
            with self.assertRaises(Stop):
                read(p, 2)

    def test_svg_scope_label(self):
        self.assertIn(b"not a leaderboard score", reference_svg())

    def test_manifest_digest_changes(self):
        self.assertNotEqual(sha(b"a"), sha(b"b"))

    def test_portfolio_starts_unexecuted(self):
        n = json.loads(portfolio_notebook())
        c = [x for x in n["cells"] if x["cell_type"] == "code"]
        self.assertTrue(all(x["execution_count"] is None and x["outputs"] == [] for x in c))

    def test_six_explicit_figures(self):
        n = json.loads(portfolio_notebook())
        self.assertEqual(sum(nb_text(c["source"]).count("show(f,") for c in n["cells"]), 6)

    def test_no_private_repo_in_new_readme(self):
        text = make_readme([]).decode()
        self.assertNotIn("commodity-prediction-research", text)


def self_test():
    STORE.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublicationTests)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    atomic(STORE / "self_test.log", output.getvalue().encode())
    receipt = dict(
        status="TESTS_PASSED" if result.wasSuccessful() else "TESTS_FAILED",
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        utc=utc(),
        helper_sha256=sha(read(Path(__file__))),
        new_forecasting_fits=0,
    )
    atomic(STORE / "self_test.json", encode(receipt))
    print(output.getvalue())
    print(json.dumps(receipt, indent=2))
    if not result.wasSuccessful():
        raise Stop("Publication tests failed. Preserve self_test.log; do not bypass.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worktree", type=Path, default=REVIEW)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("self-test", "prepare", "collect", "apply", "stage", "verify-merge"):
        actions.add_argument("--" + action, action="store_true")
    parser.add_argument("--reviewed-public-code", action="store_true")
    a = parser.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    try:
        with bounded(180):
            if a.self_test:
                self_test()
            elif a.prepare:
                prepare(a.root)
            elif a.collect:
                collect(a.root)
            elif a.apply:
                apply(a.worktree, a.reviewed_public_code)
            elif a.stage:
                stage(a.worktree, a.reviewed_public_code)
            else:
                verify_merge(a.root, a.worktree)
    except Exception as e:
        print("RESULT: STOPPED")
        print("ERROR:", type(e).__name__ + ": " + str(e))
        print("No automatic retry, source reset, remote action or compute shutdown.")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
