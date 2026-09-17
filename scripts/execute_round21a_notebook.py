#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path("/home/sagemaker-user/projects/commodity-prediction-current")
NOTEBOOK = ROOT / "notebooks/21_third_place_reproduction.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "panel", "full"], default="smoke")
    parser.add_argument("--cell-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    os.environ["ROUND21A_MODE"] = args.mode

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=args.cell_timeout_seconds,
        kernel_name="commodity-current",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()

    tmp = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(executed, tmp)
    tmp.replace(NOTEBOOK)
    print(f"EXECUTED_NOTEBOOK={NOTEBOOK}")
    print(f"MODE={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
