"""One canonical entry point for data bootstrap, feature research, and status."""

import argparse
import json
from pathlib import Path

from .research import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "research", "status"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sync-s3", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "bootstrap":
        from .cloud import bootstrap_data

        bootstrap_data(root)
    elif args.command == "research":
        run_research(root, sync=args.sync_s3)
    else:
        report = root / "reports/research.json"
        if report.exists():
            result = json.loads(report.read_text())
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in [
                            "phase",
                            "lineage",
                            "feature_gate",
                            "candidate_count",
                            "experiments_completed",
                            "holdout_evaluated",
                        ]
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {"phase": "initialized", "feature_gate": "open", "experiments_completed": 0}
                )
            )


if __name__ == "__main__":
    main()
