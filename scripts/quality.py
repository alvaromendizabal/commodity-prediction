"""Run the reproducible quality gate before publication."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for arguments in [
        ["ruff", "check", "src", "tests", "scripts"],
        ["ruff", "format", "--check", "src", "tests", "scripts"],
        ["mypy", "src"],
        ["pytest", "-q"],
    ]:
        subprocess.run([sys.executable, "-m", *arguments], cwd=root, check=True)


if __name__ == "__main__":
    main()
