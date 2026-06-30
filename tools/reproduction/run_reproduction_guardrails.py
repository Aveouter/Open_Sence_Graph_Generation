#!/usr/bin/env python3
"""Run lightweight reproduction workflow guardrails.

This script is safe for local use and CI because it does not run model
evaluation, download assets, or rewrite evidence JSON files. It checks that the
reproduction tools compile, required reports are tracked, and claim-bearing text
does not overstate reproduction evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CLAIM_PATHS = (
    "AGENTS.md",
    "reproduction",
    "docs/reproduction",
    "tools/reproduction",
)


def git_root(start: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not resolve git root from {start}") from exc
    return Path(result.stdout.strip())


def run_step(name: str, command: list[str], cwd: Path) -> int:
    print(f"==> {name}", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"{name} failed with exit code {result.returncode}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root, or any path inside the repository",
    )
    parser.add_argument(
        "--changed-from",
        default="origin/main",
        help="Git ref used by the claim-boundary scan",
    )
    parser.add_argument(
        "--claim-path",
        action="append",
        dest="claim_paths",
        help="Additional claim-bearing file or directory to scan",
    )
    args = parser.parse_args()

    try:
        root = git_root(args.root.resolve())
    except RuntimeError as exc:
        parser.error(str(exc))

    claim_paths = list(DEFAULT_CLAIM_PATHS)
    if args.claim_paths:
        claim_paths.extend(args.claim_paths)

    reproduction_tools = sorted((root / "tools" / "reproduction").glob("*.py"))
    steps = [
        (
            "Compile reproduction tools",
            [sys.executable, "-m", "py_compile", *(str(path) for path in reproduction_tools)],
        ),
        (
            "Run reproduction unit tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/reproduction"],
        ),
        (
            "Check reproduction report completeness",
            [sys.executable, "tools/reproduction/check_reproduction_docs.py"],
        ),
        (
            "Check reproduction claim boundaries",
            [
                sys.executable,
                "tools/reproduction/check_reproduction_claims.py",
                "--changed-from",
                args.changed_from,
                *claim_paths,
            ],
        ),
    ]

    for name, command in steps:
        returncode = run_step(name, command, root)
        if returncode != 0:
            return returncode

    print("Reproduction guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
