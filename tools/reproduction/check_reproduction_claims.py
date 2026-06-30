#!/usr/bin/env python3
"""Guardrail for misleading reproduction claims.

This script scans text files for strong reproduction claims and checks whether
they are made near weak evidence terms such as random init, tiny slices, or zero
metrics. Passing the script does not prove reproduction; failing it means a
human should tighten the claim boundary before review.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


CLAIM_PATTERNS = (
    r"\breproduced\b",
    r"\bsuccessfully reproduced\b",
    r"\breproduction (success|complete|completed|ready)\b",
    r"\bpaper[- ]aligned\b",
    r"\bmatches paper\b",
    r"\bbaseline results?\b",
)

WEAK_PATTERNS = (
    r"\brandom[- ]init(ialization)?\b",
    r"\btiny[- ]slice\b",
    r"\ball 0\.0\b",
    r"\ball[- ]zero\b",
    r"\bpartial remap(ping)?\b",
    r"\bpartial checkpoint\b",
    r"\bsmoke only\b",
)

QUALIFIER_PATTERNS = (
    r"\bnot\b",
    r"\bno\b",
    r"\bdo not\b",
    r"\bcannot\b",
    r"\bdeferred\b",
    r"\bnot counted\b",
    r"\bnot reproduction\b",
    r"\bnot claimed\b",
    r"\bonly when\b",
    r"\bonly be called\b",
    r"\brequired evidence\b",
    r"\bmust not\b",
    r"\bmay not\b",
    r"\bmay call\b",
    r"\bforbidden\b",
    r"\bnot established\b",
)


def matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def scan_file(path: Path) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    findings: list[str] = []

    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if low.lstrip().startswith("|") and (
            "|---" in low or "method" in low and "paper aligned" in low
        ):
            continue
        if not matches_any(low, CLAIM_PATTERNS):
            continue

        window_start = max(0, lineno - 4)
        window_end = min(len(lines), lineno + 3)
        context = "\n".join(lines[window_start:window_end]).lower()
        if matches_any(context, QUALIFIER_PATTERNS):
            continue
        if matches_any(context, WEAK_PATTERNS):
            findings.append(
                f"{path}:{lineno}: reproduction claim appears near weak evidence: {line}"
            )
        else:
            findings.append(f"{path}:{lineno}: review reproduction claim: {line}")
    return findings


def iter_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(
                child
                for child in p.rglob("*")
                if child.is_file() and child.suffix.lower() in {".md", ".txt", ".rst"}
            )
        elif p.is_file():
            paths.append(p)
        else:
            missing.append(raw)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"input path does not exist or is not a file/directory: {joined}")
    return paths


def git_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "--changed-from requires running inside a git repository with git available"
        ) from exc
    return Path(result.stdout.strip())


def changed_text_paths(base: str) -> list[Path]:
    root = git_root()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", f"{base}...HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"failed to resolve changed files for --changed-from {base!r}"
        if details:
            message += f": {details}"
        raise RuntimeError(message) from exc
    paths: list[Path] = []
    for raw in result.stdout.splitlines():
        path = root / raw
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        help="Text files or directories to scan; pass changed docs, PR text, and commit/PR descriptions when available",
    )
    parser.add_argument(
        "--changed-from",
        help="Also scan changed Markdown/text files from the merge-base with this git ref",
    )
    args = parser.parse_args()

    try:
        paths = iter_paths(args.paths)
    except ValueError as exc:
        parser.error(str(exc))
    if args.changed_from:
        try:
            paths.extend(changed_text_paths(args.changed_from))
        except RuntimeError as exc:
            parser.error(str(exc))
    # Preserve order while avoiding duplicate reports.
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("provide paths or --changed-from")

    findings: list[str] = []
    for path in paths:
        findings.extend(scan_file(path))

    if findings:
        print("Potentially misleading reproduction claims found:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("No obvious misleading reproduction claims found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
