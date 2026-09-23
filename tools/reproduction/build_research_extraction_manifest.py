#!/usr/bin/env python3
"""Build the integrity records for the relational-emergence extraction (issue #113).

The extraction does two things that must never be confused with each other: it
copies history out of this repository, and then it deliberately rewrites the
copied blobs so their relative links point at the new layout.  A single manifest
covering both would have to compare rewritten bytes against source hashes, which
is comparing two different things -- so there are two stages and two records.

``--stage raw``
    Runs against the extracted tree *before* link rewriting.  Every extracted
    file must hash to its source blob's SHA-256, with the path mapped through the
    renames.  A mismatch exits non-zero, and link rewriting must not start.
    Writes ``reproduction/migration/relational_emergence_raw_verification.json``.

``--stage transformed``
    Runs after link rewriting.  Records what the extracted tree now contains --
    post-rewrite paths and hashes -- and points at the raw record it came from.
    It compares nothing against source hashes.  Writes
    ``reproduction/migration/relational_emergence_extracted.json``.

Hashes are taken over git blobs, never worktree files: ``core.autocrlf`` is
commonly set, so the same logical content has different bytes on disk on
different platforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


RAW_RECORD = Path("reproduction") / "migration" / "relational_emergence_raw_verification.json"
TRANSFORMED_RECORD = Path("reproduction") / "migration" / "relational_emergence_extracted.json"

BASELINE_PATH = Path("reproduction") / "evidence" / "repository_boundary_baseline.json"

# Kept in step with reproduction/migration/relational_emergence_paths.txt. The
# order matters: the first matching prefix wins, and the renames there compose
# the same way.
RENAMES: tuple[tuple[str, str], ...] = (
    ("tools/relational_emergence/", "src/relational_research/"),
    ("tests/analysis/", "tests/"),
    ("outputs/reports/relational_emergence/", "reports/"),
    ("reproduction/adr/", "protocols/"),
)


def map_source_path(path: str) -> str:
    """Map a path in this repository to its path in the extracted one."""
    for old, new in RENAMES:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def git(repo: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        **({} if binary else {"encoding": "utf-8", "errors": "replace"}),
    )


def head_blobs(repo: Path) -> dict[str, str]:
    """Return ``{path: blob oid}`` for every file at ``HEAD``."""
    result = git(repo, "ls-tree", "-r", "-z", "HEAD")
    blobs: dict[str, str] = {}
    for record in result.stdout.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if len(fields) == 3 and fields[1] == "blob" and path:
            blobs[path] = fields[2]
    return blobs


def blob_digests(repo: Path, oids: list[str]) -> dict[str, tuple[str, int]]:
    """Return ``{oid: (sha256, byte count)}`` using one ``cat-file --batch``."""
    unique = sorted(set(oids))
    if not unique:
        return {}
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=b"\n".join(oid.encode() for oid in unique) + b"\n",
        check=True,
        capture_output=True,
    )
    digests: dict[str, tuple[str, int]] = {}
    out = result.stdout
    pos = 0
    while pos < len(out):
        header_end = out.find(b"\n", pos)
        if header_end < 0:
            break
        header = out[pos:header_end].split()
        pos = header_end + 1
        if len(header) != 3:
            break
        oid = header[0].decode("ascii", errors="replace")
        size = int(header[2])
        contents = out[pos : pos + size]
        pos += size + 1
        digests[oid] = (hashlib.sha256(contents).hexdigest(), size)
    return digests


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON with LF endings.

    ``newline="\\n"`` explicitly: the default translates to ``os.linesep``, which
    would make a tracked file differ between Windows and Linux for no reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def extracted_state(extraction: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return ``(tip sha, {path: {sha256, bytes}})`` for the extracted tree."""
    tip = git(extraction, "rev-parse", "HEAD").stdout.strip()
    blobs = head_blobs(extraction)
    digests = blob_digests(extraction, list(blobs.values()))
    state = {
        path: {"sha256": digests[oid][0], "bytes": digests[oid][1]}
        for path, oid in blobs.items()
    }
    return tip, state


def run_raw(root: Path, extraction: Path) -> int:
    baseline = json.loads((root / BASELINE_PATH).read_text(encoding="utf-8"))
    frozen = baseline["frozen_surface"]

    tip, state = extracted_state(extraction)

    mismatched: list[dict[str, str]] = []
    missing: list[str] = []
    entries: list[dict[str, Any]] = []
    for source_path in sorted(frozen):
        extracted_path = map_source_path(source_path)
        actual = state.get(extracted_path)
        if actual is None:
            missing.append(extracted_path)
            continue
        expected = frozen[source_path]["sha256"]
        entries.append(
            {
                "source_path": source_path,
                "extracted_path": extracted_path,
                "sha256": actual["sha256"],
            }
        )
        if actual["sha256"] != expected:
            mismatched.append(
                {
                    "source_path": source_path,
                    "extracted_path": extracted_path,
                    "expected": expected,
                    "actual": actual["sha256"],
                }
            )

    extra = sorted(set(state) - {entry["extracted_path"] for entry in entries})

    if mismatched or missing or extra:
        # The record is deliberately not written: it certifies a verified state,
        # and a failing run has verified nothing.  Overwriting a passing record
        # with a failing one would destroy the certification -- which is exactly
        # what happens if this stage is run after link rewriting, when the
        # rewritten blobs are *supposed* to differ from the source.
        print("Raw extraction verification FAILED:")
        for entry in mismatched:
            print(
                f"- {entry['extracted_path']}: expected {entry['expected']}, "
                f"got {entry['actual']} (from {entry['source_path']})"
            )
        for path in missing:
            print(f"- missing from the extraction: {path}")
        for path in extra:
            print(f"- present but not selected: {path}")
        if mismatched:
            print(
                "\nIf these files were already link-rewritten, this stage ran out "
                "of order: it must run against the tree as filtered, before any "
                "content is changed."
            )
        return 1

    record = {
        "stage": "raw",
        "preserved_commit": baseline["preserved_commit"],
        "extracted_tip": tip,
        "file_count": len(state),
        "checked": len(entries),
        "matched": len(entries),
        "mismatched": [],
        "missing": [],
        "unexpected": [],
    }
    write_json(root / RAW_RECORD, record)

    print(
        f"Raw extraction verification passed: {len(entries)} files match the "
        f"source blobs at {baseline['preserved_commit'][:8]} "
        f"(extracted tip {tip[:8]})."
    )
    return 0


def run_transformed(root: Path, extraction: Path) -> int:
    raw_path = root / RAW_RECORD
    if not raw_path.is_file():
        print(f"missing {RAW_RECORD.as_posix()}; run --stage raw first")
        return 1
    raw_bytes = raw_path.read_bytes()
    raw_record = json.loads(raw_bytes.decode("utf-8"))
    if raw_record.get("mismatched") or raw_record.get("missing") or raw_record.get("unexpected"):
        print("the raw record lists failures; re-run --stage raw before transforming")
        return 1

    tip, state = extracted_state(extraction)

    record = {
        "stage": "transformed",
        "extracted_tip": tip,
        "file_count": len(state),
        "files": [
            {"path": path, "sha256": state[path]["sha256"], "bytes": state[path]["bytes"]}
            for path in sorted(state)
        ],
        # Provenance, not a comparison: the transformed blobs were deliberately
        # rewritten and are never checked against the source hashes again.
        "raw_verification": {
            "record": RAW_RECORD.as_posix(),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "checked": raw_record["checked"],
            "matched": raw_record["matched"],
        },
    }
    write_json(root / TRANSFORMED_RECORD, record)

    print(
        f"Transformed manifest written: {len(state)} files at {tip[:8]} "
        f"(from raw record {raw_record['matched']}/{raw_record['checked']} matched)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="This repository's root (where the migration records are written)",
    )
    parser.add_argument(
        "--extraction",
        type=Path,
        required=True,
        help="Path to the extracted research repository",
    )
    parser.add_argument("--stage", choices=("raw", "transformed"), required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    extraction = args.extraction.resolve()
    if not (extraction / ".git").exists():
        parser.error(f"{extraction} is not a git repository")

    if args.stage == "raw":
        return run_raw(root, extraction)
    return run_transformed(root, extraction)


if __name__ == "__main__":
    raise SystemExit(main())
