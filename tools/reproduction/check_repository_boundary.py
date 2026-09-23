#!/usr/bin/env python3
"""Repository boundary guardrail: stable SGG system vs research-only work.

Issue #113 separates two responsibilities that had grown together: the main
repository holds stable SGG code and contracts, and exploratory research lives
elsewhere.  This checker is the mechanism that keeps them from drifting back
together, and it is deliberately a *ratchet* rather than a clean-slate rule.

Why a ratchet: the research surface is still present in main -- the extraction
is staged, not published -- so a checker that simply rejected it would be red on
every run and would be switched off within a week.  Instead the surface is
frozen by path and blob hash, and the checker fails only on *change*: a new
research path, a modification to a frozen one, or a frozen entry that has gone
stale.  Removing the surface later means deleting baseline lines, which is the
one edit the ratchet is designed to require.

The rules:

``R0`` (``research-surface-in-main``)
    The ratchet proper.  Every tracked path of the current relational-emergence
    surface must appear in the freeze with a matching blob hash.

``R1`` (``core-imports-research``)
    A plain rule with no baseline: nothing in the supported runtime may import
    the research package.  Its scope is a fixed tuple, never "all tracked .py",
    so this checker, its test, and the baseline cannot trip it.

``R2``/``R3``/``R4`` (secondary shape rules)
    These classify *shapes* rather than enumerating paths, so they outlive the
    freeze.  R0's surface is today's inventory; once the surface is emptied
    R0 has nothing left to say, and these are what still catch a generated
    research artifact (R2), an experiment runner (R3), or a retired decision
    record (R4) reappearing in main.

Run ``--help`` for usage.  Findings are printed and the exit status is 1 when
any rule fails.  This script never writes the baseline: a checker that rewrites
its own ratchet erases it, and writing tracked files is what once left
``reproduction/evidence/artifact_manifest.json`` with CRLF line endings.  Use
``--print-baseline`` and redirect it yourself, with ``newline="\\n"``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# This check is part of the documented pre-PR workflow, and a finding can name a
# path this console cannot encode.  Pinning the stream avoids a UnicodeEncodeError
# that would kill the check *after* it had already found violations.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


BASELINE_PATH = Path("reproduction") / "evidence" / "repository_boundary_baseline.json"
MIGRATION_PATH = Path("reproduction") / "migration" / "relational_emergence_migration.json"

# 2 added the main-anchor fields: the surface is described by a research-branch
# commit, but reachability has to be asserted against the squash-merged commit
# that actually lives in main. See build_baseline.
BASELINE_VERSION = 2

# Provenance carried through a re-freeze rather than recomputed. See build_baseline.
BASELINE_PROVENANCE_FIELDS = (
    "preserved_tag",
    "preserved_commit",
    "main_anchor_commit",
    "main_anchor_merge_mode",
)

RESEARCH_PACKAGE = "tools.relational_emergence"

# R1 scope.  An explicit tuple on purpose: widening this to every tracked .py
# would make the checker, its test, and its own baseline self-trip, since all
# three necessarily contain the package name as a string literal.
SUPPORTED_RUNTIME_SCOPE = ("src", "utils", "lib", "configs", "train.py")

# The surface groups R0 freezes.  Each frozen entry names one, and a group that
# is not in this table is a finding -- otherwise a typo silently orphans a path
# from every group-level report.
SURFACE_GROUPS: dict[str, str] = {
    "package": "Research package code under tools/relational_emergence/.",
    "tests": "Research test modules under tests/analysis/.",
    "reports": "Research reports under outputs/reports/relational_emergence/.",
    "decision_records": "Phase I ADRs, the retired reproduction/adr/0005-0010 band.",
    "glossary": "The relational-emergence glossary at CONTEXT.md.",
}

_PACKAGE_PREFIX = "tools/relational_emergence/"
_REPORTS_PREFIX = "outputs/reports/relational_emergence/"
_RESEARCH_TEST = re.compile(r"^tests/analysis/test_relational_emergence.*\.py$")
_RESEARCH_ADR = re.compile(r"^reproduction/adr/(?:000[5-9]|0010)-.*\.md$")
_GLOSSARY = "CONTEXT.md"

# R2: generated artifacts.  .gitignore already states this policy in prose
# ("Anything that is not a report (a stray figure, csv or json) is still
# ignored"); nothing implemented it.
_GENERATED_SUFFIXES = (".json", ".csv", ".npy", ".npz", ".pt", ".png")

# R3: experiment-runner shapes.  The issue's own list is diagnose_*/run_phase*/
# summarise_phase*, which misses run_level2.py -- that one is covered by the
# experiments/ clause below, not by the filename globs.
_RUNNER_STEM = re.compile(r"^(?:diagnose_.*|run_phase.*|summarise_phase.*)\.py$")
_RUNNER_TREE = re.compile(r"(?:^|/)relational_emergence/experiments/[^/]+\.py$")
_RUNNER_VERSIONED = re.compile(r"(?:^|/)relational_emergence/v[0-9][^/]*/")


def surface_group(posix_path: str) -> str | None:
    """Return the frozen surface group a path belongs to, or ``None``.

    A plain function rather than a rule check: both R0 and the shape rules need
    to know which paths the ratchet already owns, so that a surface path is
    reported once, by R0, and never double-counted by R2-R4.
    """
    if posix_path.startswith(_PACKAGE_PREFIX):
        return "package"
    if _RESEARCH_TEST.match(posix_path):
        return "tests"
    if posix_path.startswith(_REPORTS_PREFIX):
        return "reports"
    if _RESEARCH_ADR.match(posix_path):
        return "decision_records"
    if posix_path == _GLOSSARY:
        return "glossary"
    return None


# ---------------------------------------------------------------------------
# Git access
# ---------------------------------------------------------------------------


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


def tracked_blobs(root: Path) -> dict[str, str]:
    """Return ``{posix path: blob oid}`` for every tracked file.

    ``-z`` and an explicit encoding because the default decode is cp1252 on
    Windows, which would corrupt a non-ASCII path rather than fail loudly.
    Unmerged entries (stage != 0) are skipped: a conflicted index is not a
    state worth ratcheting.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-s", "-z"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("failed to list git-tracked files") from exc

    blobs: dict[str, str] = {}
    for record in result.stdout.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if len(fields) != 3 or not path:
            continue
        _mode, oid, stage = fields
        if stage == "0":
            blobs[path] = oid
    return blobs


def unstaged_paths(root: Path) -> set[str]:
    """Return paths modified in the worktree but not yet staged.

    ``ls-files -s`` reports the index, so a frozen file edited on disk but never
    staged would pass the hash comparison -- and the pre-PR workflow runs before
    the commit, which is exactly when that edit exists.  Comparing the worktree to
    the index uses git's clean filter on both sides, so this stays independent of
    ``core.autocrlf`` just as the blob hashes do.

    A repository with no commits, or without git, yields no paths rather than an
    error: this is an extra guard, not one worth failing the whole run over.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", "-z"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    return {path for path in result.stdout.split("\0") if path}


def blob_digests(root: Path, oids: Iterable[str]) -> dict[str, tuple[str, int]]:
    """Return ``{oid: (sha256, byte count)}`` for the given blobs.

    The hash is taken over the *blob*, never the worktree file: ``core.autocrlf``
    is commonly set, so the same logical content has different bytes on disk on
    different platforms, and a worktree hash would make this file's baseline
    machine-dependent.  One ``cat-file --batch`` process serves every blob.
    """
    unique = sorted(set(oids))
    if not unique:
        return {}

    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
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
            # "<oid> missing" or a truncated stream; nothing more is readable.
            break
        oid = header[0].decode("ascii", errors="replace")
        size = int(header[2])
        contents = out[pos : pos + size]
        pos += size + 1
        digests[oid] = (hashlib.sha256(contents).hexdigest(), size)
    return digests


# ---------------------------------------------------------------------------
# Rule R1
# ---------------------------------------------------------------------------


def names_research_package(module: str) -> bool:
    return module == RESEARCH_PACKAGE or module.startswith(RESEARCH_PACKAGE + ".")


def research_import_lines(path: Path) -> list[str]:
    """Return findings for a supported-runtime file that reaches for research code.

    AST rather than a regex: a regex over source text also matches docstrings and
    comments, and this package is discussed in prose in several places.  A syntax
    error becomes a finding rather than an exit, so one broken file does not hide
    every other violation in the run.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return [f"R1 could not parse {path.as_posix()}: {exc}"]

    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if names_research_package(alias.name):
                    findings.append(
                        f"R1 {path.as_posix()}:{node.lineno}: imports {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import and cannot name tools.*, so it can
            # never be a false negative here.
            if node.level == 0 and node.module and names_research_package(node.module):
                findings.append(
                    f"R1 {path.as_posix()}:{node.lineno}: imports {node.module}"
                )
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
            )
            if not dynamic or not node.args:
                continue
            arg = node.args[0]
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and names_research_package(arg.value)
            ):
                findings.append(
                    f"R1 {path.as_posix()}:{node.lineno}: dynamically imports {arg.value}"
                )
    return findings


# ---------------------------------------------------------------------------
# Secondary shape rules
# ---------------------------------------------------------------------------


def generated_artifact_finding(posix_path: str) -> str | None:
    """R2 -- a generated research artifact tracked under outputs/reports/."""
    if not posix_path.startswith("outputs/reports/"):
        return None
    if not posix_path.endswith(_GENERATED_SUFFIXES):
        return None
    return (
        f"R2 {posix_path}: generated artifact tracked under outputs/reports/; "
        "store it in an artifact backend and track only the URI and hash"
    )


def experiment_runner_finding(posix_path: str) -> str | None:
    """R3 -- an experiment driver or versioned research tree in main.

    Scoped so that the supported tools are never caught: tools/reproduction's own
    ``run_*.py`` entry points and tools/analysis's exporters match none of these
    shapes.
    """
    name = posix_path.rsplit("/", 1)[-1]
    if name == "__init__.py":
        return None
    if _RUNNER_STEM.match(name):
        return f"R3 {posix_path}: experiment runner in main; it belongs with its experiment"
    if _RUNNER_TREE.search(posix_path):
        return f"R3 {posix_path}: experiment driver directory in main"
    if _RUNNER_VERSIONED.search(posix_path):
        return f"R3 {posix_path}: versioned research tree in main"
    return None


def retired_decision_record_finding(posix_path: str) -> str | None:
    """R4 -- a retired Phase I decision record or the research glossary.

    The 0005-0010 band is retired rather than reused, so an ADR reusing one of
    those numbers is a finding even once the originals have been extracted.  This
    is inert while R0 still freezes the originals; it is what remains afterwards.
    """
    if _RESEARCH_ADR.match(posix_path):
        return f"R4 {posix_path}: the reproduction/adr/0005-0010 band is retired"
    if posix_path == _GLOSSARY:
        return f"R4 {posix_path}: research-only glossary in main"
    return None


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def read_json(path: Path, findings: list[str]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        findings.append(f"could not read {path.as_posix()}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        findings.append(f"{path.as_posix()}: invalid JSON: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def authorized_changes(root: Path, findings: list[str]) -> dict[str, str]:
    """Load migration-authorized edits, keyed by path -> to_sha256.

    A frozen file may change only through a recorded migration.  A flag alone
    would be an escape hatch; an entry has to name the path, both hashes, and who
    approved it, so the change is auditable after the fact.
    """
    path = root / MIGRATION_PATH
    if not path.is_file():
        return {}
    data = read_json(path, findings)
    entries = data.get("authorized_changes", [])
    if not isinstance(entries, list):
        findings.append(f"{MIGRATION_PATH.as_posix()}: authorized_changes must be a list")
        return {}

    allowed: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            findings.append(f"{MIGRATION_PATH.as_posix()}: entries must be objects")
            continue
        missing = [
            key
            for key in ("path", "from_sha256", "to_sha256", "reason", "approved_by")
            if not entry.get(key)
        ]
        if missing:
            findings.append(
                f"{MIGRATION_PATH.as_posix()}: entry for {entry.get('path')!r} "
                f"is missing {', '.join(missing)}"
            )
            continue
        allowed[entry["path"]] = entry["to_sha256"]
    return allowed


def build_baseline(
    freeze: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the baseline, carrying provenance through unchanged.

    Two commits describe the preserved surface, and conflating them is a trap:

    ``preserved_commit``
        Where the freeze and the extraction actually come from -- a research
        branch commit.  It is what the extraction was filtered from, so it is
        the truthful source, but it is *not* in ``main``'s history.

    ``main_anchor_commit``
        A commit in ``main``'s history with byte-identical content, produced by
        a squash merge.  Squash merging rewrites the commit, so the source
        commit is not an ancestor of ``main`` even though its content is.  Any
        assertion of the form ``preserved_commit is an ancestor of HEAD`` is
        therefore false by construction, which is why reachability is asserted
        against this anchor instead.

    ``main_anchor_merge_mode`` records how the two relate, so a future reader
    knows the equivalence is by content and not by ancestry.
    """
    return {
        "version": BASELINE_VERSION,
        "preserved_tag": provenance.get("preserved_tag"),
        "preserved_commit": provenance.get("preserved_commit"),
        "main_anchor_commit": provenance.get("main_anchor_commit"),
        "main_anchor_merge_mode": provenance.get("main_anchor_merge_mode"),
        "surface_groups": SURFACE_GROUPS,
        "frozen_surface": {path: freeze[path] for path in sorted(freeze)},
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_frozen_surface(
    blobs: dict[str, str],
    digests: dict[str, tuple[str, int]],
    frozen: dict[str, Any],
    allowed: dict[str, str],
    dirty: set[str],
    *,
    allow_stale: bool,
) -> tuple[list[str], list[str]]:
    """R0. Returns ``(findings, notes)``."""
    findings: list[str] = []
    notes: list[str] = []

    for path in sorted(blobs):
        group = surface_group(path)
        if group is None:
            continue
        entry = frozen.get(path)
        if entry is None:
            findings.append(f"R0 new research-surface path ({group}): {path}")
            continue
        if path in dirty and path not in allowed:
            findings.append(
                f"R0 frozen research file modified in the working tree: {path}"
            )
            continue
        current = digests.get(blobs[path], ("", -1))[0]
        if current == entry.get("sha256"):
            continue
        if allowed.get(path) == current:
            notes.append(f"R0 {path}: modified under an authorized migration")
            continue
        findings.append(
            f"R0 frozen research file modified: {path}\n"
            f"      frozen {entry.get('sha256')}\n"
            f"      actual {current}\n"
            "      revert it, or record the change in "
            f"{MIGRATION_PATH.as_posix()} and re-freeze with --print-baseline"
        )

    for path in sorted(frozen):
        entry = frozen[path]
        if not isinstance(entry, dict):
            findings.append(f"R0 malformed freeze entry for {path}")
            continue
        if entry.get("surface") not in SURFACE_GROUPS:
            findings.append(
                f"R0 {path}: unknown surface group {entry.get('surface')!r}; "
                f"known groups are {', '.join(sorted(SURFACE_GROUPS))}"
            )
        if path not in blobs:
            if allow_stale:
                notes.append(f"R0 {path}: stale freeze entry (not tracked)")
            else:
                findings.append(
                    f"R0 stale freeze entry (no longer tracked): {path}\n"
                    "      remove it from the baseline with --print-baseline"
                )
    return findings, notes


def check_shape_rules(blobs: dict[str, str]) -> list[str]:
    """R2-R4. Frozen surface paths are skipped; R0 already owns them."""
    findings: list[str] = []
    for path in sorted(blobs):
        if surface_group(path) is not None:
            continue
        for rule in (
            generated_artifact_finding,
            experiment_runner_finding,
            retired_decision_record_finding,
        ):
            finding = rule(path)
            if finding:
                findings.append(finding)
    return findings


def check_core_imports(root: Path, blobs: dict[str, str]) -> list[str]:
    """R1."""
    findings: list[str] = []
    for path in sorted(blobs):
        if not path.endswith(".py"):
            continue
        if not (
            path == "train.py"
            or any(path.startswith(scope + "/") for scope in SUPPORTED_RUNTIME_SCOPE)
        ):
            continue
        findings.extend(research_import_lines(root / path))
    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root, or any path inside the repository",
    )
    parser.add_argument("--json", type=Path, help="Optional path for a JSON report")
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Report freeze entries that are no longer tracked instead of failing",
    )
    parser.add_argument(
        "--print-baseline",
        action="store_true",
        help="Emit a refreshed baseline for the current tree on stdout and exit",
    )
    parser.add_argument(
        "--preserved-tag", help="Preservation tag naming the source commit"
    )
    parser.add_argument(
        "--preserved-commit",
        help="Research-branch commit the freeze and extraction describe",
    )
    parser.add_argument(
        "--main-anchor-commit",
        help="Content-equivalent commit in main's history (reachability anchor)",
    )
    parser.add_argument(
        "--main-anchor-merge-mode",
        help="How the source and anchor relate, e.g. 'squash'",
    )
    args = parser.parse_args()

    try:
        root = git_root(args.root.resolve())
    except RuntimeError as exc:
        parser.error(str(exc))

    bootstrap: list[str] = []
    blobs = tracked_blobs(root)
    digests = blob_digests(root, blobs.values())

    if args.print_baseline:
        existing = read_json(root / BASELINE_PATH, bootstrap) if (root / BASELINE_PATH).is_file() else {}
        freeze = {
            path: {
                "sha256": digests[oid][0],
                "bytes": digests[oid][1],
                "surface": surface_group(path),
            }
            for path, oid in sorted(blobs.items())
            if surface_group(path) is not None
        }
        # Re-freezing refreshes hashes only; the provenance of what is being
        # frozen is not something a re-freeze should silently rewrite, so it is
        # carried from the existing baseline unless given explicitly.
        provenance = {
            field: getattr(args, field) or existing.get(field)
            for field in BASELINE_PROVENANCE_FIELDS
        }
        baseline = build_baseline(freeze, provenance)
        # Written by the caller. json.dumps is deterministic (sorted keys), so
        # two runs over the same tree produce byte-identical output -- but only
        # if the newlines are too. Text-mode stdout translates "\n" to os.linesep,
        # which would make the tracked baseline CRLF on Windows and LF on Linux;
        # the bytes are emitted directly so the redirect cannot vary by platform.
        sys.stdout.flush()
        sys.stdout.buffer.write(
            (json.dumps(baseline, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        sys.stdout.buffer.flush()
        return 0

    findings: list[str] = []
    notes: list[str] = []

    baseline_path = root / BASELINE_PATH
    if not baseline_path.is_file():
        findings.append(
            f"missing baseline: {BASELINE_PATH.as_posix()} "
            "(create it with --print-baseline)"
        )
        baseline: dict[str, Any] = {}
    else:
        baseline = read_json(baseline_path, findings)

    frozen = baseline.get("frozen_surface", {})
    if not isinstance(frozen, dict):
        findings.append(f"{BASELINE_PATH.as_posix()}: frozen_surface must be an object")
        frozen = {}

    allowed = authorized_changes(root, findings)

    if not findings:
        surface_findings, surface_notes = check_frozen_surface(
            blobs,
            digests,
            frozen,
            allowed,
            unstaged_paths(root),
            allow_stale=args.allow_stale,
        )
        findings.extend(surface_findings)
        notes.extend(surface_notes)
        findings.extend(check_core_imports(root, blobs))
        findings.extend(check_shape_rules(blobs))

    report: dict[str, Any] = {
        "root": str(root),
        "baseline": BASELINE_PATH.as_posix(),
        "tracked_files": len(blobs),
        "frozen_entries": len(frozen),
        "findings": findings,
        "notes": notes,
        "status": "PASS" if not findings else "FAIL",
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for note in notes:
        print(f"note: {note}")
    if findings:
        print("Repository boundary check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print(
        f"Repository boundary check passed "
        f"({report['tracked_files']} tracked files, "
        f"{report['frozen_entries']} frozen)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
