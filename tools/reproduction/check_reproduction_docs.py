#!/usr/bin/env python3
"""Check reproduction documentation completeness.

This guardrail verifies that each target baseline has the auditable phase
documents needed for a deferred reproduction report. It does not prove that a
baseline is reproduced; it only checks that the repo contains the evidence trail
required before claims can be reviewed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


BASELINES: dict[str, dict[str, Any]] = {
    "freq": {
        "label": "FREQ",
        "evidence_section": "### FREQ",
        "input_check": "sgb_freq_input_check.json",
    },
    "tde": {
        "label": "TDE",
        "evidence_section": "### TDE",
        "input_check": "tde_official_input_check.json",
    },
    "vctree": {
        "label": "VCTree",
        "evidence_section": "### VCTree",
        "input_check": "vctree_official_input_check.json",
    },
    "penet": {
        "label": "PENet",
        "evidence_section": "### PENet",
        "input_check": "penet_official_input_check.json",
    },
    "shagcl": {
        "label": "SHA-GCL",
        "evidence_section": "### SHA-GCL",
        "input_check": "shagcl_official_input_check.json",
    },
    "ra_sgg": {
        "label": "RA-SGG",
        "matrix_label": "RA-SGG",
        "evidence_section": "### RA-SGG",
        "input_check": "rasgg_official_input_check.json",
    },
}

REQUIRED_REPRODUCTION_DOCS = (
    "README.md",
    "USAGE.md",
    "glossary.md",
    "evidence_gates.md",
    "baselines/README.md",
    "baselines/alignment_reaudit.md",
    "baselines/deferred_audit_matrix.md",
    "evidence/artifact_manifest.json",
    "evidence/evidence_gate_summary.json",
)

REQUIRED_TOOLS = (
    "build_artifact_manifest.py",
    "check_reproduction_claims.py",
    "check_reproduction_docs.py",
    "run_evidence_gate_checks.py",
    "run_reproduction_guardrails.py",
)

REQUIRED_TESTS = (
    "tests/reproduction/test_build_artifact_manifest.py",
    "tests/reproduction/test_check_reproduction_claims.py",
    "tests/reproduction/test_check_reproduction_docs.py",
    "tests/reproduction/test_run_evidence_gate_checks.py",
)

ALLOWED_NON_REPRODUCED_STATUSES = (
    "DEFERRED_NOT_REPRODUCED",
    "IMPLEMENTATION_AUDIT",
    "PIPELINE_SMOKE_ONLY",
    "CHECKPOINT_UNAVAILABLE",
    "PROTOCOL_MISMATCH",
    "FAILED_WITH_REPORT",
)


def git_tracked_paths(root: Path) -> set[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("failed to list git-tracked files") from exc
    return {root / line for line in result.stdout.splitlines()}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path, findings: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(f"{path}: invalid JSON: {exc}")
        return {}


def add_file_check(
    findings: list[str],
    path: Path,
    tracked_paths: set[Path],
    *,
    require_tracked: bool,
) -> None:
    if not path.is_file():
        findings.append(f"missing file: {path}")
        return
    if require_tracked and path not in tracked_paths:
        findings.append(f"file exists but is not tracked by git: {path}")


def check_global_docs(
    root: Path, tracked_paths: set[Path], require_tracked: bool
) -> list[str]:
    findings: list[str] = []
    reproduction_root = root / "reproduction"
    for name in REQUIRED_REPRODUCTION_DOCS:
        add_file_check(
            findings,
            reproduction_root / name,
            tracked_paths,
            require_tracked=require_tracked,
        )
    tools_root = root / "tools" / "reproduction"
    for name in REQUIRED_TOOLS:
        add_file_check(
            findings, tools_root / name, tracked_paths, require_tracked=require_tracked
        )
    for name in REQUIRED_TESTS:
        add_file_check(
            findings, root / name, tracked_paths, require_tracked=require_tracked
        )
    return findings


def check_suite_summary(
    root: Path,
    selected: list[str],
    tracked_paths: set[Path],
    require_tracked: bool,
) -> list[str]:
    findings: list[str] = []
    summary_path = root / "reproduction" / "evidence" / "evidence_gate_summary.json"
    add_file_check(
        findings, summary_path, tracked_paths, require_tracked=require_tracked
    )
    if not summary_path.is_file():
        return findings

    summary = read_json(summary_path, findings)
    if not summary:
        return findings

    if summary.get("claim") != "not reproduction-ready":
        findings.append(
            f"{summary_path}: claim must remain 'not reproduction-ready' until all gates pass"
        )
    if summary.get("status") not in {"BLOCKED", "PASS"}:
        findings.append(
            f"{summary_path}: unexpected suite status {summary.get('status')!r}"
        )

    results = summary.get("results")
    if not isinstance(results, list):
        findings.append(f"{summary_path}: results must be a list")
        return findings

    by_key = {item.get("key"): item for item in results if isinstance(item, dict)}
    for key in selected:
        item = by_key.get(key)
        if not item:
            findings.append(f"{summary_path}: missing suite result for {key}")
            continue
        expected_output = f"reproduction/evidence/{key}/{BASELINES[key]['input_check']}"
        if item.get("output") != expected_output:
            findings.append(
                f"{summary_path}: {key} output should be {expected_output}, got {item.get('output')!r}"
            )
        if item.get("status") == "PASS":
            continue
        if item.get("status") != "BLOCKED":
            findings.append(
                f"{summary_path}: {key} unexpected status {item.get('status')!r}"
            )
        if not item.get("blockers"):
            findings.append(f"{summary_path}: {key} BLOCKED result must list blockers")

    return findings


def check_baseline(
    root: Path,
    key: str,
    info: dict[str, Any],
    tracked_paths: set[Path],
    require_tracked: bool,
) -> list[str]:
    findings: list[str] = []
    evidence_root = root / "reproduction" / "evidence"
    baseline_root = evidence_root / key
    add_file_check(
        findings,
        baseline_root / info["input_check"],
        tracked_paths,
        require_tracked=require_tracked,
    )

    matrix_path = root / "reproduction" / "baselines" / "deferred_audit_matrix.md"
    if matrix_path.is_file():
        matrix_text = read_text(matrix_path)
        matrix_label = info.get("matrix_label", info["label"])
        if f"| {matrix_label} |" not in matrix_text:
            findings.append(f"{matrix_path}: missing matrix row for {matrix_label}")
        if "DEFERRED_NOT_REPRODUCED" not in matrix_text:
            findings.append(f"{matrix_path}: missing deferred status language")

    evidence_path = root / "reproduction" / "evidence_gates.md"
    if evidence_path.is_file():
        evidence_text = read_text(evidence_path)
        if info["evidence_section"] not in evidence_text:
            findings.append(
                f"{evidence_path}: missing method gate section {info['evidence_section']}"
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root, or any path inside the repository",
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        action="append",
        help="Baseline key to check; defaults to all target baselines",
    )
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="Do not fail when required docs exist locally but are not tracked by git",
    )
    parser.add_argument(
        "--json", type=Path, help="Optional path for JSON report output"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            parser.error(f"could not resolve git root from {root}: {exc}")
        root = Path(result.stdout.strip())

    try:
        tracked_paths = git_tracked_paths(root)
    except RuntimeError as exc:
        parser.error(str(exc))

    require_tracked = not args.allow_untracked
    selected = args.baseline or sorted(BASELINES)
    report: dict[str, Any] = {
        "root": str(root),
        "require_tracked": require_tracked,
        "baselines": {},
    }

    findings = check_global_docs(root, tracked_paths, require_tracked)
    findings.extend(check_suite_summary(root, selected, tracked_paths, require_tracked))
    report["global_findings"] = findings

    for key in selected:
        baseline_findings = check_baseline(
            root,
            key,
            BASELINES[key],
            tracked_paths,
            require_tracked,
        )
        report["baselines"][key] = {
            "status": "PASS" if not baseline_findings else "FAIL",
            "findings": baseline_findings,
        }
        findings.extend(baseline_findings)

    report["status"] = "PASS" if not findings else "FAIL"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if findings:
        print("Reproduction documentation check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Reproduction documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
