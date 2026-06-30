#!/usr/bin/env python3
"""Run official-input evidence checks for the target baseline suite.

The per-baseline checkers distinguish a runnable audit from a reproduction-ready
baseline. This runner executes them consistently, writes each method JSON report,
and emits a suite summary. Exit code 2 means one or more baselines are blocked
by missing official inputs/checkpoints; it is expected until evidence arrives.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CHECKS: dict[str, dict[str, str]] = {
    "freq": {
        "label": "FREQ",
        "script": "tools/reproduction/check_sgb_freq_inputs.py",
        "output": "docs/reproduction/freq/sgb_freq_input_check.json",
    },
    "tde": {
        "label": "TDE",
        "script": "tools/reproduction/check_tde_official_inputs.py",
        "output": "docs/reproduction/tde/tde_official_input_check.json",
    },
    "vctree": {
        "label": "VCTree",
        "script": "tools/reproduction/check_vctree_official_inputs.py",
        "output": "docs/reproduction/vctree/vctree_official_input_check.json",
    },
    "penet": {
        "label": "PENet",
        "script": "tools/reproduction/check_penet_official_inputs.py",
        "output": "docs/reproduction/penet/penet_official_input_check.json",
    },
    "shagcl": {
        "label": "SHA-GCL",
        "script": "tools/reproduction/check_shagcl_official_inputs.py",
        "output": "docs/reproduction/shagcl/shagcl_official_input_check.json",
    },
    "ra_sgg": {
        "label": "RA-SGG",
        "script": "tools/reproduction/check_rasgg_official_inputs.py",
        "output": "docs/reproduction/ra_sgg/rasgg_official_input_check.json",
    },
}

EXPECTED_CODES = {0, 2}
SUMMARY_CLAIM = "not reproduction-ready"
SUMMARY_CLAIM_NOTE = (
    "Official-input gates are necessary but not sufficient; config, "
    "inference, and evaluator parity must still be verified before any "
    "reproduction claim."
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


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "ERROR", "error": f"failed to read JSON: {exc}"}


def normalize_status(returncode: int, report: dict[str, Any]) -> str:
    if returncode not in EXPECTED_CODES:
        return "ERROR"
    status = str(report.get("status", "")).upper()
    blockers = report.get("blockers") or report.get("missing")
    if returncode == 2 or "BLOCKED" in status or blockers:
        return "BLOCKED"
    if returncode == 0:
        return "PASS"
    return "ERROR"


def run_check(root: Path, key: str) -> dict[str, Any]:
    spec = CHECKS[key]
    output = root / spec["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        spec["script"],
        "--output",
        spec["output"],
    ]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    report = read_json(output)
    status = normalize_status(result.returncode, report)
    blockers = report.get("blockers")
    if blockers is None:
        blockers = report.get("missing", [])
    return {
        "key": key,
        "label": spec["label"],
        "status": status,
        "returncode": result.returncode,
        "command": " ".join(command),
        "output": spec["output"],
        "reported_status": report.get("status"),
        "blockers": blockers,
        "next_action": report.get("next_action"),
        "stderr": result.stderr.strip() if status == "ERROR" else "",
    }


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
        choices=sorted(CHECKS),
        action="append",
        help="Baseline key to run; defaults to all target baselines",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/reproduction/evidence_gate_summary.json"),
        help="Suite summary JSON path, relative to repo root unless absolute",
    )
    args = parser.parse_args()

    try:
        root = git_root(args.root.resolve())
    except RuntimeError as exc:
        parser.error(str(exc))

    selected = args.baseline or list(CHECKS)
    results = [run_check(root, key) for key in selected]
    blocked = [item for item in results if item["status"] == "BLOCKED"]
    errors = [item for item in results if item["status"] == "ERROR"]
    suite_status = "ERROR" if errors else "BLOCKED" if blocked else "PASS"

    summary: dict[str, Any] = {
        "status": suite_status,
        "claim": SUMMARY_CLAIM,
        "claim_note": SUMMARY_CLAIM_NOTE,
        "results": results,
    }

    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Evidence gate suite status: {suite_status}")
    for item in results:
        print(f"- {item['label']}: {item['status']} ({item['reported_status']})")
        if item["blockers"]:
            print(f"  blockers: {item['blockers']}")
        if item["next_action"]:
            print(f"  next: {item['next_action']}")
    print(f"Summary written to {summary_path}")

    if errors:
        return 1
    if blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
