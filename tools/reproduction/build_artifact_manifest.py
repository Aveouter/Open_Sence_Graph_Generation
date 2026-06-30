#!/usr/bin/env python3
"""Build a missing-artifact manifest from reproduction evidence gates.

The manifest is a shopping list for paper-aligned reproduction inputs. It is not
evidence of reproduction; it records what official data, checkpoints, detector
weights, or memory-bank artifacts are still needed before a baseline can leave
the deferred/audit state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


KIND_RULES = (
    ("memory", "memory_bank"),
    ("detector", "detector_checkpoint"),
    ("checkpoint", "checkpoint"),
    ("ckpt", "checkpoint"),
    ("vg", "vg_inputs"),
    ("visual genome", "vg_inputs"),
    ("h5", "vg_inputs"),
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
        return {"status": "ERROR", "error": f"failed to read {path}: {exc}"}


def artifact_kind(blocker: str) -> str:
    low = blocker.lower()
    for marker, kind in KIND_RULES:
        if marker in low:
            return kind
    return "other"


def public_blocker(blocker: str) -> str:
    path = Path(blocker)
    if path.is_absolute():
        return path.name
    return blocker


def extract_blockers(summary_item: dict[str, Any], detail: dict[str, Any]) -> list[str]:
    blockers = (
        summary_item.get("blockers")
        or detail.get("blockers")
        or detail.get("missing")
        or []
    )
    if isinstance(blockers, str):
        return [blockers]
    return [str(item) for item in blockers]


def build_manifest(root: Path, summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    artifacts: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}

    for item in summary.get("results", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "unknown"))
        detail_path = root / str(item.get("output", ""))
        detail = read_json(detail_path) if detail_path.is_file() else {}
        blockers = extract_blockers(item, detail)
        requests = []
        for blocker in blockers:
            blocker_text = public_blocker(blocker)
            request = {
                "baseline": key,
                "label": item.get("label"),
                "kind": artifact_kind(blocker_text),
                "blocker": blocker_text,
                "source_report": item.get("output"),
                "next_action": item.get("next_action"),
            }
            requests.append(request)
            artifacts.append(request)
        baselines[key] = {
            "label": item.get("label"),
            "status": item.get("status"),
            "reported_status": item.get("reported_status"),
            "source_report": item.get("output"),
            "next_action": item.get("next_action"),
            "artifact_count": len(requests),
            "artifacts": requests,
        }

    return {
        "claim": "not reproduction-ready",
        "claim_note": (
            "This manifest lists missing official artifacts only. It does not "
            "prove method, checkpoint, config, inference, or evaluator parity."
        ),
        "source_summary": str(summary_path.relative_to(root)),
        "status": summary.get("status", "UNKNOWN"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "baselines": baselines,
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
        "--summary",
        type=Path,
        default=Path("docs/reproduction/evidence_gate_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reproduction/artifact_manifest.json"),
    )
    args = parser.parse_args()

    try:
        root = git_root(args.root.resolve())
    except RuntimeError as exc:
        parser.error(str(exc))

    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    output_path = args.output if args.output.is_absolute() else root / args.output
    manifest = build_manifest(root, summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Artifact manifest status: {manifest['status']}")
    print(f"Missing artifact requests: {manifest['artifact_count']}")
    print(f"Manifest written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
