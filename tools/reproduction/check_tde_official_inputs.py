#!/usr/bin/env python3
"""Check official TDE reproduction inputs.

This audit script verifies the external artifacts required before TDE can move
from deferred audit to checkpoint-backed reproduction. It intentionally does not
download checkpoints, train, or modify evaluator semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SGB_VG_FILES = (
    "VG-SGG-with-attri.h5",
    "VG-SGG-dicts-with-attri.json",
    "image_data.json",
)

CHECKPOINT_PROTOCOLS = ("predcls", "sgcls", "sgdet")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_checkpoint_dir(path: Path) -> dict[str, Any]:
    files = [p for p in sorted(path.glob("*")) if p.is_file()]
    candidates = [
        p
        for p in files
        if p.suffix.lower() in {".pth", ".pt", ".pkl", ".ckpt"}
        or "model" in p.name.lower()
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in candidates
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgb-root",
        type=Path,
        default=Path("/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch"),
    )
    parser.add_argument(
        "--vg-root",
        type=Path,
        default=None,
        help="SGB-format Visual Genome root; defaults to <sgb-root>/datasets/vg",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/pretrained/tde_official"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.sgb_root / "datasets" / "vg"
    config = args.sgb_root / "configs" / "e2e_relation_X_101_32_8_FPN_1x.yaml"
    predictor = (
        args.sgb_root
        / "maskrcnn_benchmark"
        / "modeling"
        / "roi_heads"
        / "relation_head"
        / "roi_relation_predictors.py"
    )

    sgb_checks = [
        {
            "path": str(args.sgb_root),
            "exists": args.sgb_root.is_dir(),
            "type": "directory",
        },
        {"path": str(config), "exists": config.is_file(), "type": "file"},
        {"path": str(predictor), "exists": predictor.is_file(), "type": "file"},
        {"path": str(vg_root), "exists": vg_root.is_dir(), "type": "directory"},
    ]
    for name in REQUIRED_SGB_VG_FILES:
        path = vg_root / name
        sgb_checks.append({"path": str(path), "exists": path.is_file(), "type": "file"})

    checkpoint_checks = {
        protocol: inspect_checkpoint_dir(args.checkpoint_root / protocol)
        for protocol in CHECKPOINT_PROTOCOLS
    }

    missing_sgb = [item["path"] for item in sgb_checks if not item["exists"]]
    missing_ckpts = [
        protocol
        for protocol, item in checkpoint_checks.items()
        if item["candidate_count"] == 0
    ]
    blockers = []
    if missing_sgb:
        blockers.append("missing_sgb_vg_inputs")
    if missing_ckpts:
        blockers.append("missing_official_tde_checkpoints")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "sgb_root": str(args.sgb_root),
        "vg_root": str(vg_root),
        "checkpoint_root": str(args.checkpoint_root),
        "sgb_checks": sgb_checks,
        "checkpoint_checks": checkpoint_checks,
        "missing_sgb_paths": missing_sgb,
        "missing_checkpoint_protocols": missing_ckpts,
        "next_action": (
            "Inspect checkpoint keys and run official evaluation"
            if not blockers
            else "Provide missing SGB-format VG inputs and official TDE checkpoints"
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
