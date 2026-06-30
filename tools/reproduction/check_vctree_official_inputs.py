#!/usr/bin/env python3
"""Check official VCTree reproduction inputs.

VCTree reproduction requires SGB-format Visual Genome inputs plus a trusted
VCTree checkpoint. The SGB README states that most SGG checkpoints are not
uploaded, so this script records whether the current workspace has enough
evidence to proceed without training.
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_candidates(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    suffixes = {".pth", ".pt", ".pkl", ".ckpt"}
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes and "vctree" not in path.name.lower():
            continue
        candidates.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return candidates


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
        default=Path("outputs/pretrained/vctree_official"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.sgb_root / "datasets" / "vg"
    vctree_files = (
        args.sgb_root
        / "maskrcnn_benchmark"
        / "modeling"
        / "roi_heads"
        / "relation_head"
    )
    config = args.sgb_root / "configs" / "e2e_relation_R_101_FPN_1x.yaml"

    sgb_checks = [
        {
            "path": str(args.sgb_root),
            "exists": args.sgb_root.is_dir(),
            "type": "directory",
        },
        {"path": str(config), "exists": config.is_file(), "type": "file"},
        {
            "path": str(vctree_files / "roi_relation_predictors.py"),
            "exists": (vctree_files / "roi_relation_predictors.py").is_file(),
            "type": "file",
        },
        {
            "path": str(vctree_files / "model_vctree.py"),
            "exists": (vctree_files / "model_vctree.py").is_file(),
            "type": "file",
        },
        {
            "path": str(vctree_files / "utils_vctree.py"),
            "exists": (vctree_files / "utils_vctree.py").is_file(),
            "type": "file",
        },
        {"path": str(vg_root), "exists": vg_root.is_dir(), "type": "directory"},
    ]
    for name in REQUIRED_SGB_VG_FILES:
        path = vg_root / name
        sgb_checks.append({"path": str(path), "exists": path.is_file(), "type": "file"})

    ckpts = checkpoint_candidates(args.checkpoint_root)
    missing_sgb = [item["path"] for item in sgb_checks if not item["exists"]]
    blockers = []
    if missing_sgb:
        blockers.append("missing_sgb_vg_inputs")
    if not ckpts:
        blockers.append("missing_trusted_vctree_checkpoint")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "sgb_root": str(args.sgb_root),
        "vg_root": str(vg_root),
        "checkpoint_root": str(args.checkpoint_root),
        "sgb_checks": sgb_checks,
        "missing_sgb_paths": missing_sgb,
        "checkpoint_candidates": ckpts,
        "official_readme_checkpoint_note": (
            "SGB README says most SGG model checkpoints are not uploaded; "
            "VCTree generally requires training unless a trusted checkpoint is supplied."
        ),
        "next_action": (
            "Inspect checkpoint keys and run official VCTree evaluation"
            if not blockers
            else "Provide SGB-format VG inputs and a trusted VCTree checkpoint"
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
