#!/usr/bin/env python3
"""Check official PENet reproduction inputs.

PENet reproduction requires the official PrototypeEmbeddingNetwork source,
SGB/PENET-format Visual Genome inputs, the pretrained detector checkpoint, and
a trusted PENet checkpoint for the target protocol. This script records whether
the current workspace has enough evidence to proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_VG_FILES = (
    "VG-SGG-with-attri.h5",
    "VG-SGG-dicts-with-attri.json",
    "image_data.json",
)

CHECKPOINT_PROTOCOLS = {
    "predcls": ("PE-NET_PredCls", "1rjsLs3N33iiOB5xYO7zetNhR7ebi385W"),
    "sgcls": ("PE-NET_SGCls", "1uRl-O-yXmpCs__l_V-WTdYtbWPl57M1B"),
    "sgdet": ("PE-NET_SGDet", "1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def file_check(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "type": "file",
    }
    if path.is_file():
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)
    return item


def dir_check(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_dir(), "type": "directory"}


def checkpoint_candidates(root: Path, model_dir_name: str) -> list[dict[str, Any]]:
    search_roots = [root / model_dir_name, root / model_dir_name.lower(), root]
    suffixes = {".pth", ".pt", ".pkl", ".ckpt"}
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or path in seen:
                continue
            if path.suffix.lower() not in suffixes and "model" not in path.name.lower():
                continue
            seen.add(path)
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
        "--penet-root",
        type=Path,
        default=Path("/workspace/external/penet_official/PENET"),
    )
    parser.add_argument(
        "--vg-root",
        type=Path,
        default=None,
        help="PENET/SGB-format Visual Genome root; defaults to <penet-root>/datasets/vg",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/pretrained/penet_official"),
    )
    parser.add_argument(
        "--pretrained-detector",
        type=Path,
        default=None,
        help=(
            "Official Faster R-CNN detector checkpoint; defaults to "
            "<penet-root>/checkpoints/pretrained_faster_rcnn/model_final.pth"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.penet_root / "datasets" / "vg"
    detector = (
        args.pretrained_detector
        or args.penet_root / "checkpoints" / "pretrained_faster_rcnn" / "model_final.pth"
    )
    relation_head = (
        args.penet_root
        / "maskrcnn_benchmark"
        / "modeling"
        / "roi_heads"
        / "relation_head"
    )

    source_checks = [
        dir_check(args.penet_root),
        file_check(args.penet_root / "README.md"),
        file_check(args.penet_root / "configs" / "e2e_relation_X_101_32_8_FPN_1x.yaml"),
        file_check(args.penet_root / "scripts" / "train.sh"),
        file_check(args.penet_root / "scripts" / "test.sh"),
        file_check(args.penet_root / "tools" / "relation_train_net.py"),
        file_check(args.penet_root / "tools" / "relation_test_net.py"),
        file_check(relation_head / "roi_relation_predictors.py"),
        file_check(relation_head / "loss.py"),
        file_check(relation_head / "relation_head.py"),
    ]

    vg_checks = [dir_check(vg_root)]
    for name in REQUIRED_VG_FILES:
        vg_checks.append(file_check(vg_root / name))

    detector_check = file_check(detector)
    checkpoint_checks = {}
    for protocol, (model_dir_name, drive_id) in CHECKPOINT_PROTOCOLS.items():
        candidates = checkpoint_candidates(args.checkpoint_root, model_dir_name)
        checkpoint_checks[protocol] = {
            "expected_model_dir": model_dir_name,
            "official_google_drive_id": drive_id,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    missing_source = [item["path"] for item in source_checks if not item["exists"]]
    missing_vg = [item["path"] for item in vg_checks if not item["exists"]]
    missing_ckpts = [
        protocol
        for protocol, item in checkpoint_checks.items()
        if item["candidate_count"] == 0
    ]
    blockers = []
    if missing_source:
        blockers.append("missing_official_penet_source")
    if missing_vg:
        blockers.append("missing_penet_vg_inputs")
    if not detector_check["exists"]:
        blockers.append("missing_pretrained_detector_checkpoint")
    if missing_ckpts:
        blockers.append("missing_official_penet_checkpoints")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "official_repo": "https://github.com/VL-Group/PENET",
        "official_commit": git_commit(args.penet_root),
        "penet_root": str(args.penet_root),
        "vg_root": str(vg_root),
        "checkpoint_root": str(args.checkpoint_root),
        "source_checks": source_checks,
        "vg_checks": vg_checks,
        "pretrained_detector_check": detector_check,
        "checkpoint_checks": checkpoint_checks,
        "missing_source_paths": missing_source,
        "missing_vg_paths": missing_vg,
        "missing_checkpoint_protocols": missing_ckpts,
        "official_protocol_summary": {
            "predictor": "PrototypeEmbeddingNetwork",
            "config": "configs/e2e_relation_X_101_32_8_FPN_1x.yaml",
            "predcls_overrides": [
                "MODEL.ROI_RELATION_HEAD.USE_GT_BOX True",
                "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True",
                "MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork",
                "DTYPE float32",
                "GLOVE_DIR ./datasets/vg/",
                "TEST.ALLOW_LOAD_FROM_CACHE False",
            ],
            "notable_evaluator_detail": (
                "README states PredCls/SGCls use rel_nms from RU-Net/HL-Net "
                "during evaluation."
            ),
        },
        "next_action": (
            "Run official PENet evaluation and compare OpenSGG adapter only after checkpoint/config/evaluator parity"
            if not blockers
            else "Provide missing VG inputs, pretrained detector, and trusted PENet checkpoints"
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
