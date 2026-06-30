#!/usr/bin/env python3
"""Check official SHA-GCL reproduction inputs."""

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
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "type": "file"}
    if path.is_file():
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)
    return item


def dir_check(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_dir(), "type": "directory"}


def checkpoint_candidates(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    suffixes = {".pth", ".pt", ".pkl", ".ckpt"}
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes and "model" not in path.name.lower():
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
        "--shagcl-root",
        type=Path,
        default=Path("/workspace/external/shagcl_official/SHA-GCL-for-SGG"),
    )
    parser.add_argument(
        "--vg-root",
        type=Path,
        default=None,
        help="Official VG root; defaults to <shagcl-root>/datasets/vg",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/pretrained/shagcl_official"),
    )
    parser.add_argument(
        "--pretrained-detector",
        type=Path,
        default=None,
        help="Defaults to <vg-root>/detector_model/pretrained_faster_rcnn/model_final.pth",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.shagcl_root / "datasets" / "vg"
    detector = (
        args.pretrained_detector
        or vg_root / "detector_model" / "pretrained_faster_rcnn" / "model_final.pth"
    )
    relation_head = (
        args.shagcl_root
        / "maskrcnn_benchmark"
        / "modeling"
        / "roi_heads"
        / "relation_head"
    )

    source_checks = [
        dir_check(args.shagcl_root),
        file_check(args.shagcl_root / "README.md"),
        file_check(
            args.shagcl_root / "configs" / "SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml"
        ),
        file_check(args.shagcl_root / "tools" / "relation_train_net.py"),
        file_check(args.shagcl_root / "tools" / "relation_test_net.py"),
        file_check(relation_head / "roi_relation_predictors.py"),
        file_check(args.shagcl_root / "SHA_GCL_extra" / "group_chosen_function.py"),
        file_check(args.shagcl_root / "SHA_GCL_extra" / "utils_funcion.py"),
        file_check(args.shagcl_root / "SHA_GCL_extra" / "kl_divergence.py"),
    ]
    vg_checks = [dir_check(vg_root)]
    for name in REQUIRED_VG_FILES:
        vg_checks.append(file_check(vg_root / name))

    ckpts = checkpoint_candidates(args.checkpoint_root)
    detector_check = file_check(detector)
    missing_source = [item["path"] for item in source_checks if not item["exists"]]
    missing_vg = [item["path"] for item in vg_checks if not item["exists"]]
    blockers = []
    if missing_source:
        blockers.append("missing_official_shagcl_source")
    if missing_vg:
        blockers.append("missing_shagcl_vg_inputs")
    if not detector_check["exists"]:
        blockers.append("missing_pretrained_detector_checkpoint")
    if not ckpts:
        blockers.append("missing_official_shagcl_checkpoint")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "official_repo": "https://github.com/dongxingning/SHA-GCL-for-SGG",
        "official_commit": git_commit(args.shagcl_root),
        "shagcl_root": str(args.shagcl_root),
        "vg_root": str(vg_root),
        "checkpoint_root": str(args.checkpoint_root),
        "source_checks": source_checks,
        "vg_checks": vg_checks,
        "pretrained_detector_check": detector_check,
        "checkpoint_candidates": ckpts,
        "missing_source_paths": missing_source,
        "missing_vg_paths": missing_vg,
        "official_protocol_summary": {
            "config": "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml",
            "predictor": "TransLike_GCL",
            "basic_encoder": "Hybrid-Attention",
            "dataset_choice": "VG",
            "group_split_mode": "divide4",
            "knowledge_transfer_mode": "KL_logit_TopDown",
            "predcls_overrides": [
                "MODEL.ROI_RELATION_HEAD.USE_GT_BOX True",
                "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True",
                "DTYPE float16",
            ],
            "public_checkpoint_note": (
                "README lists a VG PredCls model OneDrive link; other trained "
                "models require contacting the authors."
            ),
        },
        "next_action": (
            "Run official SHA-GCL evaluation and compare OpenSGG only after checkpoint/config/evaluator parity"
            if not blockers
            else "Provide missing VG inputs, pretrained detector, and trusted SHA-GCL checkpoint"
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
