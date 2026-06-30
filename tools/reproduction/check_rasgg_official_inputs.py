#!/usr/bin/env python3
"""Check official RA-SGG/ReTAG reproduction inputs."""

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

CHECKPOINT_PROTOCOLS = ("predcls", "sgcls", "sgdet")


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


def candidates(root: Path, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    suffixes = {".pth", ".pt", ".pkl", ".ckpt", ".npy", ".npz"}
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() not in suffixes and not any(k in name for k in keywords):
            continue
        out.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rasgg-root",
        type=Path,
        default=Path("/workspace/external/ra_sgg_official/torch-rasgg"),
    )
    parser.add_argument(
        "--vg-root",
        type=Path,
        default=None,
        help="Official VG root; defaults to <rasgg-root>/datasets/vg",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/pretrained/ra_sgg_official"),
    )
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=Path("outputs/pretrained/ra_sgg_official/featurebank"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.rasgg_root / "datasets" / "vg"
    relation_head = (
        args.rasgg_root
        / "maskrcnn_benchmark"
        / "modeling"
        / "roi_heads"
        / "relation_head"
    )
    source_checks = [
        dir_check(args.rasgg_root),
        file_check(args.rasgg_root / "README.md"),
        file_check(args.rasgg_root / "Model_Zoo.md"),
        file_check(args.rasgg_root / "configs" / "e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml"),
        file_check(args.rasgg_root / "scripts" / "predcls_train_retag.sh"),
        file_check(args.rasgg_root / "scripts" / "sgcls_train_retag.sh"),
        file_check(args.rasgg_root / "scripts" / "sgdet_train_retag.sh"),
        file_check(args.rasgg_root / "scripts" / "test.sh"),
        file_check(args.rasgg_root / "tools" / "relation_memory_extractor.py"),
        file_check(relation_head / "roi_relation_predictors.py"),
    ]

    vg_checks = [dir_check(vg_root)]
    for name in REQUIRED_VG_FILES:
        vg_checks.append(file_check(vg_root / name))

    retag_checkpoint_checks = {
        protocol: candidates(args.checkpoint_root / protocol, ("model", "retag", "rasgg"))
        for protocol in CHECKPOINT_PROTOCOLS
    }
    penet_checkpoint_checks = {
        protocol: candidates(
            args.rasgg_root / "checkpoints" / f"PE-NET_{protocol.capitalize()}",
            ("model",),
        )
        for protocol in CHECKPOINT_PROTOCOLS
    }
    memory_candidates = candidates(args.memory_root, ("feature", "memory", "fb", "bank"))

    missing_source = [item["path"] for item in source_checks if not item["exists"]]
    missing_vg = [item["path"] for item in vg_checks if not item["exists"]]
    missing_retag_ckpts = [
        protocol for protocol, items in retag_checkpoint_checks.items() if not items
    ]
    missing_penet_ckpts = [
        protocol for protocol, items in penet_checkpoint_checks.items() if not items
    ]

    blockers = []
    if missing_source:
        blockers.append("missing_official_rasgg_source")
    if missing_vg:
        blockers.append("missing_rasgg_vg_inputs")
    if missing_retag_ckpts:
        blockers.append("missing_official_retag_checkpoints")
    if missing_penet_ckpts:
        blockers.append("missing_pretrained_penet_checkpoints")
    if not memory_candidates:
        blockers.append("missing_official_memory_bank_features")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "official_repo": "https://github.com/KanghoonYoon/torch-rasgg",
        "official_commit": git_commit(args.rasgg_root),
        "rasgg_root": str(args.rasgg_root),
        "vg_root": str(vg_root),
        "checkpoint_root": str(args.checkpoint_root),
        "memory_root": str(args.memory_root),
        "source_checks": source_checks,
        "vg_checks": vg_checks,
        "retag_checkpoint_checks": retag_checkpoint_checks,
        "penet_checkpoint_checks": penet_checkpoint_checks,
        "memory_candidates": memory_candidates,
        "missing_source_paths": missing_source,
        "missing_vg_paths": missing_vg,
        "missing_retag_checkpoint_protocols": missing_retag_ckpts,
        "missing_penet_checkpoint_protocols": missing_penet_ckpts,
        "official_protocol_summary": {
            "predcls_train_script": "scripts/predcls_train_retag.sh",
            "config": "configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml",
            "train_predictor": "ReTAGPENet",
            "config_predictor_example": "RA-PENetCorrectProtoBetaEnhanceBG",
            "pretrained_penet": "checkpoints/PE-NET_PredCls/model_final.pth",
            "memory_bank_pattern": "featurebank/<mode>_bg_processed_fb_train_<MEMORY_SIZE>.npy",
            "predcls_overrides": [
                "TYPE retag",
                "MODEL.ROI_RELATION_HEAD.USE_GT_BOX True",
                "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True",
                "RASGG.NUM_RETRIEVALS 20",
                "RASGG.MEMORY_SIZE 8",
                "RASGG.THRESHOLD 0.3",
                "RASGG.MIXUP True",
                "RASGG.MIXUP_ALPHA 20",
                "RASGG.MIXUP_BETA 5",
                "MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False",
                "DTYPE float32",
            ],
            "official_resource_note": (
                "README links Google Drive folders for pretrained PE-Net "
                "models, memory-bank features, and ReTAG trained models."
            ),
        },
        "next_action": (
            "Run official ReTAG evaluation and compare OpenSGG only after checkpoint/memory/evaluator parity"
            if not blockers
            else "Provide official VG inputs, pretrained PE-Net, ReTAG checkpoint, and memory-bank features"
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
