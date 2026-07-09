#!/usr/bin/env python3
"""
Extract a small sample from the full VisualGenome dataset for CI smoke tests.

Usage:  python tools/make_sample_data.py [--n-images 10]

Output: data/VisualGenome_sample/
        ├── images/       (copies of selected JPEGs)
        ├── train.json    (COCO-style, filtered)
        ├── val.json      (same as train)
        ├── test.json     (same as train)
        └── rel.json      (relation triplets, filtered)

VG rel.json uses 0-based per-image object indices — no remapping needed.
We just keep the same images, same objects, same rels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VG_DIR = ROOT / "data" / "VisualGenome"
SAMPLE_DIR = ROOT / "data" / "VisualGenome_sample"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract VG sample dataset")
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--start-offset", type=int, default=0)
    args = parser.parse_args()

    print(f"Extracting {args.n_images} images from VG ...")

    with open(VG_DIR / "train.json") as f:
        train = json.load(f)
    with open(VG_DIR / "rel.json") as f:
        rel = json.load(f)

    # Index objs per image, pick those with rels
    ann_by_img: dict[int, list] = {}
    for a in train["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    train_rel_ids = {int(k) for k in rel["train"]}
    eligible = [img for img in train["images"]
                if img["id"] in train_rel_ids and img["id"] in ann_by_img]
    print(f"  {len(eligible)} eligible images")

    selected = eligible[args.start_offset:args.start_offset + args.n_images]
    sel_ids = {img["id"] for img in selected}
    print(f"  Selected: {[img['id'] for img in selected]}")

    # ---- Copy images ----
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "images").mkdir(exist_ok=True)
    copied = 0
    for img in selected:
        src = VG_DIR / "images" / img["file_name"]
        dst = SAMPLE_DIR / "images" / img["file_name"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} images")

    # ---- Filter annotations (keep original IDs) ----
    sample_anns = [a for a in train["annotations"] if a["image_id"] in sel_ids]

    # ---- Filter relations (same images → same rels for all splits) ----
    rel_data: dict[str, list] = {}
    for img in selected:
        iid = img["id"]
        triplets = rel.get("train", {}).get(str(iid),
                     rel.get("train", {}).get(iid, []))[:5]
        if triplets:
            rel_data[str(iid)] = triplets
    sample_rel = {"rel_categories": rel["rel_categories"],
                  "train": rel_data, "val": rel_data, "test": rel_data}

    # ---- Write ----
    coco = {"images": selected, "annotations": sample_anns,
            "categories": train["categories"]}
    for fname in ("train.json", "val.json", "test.json"):
        (SAMPLE_DIR / fname).write_text(json.dumps(coco))
    (SAMPLE_DIR / "rel.json").write_text(json.dumps(sample_rel))

    total_kb = sum(f.stat().st_size for f in SAMPLE_DIR.rglob("*") if f.is_file()) / 1024
    print(f"  Wrote sample: {len(selected)} images, {len(sample_anns)} anns, "
          f"{total_kb:.0f} KB total")
    print(f"  Ready at: {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    exit(main())
