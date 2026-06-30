#!/usr/bin/env python3
"""Check whether official SGB inputs exist for FREQ prior export.

The SGB reference builds ``statistics['pred_dist']`` from Visual Genome roidb
files. This script verifies those inputs before an agent attempts to export or
compare the official prior. It does not modify data or evaluator semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "VG-SGG-with-attri.h5",
    "VG-SGG-dicts-with-attri.json",
    "image_data.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgb-root",
        type=Path,
        default=Path("/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch"),
        help="Local Scene-Graph-Benchmark.pytorch checkout",
    )
    parser.add_argument(
        "--vg-root",
        type=Path,
        default=None,
        help="Directory containing SGB-format VG files; defaults to <sgb-root>/datasets/vg",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="SGB relation config used for the target PredCls setup",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vg_root = args.vg_root or args.sgb_root / "datasets" / "vg"
    config = (
        args.config or args.sgb_root / "configs" / "e2e_relation_X_101_32_8_FPN_1x.yaml"
    )
    paths_catalog = args.sgb_root / "maskrcnn_benchmark" / "config" / "paths_catalog.py"

    checks = []
    for path in (args.sgb_root, paths_catalog, config, vg_root):
        checks.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "type": "directory" if path.is_dir() else "file",
            }
        )
    for name in REQUIRED_FILES:
        path = vg_root / name
        checks.append({"path": str(path), "exists": path.exists(), "type": "file"})

    missing = [item["path"] for item in checks if not item["exists"]]
    report = {
        "status": "PASS" if not missing else "BLOCKED_MISSING_SGB_VG_INPUTS",
        "sgb_root": str(args.sgb_root),
        "vg_root": str(vg_root),
        "config": str(config),
        "required_files": list(REQUIRED_FILES),
        "missing": missing,
        "checks": checks,
        "next_action": (
            "Run SGB get_dataset_statistics and compare pred_dist"
            if not missing
            else "Provide missing SGB-format Visual Genome files before exporting pred_dist"
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
