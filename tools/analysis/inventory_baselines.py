#!/usr/bin/env python3
"""Inventory available checkpoints and baseline reproduction status.

Usage:
    python tools/analysis/inventory_baselines.py
    python tools/analysis/inventory_baselines.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PRETRAINED_DIR = _PROJECT_ROOT / "outputs" / "pretrained"
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "analysis" / "baseline_reproduction"

# Known methods and their expected checkpoints
CANDIDATE_METHODS = [
    {
        "method": "Motifs",
        "config": "configs/VisualGenome/Motifs.py",
        "known_checkpoints": [
            "outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth",
        ],
        "predcls_compatible": True,
        "notes": "Classic SGG baseline with strong context bias",
    },
    {
        "method": "VCTree",
        "config": "configs/VisualGenome/VCTree.py",
        "known_checkpoints": [],
        "predcls_compatible": True,
        "notes": "Tree-structured context model",
    },
    {
        "method": "Transformer",
        "config": "configs/VisualGenome/Transformer.py",
        "known_checkpoints": [],
        "predcls_compatible": True,
        "notes": "Attention-based SGG model",
    },
    {
        "method": "TDE",
        "config": "configs/VisualGenome/TDE.py",
        "known_checkpoints": [],
        "predcls_compatible": False,
        "notes": "Debiasing model; may require causal statistics. Do not assume Motifs checkpoint compatibility.",
    },
]


def scan_checkpoints():
    """Scan pretrained directory for available checkpoints."""
    available = []
    if not PRETRAINED_DIR.exists():
        return available

    for root, dirs, files in os.walk(PRETRAINED_DIR):
        for f in files:
            if f.endswith(('.ckpt', '.pth', '.pt')):
                full_path = Path(root) / f
                size_mb = full_path.stat().st_size / (1024 * 1024)
                available.append({
                    "path": str(full_path.relative_to(_PROJECT_ROOT)),
                    "size_mb": round(size_mb, 1),
                    "filename": f,
                })
    return available


def check_checkpoint_format(ckpt_path):
    """Check if checkpoint exists and estimate its format from extension."""
    full_path = _PROJECT_ROOT / ckpt_path
    if not full_path.exists():
        return "not_found", None

    ext = full_path.suffix
    if ext == '.ckpt':
        return "lightning_ckpt", None
    elif ext in ('.pth', '.pt'):
        return "pytorch_state_dict", None
    else:
        return f"unknown:{ext}", None


def generate_baseline_matrix(methods, checkpoints):
    """Generate baseline matrix CSV."""
    lines = []
    lines.append("method,config_exists,checkpoint_found,checkpoint_path,"
                 "checkpoint_format,predcls_compatible,smoke_tested,status,notes")

    for m in methods:
        config_exists = os.path.exists(str(_PROJECT_ROOT / m["config"]))

        # Find matching checkpoints
        matching = [c for c in checkpoints
                    if m["method"].lower() in c["path"].lower()]

        if matching:
            ckpt_path = matching[0]["path"]
            fmt, keys = check_checkpoint_format(ckpt_path)
            ckpt_found = True
            ckpt_format = fmt
        elif m["known_checkpoints"]:
            # Check known paths
            found = False
            for known in m["known_checkpoints"]:
                if os.path.exists(str(_PROJECT_ROOT / known)):
                    fmt, keys = check_checkpoint_format(known)
                    ckpt_path = known
                    ckpt_format = fmt
                    found = True
                    break
            if not found:
                ckpt_path = ""
                ckpt_format = ""
                ckpt_found = False
        else:
            ckpt_path = ""
            ckpt_format = ""
            ckpt_found = False

        smoke_tested = "no"
        if m["method"] == "Motifs" and ckpt_found:
            status = "ready_for_smoke"
        elif ckpt_found:
            status = "ready_for_smoke"
        elif m["predcls_compatible"]:
            status = "blocked:no_checkpoint"
        else:
            status = "blocked:incompatible_or_no_checkpoint"

        lines.append(
            f"{m['method']},{config_exists},{ckpt_found},{ckpt_path},"
            f"{ckpt_format},{m['predcls_compatible']},{smoke_tested},"
            f"{status},{m['notes']}"
        )

    return "\n".join(lines) + "\n"


def generate_reproduction_commands():
    """Generate documented reproduction commands."""
    lines = []
    lines.append("# Baseline Reproduction Commands")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("## Motifs PredCLS Smoke Test")
    lines.append("")
    lines.append("### Prerequisites")
    lines.append("- Checkpoint: `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth`")
    lines.append("- Config: `configs/VisualGenome/Motifs.py`")
    lines.append("- Dataname: `VisualGenome`")
    lines.append("")

    lines.append("### Small-sample smoke test (20 samples)")
    lines.append("```bash")
    lines.append("python train.py --test \\")
    lines.append("  --method motifs \\")
    lines.append("  --dataname VisualGenome \\")
    lines.append("  --ckpt_path outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth \\")
    lines.append("  --test_dataset_size 20 \\")
    lines.append("  --num_workers 0 \\")
    lines.append("  --gpus 0 \\")
    lines.append("  --eval_mode predcls")
    lines.append("```")
    lines.append("")

    lines.append("### Full test evaluation")
    lines.append("```bash")
    lines.append("python train.py --test \\")
    lines.append("  --method motifs \\")
    lines.append("  --dataname VisualGenome \\")
    lines.append("  --ckpt_path outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth \\")
    lines.append("  --num_workers 4 \\")
    lines.append("  --gpus 0 \\")
    lines.append("  --eval_mode predcls")
    lines.append("```")
    lines.append("")

    lines.append("### Expected output")
    lines.append("- Metrics saved to: `outputs/runs/<run_dir>/eval/predcls/metrics.json`")
    lines.append("- Expected metric keys: `predcls_R@10`, `predcls_R@20`, `predcls_mR@10`, `predcls_mR@20`")
    lines.append("")

    lines.append("## Other Methods (pending checkpoint availability)")
    lines.append("")
    lines.append("### VCTree")
    lines.append("```bash")
    lines.append("# python train.py --test --method vctree --dataname VisualGenome --ckpt_path <path> --eval_mode predcls --gpus 0")
    lines.append("```")
    lines.append("")
    lines.append("### Transformer")
    lines.append("```bash")
    lines.append("# python train.py --test --method transformer --dataname VisualGenome --ckpt_path <path> --eval_mode predcls --gpus 0")
    lines.append("```")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_checkpoint_inventory(checkpoints):
    """Generate checkpoint inventory markdown."""
    lines = []
    lines.append("# Checkpoint Inventory")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total checkpoints found: {len(checkpoints)}")
    lines.append("")

    if checkpoints:
        lines.append("| Path | Size (MB) | Format |")
        lines.append("|---|---|")
        for c in checkpoints:
            fmt, _ = check_checkpoint_format(c["path"])
            lines.append(f"| {c['path']} | {c['size_mb']} | {fmt} |")
    else:
        lines.append("No checkpoints found under outputs/pretrained/")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Inventory baselines and checkpoints for reproduction"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate inventory without attempting smoke test")
    args = parser.parse_args()

    print("=" * 70)
    print("Baseline Reproduction Inventory")
    print("=" * 70)
    print()

    # 1. Scan checkpoints
    print("[1/4] Scanning checkpoints...")
    checkpoints = scan_checkpoints()
    print(f"      Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        fmt, _ = check_checkpoint_format(c["path"])
        print(f"        {c['path']} ({c['size_mb']} MB, format={fmt})")
    print()

    # 2. Generate baseline matrix
    print("[2/4] Generating baseline matrix...")
    matrix_csv = generate_baseline_matrix(CANDIDATE_METHODS, checkpoints)
    print(f"      {len(CANDIDATE_METHODS)} methods inventoried")

    # 3. Generate commands
    print("[3/4] Generating reproduction commands...")
    commands_md = generate_reproduction_commands()

    # 4. Generate inventory
    inventory_md = generate_checkpoint_inventory(checkpoints)
    print()

    # Write outputs
    print("[4/4] Writing outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline matrix CSV
    matrix_path = OUTPUT_DIR / "baseline_matrix.csv"
    with open(matrix_path, "w") as f:
        f.write(matrix_csv)
    print(f"      Wrote: {matrix_path}")

    # Commands markdown
    commands_path = OUTPUT_DIR / "reproduction_commands.md"
    with open(commands_path, "w") as f:
        f.write(commands_md)
    print(f"      Wrote: {commands_path}")

    # Checkpoint inventory
    inventory_path = OUTPUT_DIR / "checkpoint_inventory.md"
    with open(inventory_path, "w") as f:
        f.write(inventory_md)
    print(f"      Wrote: {inventory_path}")

    # Overall report
    report_lines = []
    report_lines.append("# Baseline Reproduction Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- Total checkpoints available: {len(checkpoints)}")
    report_lines.append(f"- Methods inventoried: {len(CANDIDATE_METHODS)}")
    report_lines.append("")

    # Method status
    for m in CANDIDATE_METHODS:
        has_ckpt = any(m["method"].lower() in c["path"].lower()
                       for c in checkpoints) or any(
            os.path.exists(str(_PROJECT_ROOT / k))
            for k in m["known_checkpoints"])
        status = "READY" if has_ckpt and m["predcls_compatible"] else "BLOCKED"
        report_lines.append(f"- **{m['method']}**: {status} — {m['notes']}")

    report_lines.append("")
    report_lines.append("## Motifs PredCLS Smoke Test")
    report_lines.append("")
    report_lines.append("Checkpoint: `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth`")
    report_lines.append("")
    report_lines.append("Command noted in `reproduction_commands.md`. Actual execution")
    report_lines.append("requires the full PyTorch Lightning environment and may need")
    report_lines.append("GPU or CPU-only settings adjusted for the local machine.")
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("")
    report_lines.append("1. Run Motifs PredCLS smoke test with `--test_dataset_size 20`")
    report_lines.append("2. Verify metrics appear in `outputs/runs/.../eval/predcls/metrics.json`")
    report_lines.append("3. If successful, proceed to full evaluation for collapse analysis")
    report_lines.append("4. Locate or train VCTree/Transformer checkpoints for cross-model validation")

    report_path = OUTPUT_DIR / "baseline_reproduction_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"      Wrote: {report_path}")

    print()
    print("Done. Run Motifs smoke test with command in reproduction_commands.md")


if __name__ == "__main__":
    main()
