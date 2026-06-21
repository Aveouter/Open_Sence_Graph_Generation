#!/usr/bin/env python3
"""Inventory available checkpoints and baseline reproduction status.

Usage:
    python tools/analysis/inventory_baselines.py
    python tools/analysis/inventory_baselines.py --dry-run
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
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


def find_method_checkpoint(method, checkpoints):
    """Return the best known checkpoint path for a method, or an empty string."""
    matching = [
        c for c in checkpoints
        if method["method"].lower() in c["path"].lower()
    ]
    if matching:
        return matching[0]["path"]
    for known in method["known_checkpoints"]:
        if os.path.exists(str(_PROJECT_ROOT / known)):
            return known
    return ""


def build_motifs_smoke_command(ckpt_path, test_dataset_size=20,
                               device="cuda", num_workers=0):
    command = [
        sys.executable,
        "train.py",
        "--test",
        "--method", "Motifs",
        "--dataname", "VisualGenome",
        "--ckpt_path", ckpt_path,
        "--test_dataset_size", str(test_dataset_size),
        "--val_batch_size", "1",
        "--num_workers", str(num_workers),
        "--eval_mode", "predcls",
        "--no_display_method_info",
    ]
    if device:
        command.extend(["--device", device])
    if device == "cuda":
        command.extend(["--gpus", "0"])
    return command


def smoke_env_overrides(device):
    if device == "cpu":
        return {"CUDA_VISIBLE_DEVICES": ""}
    return {}


def format_shell_command(command, env_overrides=None):
    prefixes = []
    for key, value in (env_overrides or {}).items():
        prefixes.append(f"{key}={shlex.quote(value)}")
    return " ".join(prefixes + [shlex.quote(part) for part in command])


def run_motifs_smoke_test(ckpt_path, output_dir, test_dataset_size,
                          device, timeout_sec):
    """Run Motifs PredCLS smoke test and capture a small execution record."""
    command = build_motifs_smoke_command(
        ckpt_path,
        test_dataset_size=test_dataset_size,
        device=device,
        num_workers=0,
    )
    env_overrides = smoke_env_overrides(device)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = output_dir / "motifs_predcls_smoke_test.log"
    started_at = datetime.now().isoformat()
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_sec,
            check=False,
        )
        output = proc.stdout or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        returncode = None
        timed_out = True

    duration_sec = round(time.time() - start, 2)
    log_path.write_text(output)
    status = "passed" if returncode == 0 and not timed_out else "failed"
    if timed_out:
        status = "timeout"

    return {
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_sec": duration_sec,
        "started_at": started_at,
        "command": command,
        "command_text": format_shell_command(command, env_overrides),
        "log_path": str(log_path.relative_to(_PROJECT_ROOT)),
        "output_tail": "\n".join(output.splitlines()[-40:]),
    }


def generate_baseline_matrix(methods, checkpoints, smoke_result=None):
    """Generate baseline matrix CSV."""
    lines = []
    lines.append("method,config_exists,checkpoint_found,checkpoint_path,"
                 "checkpoint_format,predcls_compatible,smoke_tested,status,notes")

    for m in methods:
        config_exists = os.path.exists(str(_PROJECT_ROOT / m["config"]))

        ckpt_path = find_method_checkpoint(m, checkpoints)
        if ckpt_path:
            fmt, keys = check_checkpoint_format(ckpt_path)
            ckpt_found = True
            ckpt_format = fmt
        else:
            ckpt_path = ""
            ckpt_format = ""
            ckpt_found = False

        smoke_tested = "no"
        if m["method"] == "Motifs" and smoke_result:
            smoke_tested = "yes"
            status = f"smoke_{smoke_result['status']}"
        elif m["method"] == "Motifs" and ckpt_found:
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


def generate_reproduction_commands(smoke_command=None, smoke_env=None):
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
    lines.append("python train.py \\")
    lines.append("  --test \\")
    lines.append("  --method Motifs \\")
    lines.append("  --dataname VisualGenome \\")
    lines.append("  --ckpt_path outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth \\")
    lines.append("  --test_dataset_size 20 \\")
    lines.append("  --val_batch_size 1 \\")
    lines.append("  --num_workers 0 \\")
    lines.append("  --device cuda \\")
    lines.append("  --gpus 0 \\")
    lines.append("  --eval_mode predcls \\")
    lines.append("  --no_display_method_info")
    lines.append("```")
    lines.append("")

    if smoke_command:
        lines.append("### Last smoke command generated by this tool")
        lines.append("```bash")
        lines.append(format_shell_command(smoke_command, smoke_env))
        lines.append("```")
        lines.append("")

    lines.append("### Full test evaluation")
    lines.append("```bash")
    lines.append("python train.py \\")
    lines.append("  --test \\")
    lines.append("  --method Motifs \\")
    lines.append("  --dataname VisualGenome \\")
    lines.append("  --ckpt_path outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth \\")
    lines.append("  --val_batch_size 1 \\")
    lines.append("  --num_workers 4 \\")
    lines.append("  --device cuda \\")
    lines.append("  --gpus 0 \\")
    lines.append("  --eval_mode predcls \\")
    lines.append("  --no_display_method_info")
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
    lines.append("# python train.py --test --method VCTree --dataname VisualGenome --ckpt_path <path> --eval_mode predcls --gpus 0")
    lines.append("```")
    lines.append("")
    lines.append("### Transformer")
    lines.append("```bash")
    lines.append("# python train.py --test --method Transformer --dataname VisualGenome --ckpt_path <path> --eval_mode predcls --gpus 0")
    lines.append("```")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_motifs_smoke_report(smoke_result, smoke_command, ckpt_path,
                                 smoke_env=None):
    lines = []
    lines.append("# Motifs PredCLS Smoke Test")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append(f"- Checkpoint: `{ckpt_path}`")
    lines.append(f"- Status: `{smoke_result['status'] if smoke_result else 'not_run'}`")
    if smoke_result:
        lines.append(f"- Return code: `{smoke_result['returncode']}`")
        lines.append(f"- Duration: `{smoke_result['duration_sec']}s`")
        lines.append(f"- Log path: `{smoke_result['log_path']}`")
    lines.append("")
    lines.append("## Command")
    lines.append("")
    lines.append("```bash")
    lines.append(
        smoke_result["command_text"] if smoke_result
        else format_shell_command(smoke_command, smoke_env)
    )
    lines.append("```")
    lines.append("")
    if smoke_result:
        lines.append("## Output Tail")
        lines.append("")
        lines.append("```text")
        lines.append(smoke_result["output_tail"])
        lines.append("```")
        lines.append("")
    else:
        lines.append("Smoke execution was not requested. Run this script with `--run_motifs_smoke`.")
        lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--run_motifs_smoke", action="store_true",
                        help="Actually run a Motifs PredCLS smoke test")
    parser.add_argument("--smoke_test_size", type=int, default=20,
                        help="Number of test samples for Motifs smoke test")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for the optional smoke test")
    parser.add_argument("--timeout_sec", type=int, default=1800,
                        help="Timeout for the optional smoke test")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for generated reports")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir

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

    motifs = next(m for m in CANDIDATE_METHODS if m["method"] == "Motifs")
    motifs_ckpt = find_method_checkpoint(motifs, checkpoints)
    smoke_command = build_motifs_smoke_command(
        motifs_ckpt or "outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth",
        test_dataset_size=args.smoke_test_size,
        device=args.device,
        num_workers=0,
    )
    smoke_env = smoke_env_overrides(args.device)
    smoke_result = None

    if args.run_motifs_smoke:
        if not motifs_ckpt:
            print("      ERROR: Motifs checkpoint not found; cannot run smoke test")
            sys.exit(1)
        if args.dry_run:
            print("      Dry-run: smoke command would be:")
            print(f"        {format_shell_command(smoke_command, smoke_env)}")
        else:
            print("[2/5] Running Motifs PredCLS smoke test...")
            output_dir.mkdir(parents=True, exist_ok=True)
            smoke_result = run_motifs_smoke_test(
                motifs_ckpt,
                output_dir,
                args.smoke_test_size,
                args.device,
                args.timeout_sec,
            )
            print(f"      Smoke status: {smoke_result['status']}")
            print(f"      Log: {smoke_result['log_path']}")
            print()

    # 2. Generate baseline matrix
    print("[2/4] Generating baseline matrix...")
    matrix_csv = generate_baseline_matrix(
        CANDIDATE_METHODS, checkpoints, smoke_result=smoke_result)
    print(f"      {len(CANDIDATE_METHODS)} methods inventoried")

    # 3. Generate commands
    print("[3/4] Generating reproduction commands...")
    commands_md = generate_reproduction_commands(
        smoke_command=smoke_command,
        smoke_env=smoke_env,
    )

    # 4. Generate inventory
    inventory_md = generate_checkpoint_inventory(checkpoints)
    print()

    # Write outputs
    if args.dry_run:
        print("[4/4] Dry-run preview; no files written.")
        print()
        print("Motifs smoke command:")
        print(format_shell_command(smoke_command, smoke_env))
        print()
        print(matrix_csv)
        print("Done.")
        return

    print("[4/4] Writing outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline matrix CSV
    matrix_path = output_dir / "baseline_matrix.csv"
    with open(matrix_path, "w") as f:
        f.write(matrix_csv)
    print(f"      Wrote: {matrix_path}")

    # Commands markdown
    commands_path = output_dir / "reproduction_commands.md"
    with open(commands_path, "w") as f:
        f.write(commands_md)
    print(f"      Wrote: {commands_path}")

    # Checkpoint inventory
    inventory_path = output_dir / "checkpoint_inventory.md"
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
        has_ckpt = bool(find_method_checkpoint(m, checkpoints))
        status = "READY" if has_ckpt and m["predcls_compatible"] else "BLOCKED"
        if m["method"] == "Motifs" and smoke_result:
            status = f"SMOKE_{smoke_result['status'].upper()}"
        report_lines.append(f"- **{m['method']}**: {status} — {m['notes']}")

    report_lines.append("")
    report_lines.append("## Motifs PredCLS Smoke Test")
    report_lines.append("")
    report_lines.append("Checkpoint: `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth`")
    report_lines.append("")
    if smoke_result:
        report_lines.append(f"Smoke status: `{smoke_result['status']}`")
        report_lines.append(f"Log path: `{smoke_result['log_path']}`")
    else:
        report_lines.append("Command noted in `reproduction_commands.md`; smoke execution was not requested.")
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("")
    report_lines.append("1. Run Motifs PredCLS smoke test with `--test_dataset_size 20`")
    report_lines.append("2. Verify metrics appear in `outputs/runs/.../eval/predcls/metrics.json`")
    report_lines.append("3. If successful, proceed to full evaluation for collapse analysis")
    report_lines.append("4. Locate or train VCTree/Transformer checkpoints for cross-model validation")

    smoke_path = output_dir / "motifs_predcls_smoke_test.md"
    with open(smoke_path, "w") as f:
        f.write(generate_motifs_smoke_report(
            smoke_result, smoke_command,
            motifs_ckpt or "outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth",
            smoke_env=smoke_env,
        ))
    print(f"      Wrote: {smoke_path}")

    report_path = output_dir / "baseline_reproduction_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"      Wrote: {report_path}")

    print()
    print("Done. Run Motifs smoke test with command in reproduction_commands.md")


if __name__ == "__main__":
    main()
