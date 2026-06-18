#!/usr/bin/env python
"""Download the public SGB/Kaihua Motifs PredCls checkpoint bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


SGB_MOTIFS_BUNDLE_ID = "19n2Z8udljxgSpDwG8vPRWSEe76GzL9DM"
DEFAULT_OUTPUT = Path("outputs/pretrained/motifs/coldmanck/output.zip")


def run_gdown(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        file_id,
        "-O",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def extract_checkpoint(bundle: Path, member: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as zf:
        zf.extract(member, output_dir)
    return output_dir / member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--file-id", default=SGB_MOTIFS_BUNDLE_ID)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--member",
        default="output/motif-precls-exmp/model_0022000.pth",
        help="Checkpoint member to extract from the downloaded zip.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("outputs/pretrained/motifs/coldmanck/extracted"),
    )
    args = parser.parse_args()

    run_gdown(args.file_id, args.output)
    print(f"downloaded: {args.output} ({args.output.stat().st_size} bytes)")

    if args.extract:
        extracted = extract_checkpoint(args.output, args.member, args.extract_dir)
        print(f"extracted: {extracted}")


if __name__ == "__main__":
    main()
