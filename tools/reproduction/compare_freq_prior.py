#!/usr/bin/env python3
"""Compare official SGB frequency statistics with OpenSGG PairFrequencyBias.

This is an audit tool, not a reproduction result generator. It checks whether
the OpenSGG FREQ prior table matches an exported Scene-Graph-Benchmark
``statistics['pred_dist']`` tensor under the same class layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.motifs import PairFrequencyBias  # noqa: E402


def load_sgb_pred_dist(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "pred_dist" in payload:
        pred_dist = payload["pred_dist"]
    elif isinstance(payload, dict) and "statistics" in payload:
        pred_dist = payload["statistics"]["pred_dist"]
    else:
        raise ValueError(
            f"{path} does not look like an SGB statistics export with pred_dist"
        )
    pred_dist = torch.as_tensor(pred_dist).float()
    if pred_dist.dim() != 3:
        raise ValueError(
            f"pred_dist must be rank-3, got shape {tuple(pred_dist.shape)}"
        )
    return pred_dist


def build_opensgg_prior(
    data_root: Path,
    num_objects: int,
    num_predicates: int,
    eps: float,
    predicate_bg_index: str | None,
) -> torch.Tensor:
    bias = PairFrequencyBias(
        num_objects=num_objects,
        num_predicates=num_predicates,
        eps=eps,
        data_root=str(data_root),
        predicate_bg_index=predicate_bg_index,
    )
    return (
        bias.obj_baseline.weight.detach()
        .cpu()
        .view(num_objects, num_objects, num_predicates)
    )


def top_differences(
    diff: torch.Tensor,
    limit: int,
) -> list[dict[str, Any]]:
    flat = diff.abs().reshape(-1)
    k = min(limit, flat.numel())
    if k == 0:
        return []
    values, indices = torch.topk(flat, k)
    rows: list[dict[str, Any]] = []
    _, num_objects, num_predicates = diff.shape
    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        subj = flat_idx // (num_objects * num_predicates)
        rem = flat_idx % (num_objects * num_predicates)
        obj = rem // num_predicates
        pred = rem % num_predicates
        rows.append(
            {
                "subject_index": int(subj),
                "object_index": int(obj),
                "predicate_index": int(pred),
                "abs_difference": float(value),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgb-statistics",
        required=True,
        type=Path,
        help="Torch file containing Scene-Graph-Benchmark statistics['pred_dist']",
    )
    parser.add_argument(
        "--opensgg-data-root",
        default=PROJECT_ROOT / "data" / "VisualGenome",
        type=Path,
        help="OpenSGG VisualGenome data root containing train.json and rel.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument(
        "--predicate-bg-index", default="first", choices=["first", "last", "none"]
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    sgb = load_sgb_pred_dist(args.sgb_statistics)
    num_objects, _, num_predicates = sgb.shape
    bg_index = None if args.predicate_bg_index == "none" else args.predicate_bg_index
    opensgg = build_opensgg_prior(
        args.opensgg_data_root,
        num_objects=num_objects,
        num_predicates=num_predicates,
        eps=args.eps,
        predicate_bg_index=bg_index,
    )

    if tuple(opensgg.shape) != tuple(sgb.shape):
        raise ValueError(
            f"shape mismatch: OpenSGG {tuple(opensgg.shape)} vs SGB {tuple(sgb.shape)}"
        )

    diff = opensgg - sgb
    abs_diff = diff.abs()
    report = {
        "status": "PASS" if int((abs_diff > args.tolerance).sum()) == 0 else "MISMATCH",
        "sgb_statistics": str(args.sgb_statistics),
        "opensgg_data_root": str(args.opensgg_data_root),
        "num_objects": int(num_objects),
        "num_predicates": int(num_predicates),
        "predicate_bg_index": bg_index,
        "eps": args.eps,
        "tolerance": args.tolerance,
        "max_abs_difference": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs_difference": float(abs_diff.mean().item())
        if abs_diff.numel()
        else 0.0,
        "num_entries": int(abs_diff.numel()),
        "num_entries_above_tolerance": int((abs_diff > args.tolerance).sum().item()),
        "top_differences": top_differences(diff, args.topk),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
