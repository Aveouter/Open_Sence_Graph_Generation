#!/usr/bin/env python3
"""EGTR GT-aligned relation export.

Runs EGTR SGDet inference on test images, converts per-query predictions to
GT-aligned relation scores via _egtr_to_compact(), and exports JSONL in the
same schema as the Motifs export so downstream collapse tools can consume it.

This is NOT Motifs-style PredCLS. EGTR runs full SGDet inference, then
post-hoc matches predicted queries to ground-truth boxes via IoU. The
resulting relation scores are aligned to GT relations, not model-internal
pair indices. Throughout all reports, call this "EGTR GT-aligned relation
export" or "EGTR post-hoc GT-aligned evaluation".

Usage:
    python tools/analysis/export_egtr_predictions.py --dry-run
    python tools/analysis/export_egtr_predictions.py \
        --ckpt_path <path> --test_dataset_size 50 --device cuda
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse shared utilities from Motifs export
from tools.analysis.export_relation_predictions import (
    load_predicate_names,
    load_object_class_names,
    predicate_list_checksum,
    validate_jsonl,
    write_jsonl,
    write_export_summary,
    write_schema_md,
    REQUIRED_FIELDS,
)

EGTR_CKPT_DEFAULT = (
    "outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr"
    "__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune"
    "__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002"
    "__visual_genome__finetune/version_0/checkpoints/"
    "epoch=03-validation_loss=1.71.ckpt"
)

SCHEMA_MD_TEMPLATE = """# EGTR GT-Aligned Relation Export Schema

## Important: this is NOT PredCLS
EGTR runs full SGDet inference (no GT boxes/labels as model input), then
post-hoc matches predicted DETR queries to ground-truth boxes via IoU in
`_egtr_to_compact()`. Relation scores come from `pred_rel[matched_query_s,
matched_query_o, :]` — 50 sigmoid probabilities per predicate.

## Differences from Motifs export
- Scores are sigmoid probabilities (0-1), not softmax.
- No background predicate channel; scores are always 50-dim.
- Matching is query-to-GT-box IoU, not model-internal pair indices.
- All emitted rows have `matched_pair_found=true` (one row per GT relation
  that was successfully matched; unmatched GT relations are counted but
  not emitted unless the relation count is zero for an image).

## Fields (same as Motifs JSONL)
See `export_relation_predictions.py` schema for field definitions.
"""


def _build_egtr_config(test_size, val_batch_size, num_workers, device):
    """Build config namespace for EGTR on VisualGenome."""
    from types import SimpleNamespace
    from utils.main_utils import load_config, update_config
    from utils.parser import create_parser, default_parser

    args = create_parser().parse_args([])
    config = args.__dict__
    config.update({
        "method": "EGTR",
        "dataname": "VisualGenome",
        "test": True,
        "test_dataset_size": int(test_size) if test_size and int(test_size) > 0 else None,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "device": device,
        "no_display_method_info": True,
        "ex_name": "egtr_relation_export",
        "overwrite": True,
        "egtr_sgdet_postprocess": "query",
    })

    cfg_path = _PROJECT_ROOT / "configs" / "VisualGenome" / "EGTR.py"
    loaded_cfg = load_config(str(cfg_path))
    config = update_config(
        config,
        loaded_cfg,
        exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"],
    )
    for key, value in default_parser().items():
        if config.get(key) is None:
            config[key] = value

    config.update({
        "method": "EGTR",
        "dataname": "VisualGenome",
        "test": True,
        "test_dataset_size": int(test_size) if test_size and int(test_size) > 0 else None,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "device": device,
        "no_display_method_info": True,
    })
    return SimpleNamespace(**config), config


def _gt_relations_from_image(target, predicate_names, object_names, image_idx):
    """Extract GT relation metadata from one image target dict.

    Returns list of dicts with subject/object/GT info (no scores yet).
    """
    import torch

    rel_annotations = target.get("rel_annotations")
    if rel_annotations is None:
        return []

    if torch.is_tensor(rel_annotations):
        rel_annotations = rel_annotations.detach().cpu()
    if rel_annotations.numel() == 0:
        return []

    if rel_annotations.dim() == 1:
        rel_annotations = rel_annotations.reshape(-1, 3)

    labels = target["labels"].detach().cpu().long()
    image_id = int(target.get("image_id", image_idx))

    gt_rows = []
    for rel in rel_annotations:
        subj_idx = int(rel[0].item())
        obj_idx = int(rel[1].item())
        gt_pred_id = int(rel[2].item())
        gt_pred_name = predicate_names[gt_pred_id]
        subj_cls = int(labels[subj_idx].item())
        obj_cls = int(labels[obj_idx].item())

        gt_rows.append({
            "image_id": image_id,
            "subject_idx": subj_idx,
            "object_idx": obj_idx,
            "subject_class_id": subj_cls,
            "object_class_id": obj_cls,
            "subject_class_name": object_names.get(subj_cls, f"class_{subj_cls}"),
            "object_class_name": object_names.get(obj_cls, f"class_{obj_cls}"),
            "gt_predicate_id": gt_pred_id,
            "gt_predicate_name": gt_pred_name,
        })

    return gt_rows


def _compact_to_export_rows(compact_list, gt_rows_by_image, predicate_names,
                            model, task, top_k):
    """Merge compact relation scores with GT metadata into JSONL rows.

    compact_list: list of per-image compact dicts from _egtr_to_compact()
    gt_rows_by_image: list of lists of per-image GT relation metadata dicts
    """
    rows = []
    total_gt = 0
    matched_count = 0
    unmatched_count = 0

    for image_idx, (compact, gt_rows) in enumerate(
        zip(compact_list, gt_rows_by_image)
    ):
        rel_scores = compact.get("rel_scores")
        if rel_scores is None or len(gt_rows) == 0:
            total_gt += len(gt_rows)
            unmatched_count += len(gt_rows)
            continue

        scores_all = np.asarray(rel_scores, dtype=np.float64)
        R = scores_all.shape[0]
        total_gt += len(gt_rows)

        for r_idx, gt in enumerate(gt_rows):
            if r_idx >= R:
                # Compact output has fewer rows than GT (unmatched)
                row = dict(gt)
                row.update({
                    "pred_predicate_id": -1,
                    "pred_predicate_name": None,
                    "predicate_scores_all": [],
                    "topk_predicate_ids": [],
                    "topk_predicate_names": [],
                    "topk_predicate_scores": [],
                    "matched_pair_found": False,
                    "model": model,
                    "task": task,
                })
                rows.append(row)
                unmatched_count += 1
                continue

            scores = scores_all[r_idx]
            # EGTR produces sigmoid probabilities (no softmax needed)
            topk_indices = np.argsort(-scores)[:top_k]
            # EGTR scores are 50-dim, index i → VG predicate i+1
            topk_vg_ids = [int(i) + 1 for i in topk_indices]
            topk_names = [predicate_names[pid] for pid in topk_vg_ids]
            topk_scores = [float(scores[i]) for i in topk_indices]
            pred_id = topk_vg_ids[0] if topk_vg_ids else -1
            pred_name = predicate_names[pred_id] if pred_id > 0 else None

            row = dict(gt)
            row.update({
                "pred_predicate_id": pred_id,
                "pred_predicate_name": pred_name,
                "predicate_scores_all": [float(s) for s in scores],
                "topk_predicate_ids": topk_vg_ids,
                "topk_predicate_names": topk_names,
                "topk_predicate_scores": topk_scores,
                "matched_pair_found": True,
                "model": model,
                "task": task,
            })
            rows.append(row)
            matched_count += 1

    return rows, total_gt, matched_count, unmatched_count


def main():
    parser = argparse.ArgumentParser(
        description="EGTR GT-aligned relation export"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate format without running inference")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to EGTR Lightning checkpoint")
    parser.add_argument("--test_dataset_size", type=int, default=50,
                        help="Limit test dataset size (default: 50)")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help="Evaluation batch size (default: 1)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available else cpu)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Hard cap on batches processed")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top predictions to store")
    parser.add_argument("--task", type=str, default="GT-aligned",
                        help="Task label for export metadata")
    parser.add_argument("--model", type=str, default="EGTR",
                        help="Model label for export metadata")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    model = args.model
    task = args.task

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (_PROJECT_ROOT / "outputs" / "analysis" / "strong_go" /
                      "EGTR_sanity" / "PredCLS")

    jsonl_path = output_dir / "relation_predictions.jsonl"
    summary_path = output_dir / "export_summary.json"
    schema_path = output_dir / "export_schema.md"

    print("=" * 70)
    print("EGTR GT-Aligned Relation Export")
    print("=" * 70)
    print(f"  Model:       {model}")
    print(f"  Task:        {task}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load predicate and object names
    print("[1/4] Loading VG150 names...")
    predicate_names = load_predicate_names()
    object_names = load_object_class_names()
    print(f"      Predicates: {len(predicate_names)} (including background)")
    print(f"      Object classes: {len(object_names)}")
    print()

    if args.dry_run:
        print("[2/4] Dry-run: format validation only")
        print("      (no checkpoint or inference)")
        rows = []
        summary = {
            "total_gt_relations": 0,
            "exported_rows": 0,
            "matched_rows": 0,
            "unmatched_rows": 0,
            "unmatched_ratio": 0.0,
            "score_dimension": 50,
            "predicate_names_sha256": predicate_list_checksum(predicate_names),
            "errors": 0,
            "warnings": 0,
            "mode": "dry_run",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_schema_md(schema_path)
        write_export_summary(summary, summary_path)
        print("      Done (dry-run).")
        print()
        return

    if not args.ckpt_path:
        print("[2/4] ERROR: --ckpt_path is required (or use --dry-run)")
        sys.exit(1)

    # Real export
    print("[2/4] Running EGTR inference with GT-aligned compact conversion...")
    print(f"      ckpt_path: {args.ckpt_path}")

    import torch
    from src.exp import BaseExperiment
    from src.methods import method_maps
    from utils.main_utils import get_dataset

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("      WARNING: CUDA unavailable; falling back to CPU")
        device = "cpu"

    args_ns, config = _build_egtr_config(
        test_size=args.test_dataset_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"      device: {device}")
    print(f"      test_dataset_size: {config.get('test_dataset_size')}")
    print(f"      val_batch_size: {config.get('val_batch_size')}")

    print("      Loading VisualGenome test loader...")
    _, _, test_loader = get_dataset("VisualGenome", config)

    print("      Building EGTR model...")
    method_cls = method_maps["egtr"]
    method = method_cls(
        steps_per_epoch=1,
        save_dir=str(output_dir),
        **config,
    )

    print("      Loading checkpoint weights...")
    state_dict = BaseExperiment._load_checkpoint_state_dict(args.ckpt_path)
    BaseExperiment._adapt_state_dict(state_dict, method.model)

    torch_device = torch.device(device)
    method.to(torch_device)
    method.eval()

    rows = []
    total_gt_sum = 0
    matched_sum = 0
    unmatched_sum = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed_batches >= args.max_batches:
                break

            images, targets_orig = method._split_batch(batch)
            targets_orig = method._move_targets_to_device(targets_orig)

            # Keep original targets for GT metadata and compact conversion
            # (which needs rel_annotations, labels, boxes, orig_size)

            # Adapt targets for EGTR forward pass
            # (model expects class_labels and dense rel matrix)
            targets_adapted = method._adapt_targets(targets_orig)

            # EGTR forward pass (SGDet inference with adapted targets)
            result = method.forward(images, targets_adapted)
            outputs = result.get("outputs", {})

            # Extract GT relation metadata from original targets
            gt_rows_by_image = []
            for i, tgt in enumerate(targets_orig):
                gt_rows_by_image.append(
                    _gt_relations_from_image(tgt, predicate_names, object_names, i)
                )

            # Convert to compact per-relation format using ORIGINAL targets
            # (_egtr_to_compact accesses rel_annotations, labels, boxes)
            compact_list = method._egtr_to_compact(outputs, targets_orig)

            # Merge compact scores with GT metadata
            batch_rows, tg, mc, um = _compact_to_export_rows(
                compact_list, gt_rows_by_image,
                predicate_names, model, task, args.top_k,
            )
            rows.extend(batch_rows)
            total_gt_sum += tg
            matched_sum += mc
            unmatched_sum += um

            processed_batches += 1
            if processed_batches % 10 == 0:
                print(f"        batch {batch_idx}: cum rows={len(rows)}, "
                      f"matched={matched_sum}, unmatched={unmatched_sum}",
                      flush=True)

    print(f"      Done. Total: {len(rows)} rows, "
          f"matched={matched_sum}, unmatched={unmatched_sum}")
    print()

    # Validate
    print("[3/4] Validating JSONL...")
    errors, warnings, summary = validate_jsonl(
        rows, predicate_names, require_non_empty=True,
    )
    summary["mode"] = "checkpoint"
    summary["checkpoint_path"] = args.ckpt_path
    summary["total_gt_relations"] = total_gt_sum
    summary["exported_rows"] = len(rows)
    summary["matched_rows"] = matched_sum
    summary["unmatched_rows"] = unmatched_sum
    summary["unmatched_ratio"] = unmatched_sum / total_gt_sum if total_gt_sum > 0 else 0.0
    summary["predicate_names_sha256"] = predicate_list_checksum(predicate_names)
    summary["export_method"] = "egtr_gt_aligned_compact"

    print(f"      Total GT:     {total_gt_sum}")
    print(f"      Matched:      {matched_sum}")
    print(f"      Unmatched:    {unmatched_sum}")
    print(f"      Score dim:    {summary['score_dimension']}")

    if errors:
        print(f"      ERRORS: {len(errors)}")
        for err in errors[:10]:
            print(f"        - {err}")
    else:
        print(f"      PASS: No validation errors")
    print()

    # Write
    print("[4/4] Writing output files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, jsonl_path)
    print(f"      Wrote: {jsonl_path}")
    write_export_summary(summary, summary_path)
    print(f"      Wrote: {summary_path}")
    write_schema_md(schema_path)
    print(f"      Wrote: {schema_path}")
    print()

    print("Done.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
