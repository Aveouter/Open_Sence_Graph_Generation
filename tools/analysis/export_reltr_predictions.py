#!/usr/bin/env python3
"""RelTR PredCLS relation export via Hungarian matcher indices.

RelTR runs full SGDet inference, then uses the SetCriterion's Hungarian
matcher to assign GT relations to triplet queries. This script captures
the matched triplet predictions and exports per-relation JSONL.

Usage:
    python tools/analysis/export_reltr_predictions.py --dry-run
    python tools/analysis/export_reltr_predictions.py \
        --ckpt_path <path> --test_dataset_size 50 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.analysis.export_relation_predictions import (
    load_predicate_names,
    load_object_class_names,
    predicate_list_checksum,
    extract_relation_scores,
    validate_jsonl,
    write_jsonl,
    write_export_summary,
    REQUIRED_FIELDS,
)

RELTR_CKPT_DEFAULT = "outputs/pretrained/reltr/reltr_vg.pth"


def _build_reltr_config(test_size, val_batch_size, num_workers, device):
    from types import SimpleNamespace
    from utils.main_utils import load_config, update_config
    from utils.parser import create_parser, default_parser

    args = create_parser().parse_args([])
    config = args.__dict__
    config.update({
        "method": "RelTR",
        "dataname": "VisualGenome",
        "test": True,
        "test_dataset_size": int(test_size) if test_size and int(test_size) > 0 else None,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "device": device,
        "no_display_method_info": True,
        "ex_name": "reltr_relation_export",
        "overwrite": True,
    })

    cfg_path = _PROJECT_ROOT / "configs" / "VisualGenome" / "RelTR.py"
    loaded_cfg = load_config(str(cfg_path))
    config = update_config(config, loaded_cfg,
                           exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"])
    for key, value in default_parser().items():
        if config.get(key) is None:
            config[key] = value
    config.update({
        "method": "RelTR", "dataname": "VisualGenome", "test": True,
        "test_dataset_size": int(test_size) if test_size and int(test_size) > 0 else None,
        "val_batch_size": val_batch_size, "num_workers": num_workers, "device": device,
        "no_display_method_info": True,
    })
    return SimpleNamespace(**config), config


def _scalar(v):
    if v is None:
        return None
    try:
        import torch
        if torch.is_tensor(v):
            v = v.detach().cpu()
            if v.numel() == 1:
                return v.item()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def _export_reltr(ckpt_path, predicate_names, object_names, test_size,
                  val_batch_size, num_workers, device, max_batches, top_k,
                  output_dir):
    import torch
    from src.exp import BaseExperiment
    from utils.main_utils import get_dataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    _, config = _build_reltr_config(test_size, val_batch_size, num_workers, device)
    print(f"      device: {device}, test_dataset_size: {config.get('test_dataset_size')}")

    print("      Loading test loader...")
    _, _, test_loader = get_dataset("VisualGenome", config)

    print("      Building RelTR method...")
    from src.methods.reltr_method import RelTR_Method
    method = RelTR_Method(
        steps_per_epoch=1,
        save_dir=str(output_dir),
        **config,
    )
    torch_device = torch.device(device)
    method.to(torch_device)
    method.eval()

    print("      Loading checkpoint weights...")
    # RelTR checkpoint is SGB format — needs weights_only=False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        state_dict = {k[6:] if k.startswith("model.") else k: v
                      for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    BaseExperiment._adapt_state_dict(state_dict, method.model)

    rel_nums = len(predicate_names) - 1  # 50
    rows = []
    processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and processed >= max_batches:
                break

            images, targets = method._split_batch(batch)
            targets = method._move_targets_to_device(targets)

            samples = method._to_nested_tensor(images)

            # RelTR forward
            outputs = method.model(samples)

            # Run criterion to get Hungarian matcher indices
            loss_dict = method.criterion(outputs, targets)
            triplet_indices = None
            if hasattr(method.criterion, 'indices') and method.criterion.indices is not None:
                triplet_indices = method.criterion.indices[1]

            # Extract relation scores per image
            rel_logits_all = outputs["rel_logits"]  # [B, T, 51]
            for i, target in enumerate(targets):
                rel_annotations = target.get("rel_annotations")
                if rel_annotations is None:
                    continue
                if torch.is_tensor(rel_annotations):
                    rel_annotations = rel_annotations.detach().cpu()
                if rel_annotations.numel() == 0:
                    continue
                if rel_annotations.dim() == 1:
                    rel_annotations = rel_annotations.reshape(-1, 3)

                labels = target["labels"].detach().cpu().long()
                image_id = int(_scalar(target.get("image_id", i)))

                rel_logits = rel_logits_all[i].float()
                if rel_logits.dim() == 3:
                    rel_logits = rel_logits.squeeze(0)

                rel_scores_all = extract_relation_scores(
                    rel_logits, rel_nums=rel_nums,
                    predicate_bg_index="first", softmax_scope="all",
                )

                R = rel_annotations.shape[0]
                best_rel_scores = np.zeros((R, rel_nums), dtype=np.float32)

                # Hungarian matching
                if triplet_indices is not None and i < len(triplet_indices):
                    tgt_triplet = triplet_indices[i]
                    if tgt_triplet is not None:
                        src_idx, tgt_idx = tgt_triplet
                        src_np = src_idx.detach().cpu().numpy().astype(np.int64)
                        tgt_np = tgt_idx.detach().cpu().numpy().astype(np.int64)
                        for r in range(R):
                            match_mask = (tgt_np == r)
                            if match_mask.any():
                                q_idx = src_np[match_mask][0]
                                if q_idx < rel_scores_all.shape[0]:
                                    best_rel_scores[r] = rel_scores_all[q_idx]

                # Build rows
                for r in range(R):
                    subj_idx = int(rel_annotations[r, 0].item())
                    obj_idx = int(rel_annotations[r, 1].item())
                    gt_pred_id = int(rel_annotations[r, 2].item())
                    gt_pred_name = predicate_names[gt_pred_id]
                    subj_cls = int(labels[subj_idx].item())
                    obj_cls = int(labels[obj_idx].item())

                    scores = best_rel_scores[r]
                    matched = bool(scores.sum() > 0)

                    if matched:
                        topk_indices = np.argsort(-scores)[:top_k]
                        topk_vg_ids = [int(i) + 1 for i in topk_indices]
                        topk_names = [predicate_names[pid] for pid in topk_vg_ids]
                        topk_scores = [float(scores[i]) for i in topk_indices]
                        pred_id = topk_vg_ids[0]
                        pred_name = predicate_names[pred_id]
                    else:
                        topk_vg_ids = []
                        topk_names = []
                        topk_scores = []
                        pred_id = -1
                        pred_name = None

                    rows.append({
                        "image_id": image_id,
                        "subject_idx": subj_idx, "object_idx": obj_idx,
                        "subject_class_id": subj_cls, "object_class_id": obj_cls,
                        "subject_class_name": object_names.get(subj_cls, f"class_{subj_cls}"),
                        "object_class_name": object_names.get(obj_cls, f"class_{obj_cls}"),
                        "gt_predicate_id": gt_pred_id, "gt_predicate_name": gt_pred_name,
                        "pred_predicate_id": pred_id, "pred_predicate_name": pred_name,
                        "predicate_scores_all": [float(s) for s in scores],
                        "topk_predicate_ids": topk_vg_ids,
                        "topk_predicate_names": topk_names,
                        "topk_predicate_scores": topk_scores,
                        "matched_pair_found": matched,
                        "model": "RelTR", "task": "PredCLS",
                    })

            processed += 1
            if processed % 50 == 0:
                print(f"        batch {batch_idx}: cum rows={len(rows)}", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser(description="RelTR PredCLS relation export")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--test_dataset_size", type=int, default=50)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "outputs" / "analysis" / "strong_go" / "RelTR_sanity" / "PredCLS")

    print("=" * 70)
    print("RelTR PredCLS Relation Export")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")

    predicate_names = load_predicate_names()
    object_names = load_object_class_names()
    print(f"  Predicates: {len(predicate_names)}, Objects: {len(object_names)}")

    if args.dry_run:
        print("  Dry-run: done.")
        return

    if not args.ckpt_path:
        print("ERROR: --ckpt_path required")
        sys.exit(1)

    rows = _export_reltr(
        args.ckpt_path, predicate_names, object_names,
        args.test_dataset_size, args.val_batch_size,
        args.num_workers, args.device, args.max_batches, args.top_k,
        output_dir,
    )

    errors, warnings, summary = validate_jsonl(rows, predicate_names,
                                                require_non_empty=True)
    summary["mode"] = "checkpoint"
    summary["checkpoint_path"] = args.ckpt_path
    summary["export_method"] = "reltr_hungarian_matched"

    print(f"  Total: {len(rows)}, Matched: {summary['matched_rows']}, "
          f"Unmatched: {summary['unmatched_rows']}")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    - {e}")
    else:
        print("  PASSED")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "relation_predictions.jsonl")
    write_export_summary(summary, output_dir / "export_summary.json")
    print("  Done.")


if __name__ == "__main__":
    main()
