#!/usr/bin/env python3
"""Run official SGG metrics (R@K, mR@K) on a RelTR or Primitive-RelTR checkpoint.

Uses src/core/metrics.py metric() function, which is the project's official
SceneGraphEvaluator-based evaluation path. This is NOT the JSONL GT-aligned
predicate recall; it is the full SGG recall evaluation.
"""

import argparse
import json
from pathlib import Path

import torch


def build_config(test_size, batch_size, num_workers, device):
    from utils.parser import create_parser, default_parser
    from utils.main_utils import load_config, update_config
    args = create_parser().parse_args([])
    cfg = args.__dict__
    cfg.update({
        "method": "reltr", "dataname": "VisualGenome", "test": True,
        "test_dataset_size": test_size if test_size and test_size > 0 else None,
        "val_batch_size": batch_size, "num_workers": num_workers,
        "device": device, "no_display_method_info": True,
        "ex_name": "official_metric_eval", "overwrite": True,
    })
    loaded = load_config("configs/VisualGenome/RelTR.py")
    cfg = update_config(cfg, loaded, exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"])
    for k, v in default_parser().items():
        if cfg.get(k) is None:
            cfg[k] = v
    cfg.update({
        "method": "reltr", "dataname": "VisualGenome", "test": True,
        "test_dataset_size": test_size if test_size and test_size > 0 else None,
        "val_batch_size": batch_size, "num_workers": num_workers,
        "device": device, "no_display_method_info": True,
    })
    return cfg


def load_checkpoint(method, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    from src.exp import BaseExperiment
    BaseExperiment._adapt_state_dict(state, method.model)
    if isinstance(ckpt, dict) and "primitive_head" in ckpt:
        method.primitive_head.load_state_dict(ckpt["primitive_head"], strict=False)


def main():
    parser = argparse.ArgumentParser(description="Official SGG metric evaluation")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--method_name", default="RelTR", choices=["RelTR", "Primitive"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_dataset_size", type=int, default=0, help="0 = full test")
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from src.methods import method_maps
    from utils.main_utils import get_dataset
    from src.core.metrics import metric

    method_key = "reltr_primitive" if args.method_name == "Primitive" else "reltr"
    cfg = build_config(args.test_dataset_size, args.val_batch_size, args.num_workers, device)
    if method_key == "reltr_primitive":
        from utils.main_utils import load_config as _lc
        loaded = _lc("configs/VisualGenome/RelTR_Primitive.py")
        cfg.update({k: v for k, v in loaded.items() if k not in ("method", "val_batch_size", "drop_path", "warmup_epoch")})
        cfg["method"] = "reltr_primitive"

    _, _, test_loader = get_dataset("VisualGenome", cfg)
    method_cls = method_maps[method_key]
    method = method_cls(steps_per_epoch=1, save_dir=args.output_dir, **cfg)
    load_checkpoint(method, args.ckpt_path)
    method.to(torch.device(device))
    method.eval()

    rel_nums = cfg.get("rel_nums", 51)
    entity_nums = cfg.get("entity_nums", 151)
    metrics_list = [
        "predcls_R@50", "predcls_R@100",
        "predcls_mR@50", "predcls_mR@100",
    ]

    # Accumulate across batches
    from collections import defaultdict
    all_evaluators = {}
    all_mr_evaluators = {}
    from lib.evaluation.sg_eval import SceneGraphEvaluator
    for m in ["predcls"]:
        all_evaluators[m] = SceneGraphEvaluator(m)
        all_mr_evaluators[m] = [SceneGraphEvaluator(m) for _ in range(rel_nums)]

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed >= args.max_batches:
                break
            images, targets = method._split_batch(batch)
            targets = method._move_targets_to_device(targets)
            result = method.forward(images, targets)
            outputs = result.get("outputs", {})
            loss_dict = method.criterion(outputs, targets)
            triplet_indices = None
            if hasattr(method.criterion, 'indices') and method.criterion.indices is not None:
                triplet_indices = [
                    (src.cpu().clone(), tgt.cpu().clone())
                    for src, tgt in method.criterion.indices[1]
                ]
            eval_res, eval_log = metric(
                pred=outputs, true=targets, metrics=metrics_list,
                rel_nums=rel_nums, entity_nums=entity_nums,
                triplet_match_indices=triplet_indices,
            )
            processed += 1
            if processed % 200 == 0:
                print(f"  batch {batch_idx}: processed={processed}", flush=True)

    # Collect final results from the metric() calls are per-batch, so we need
    # to use the evaluators directly. Actually metric() creates new evaluators
    # each call. Let's re-do with a single-pass accumulation approach.
    # Instead, let's just accumulate by calling metric() and collecting results.
    # But metric() creates fresh evaluators each time, so it won't accumulate.
    # We need to call the internal functions directly.
    print("Re-running with direct evaluator accumulation...")
    all_evaluators = {"predcls": SceneGraphEvaluator("predcls")}
    all_mr_evaluators = {"predcls": [SceneGraphEvaluator("predcls") for _ in range(rel_nums)]}
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed >= args.max_batches:
                break
            images, targets = method._split_batch(batch)
            targets = method._move_targets_to_device(targets)
            result = method.forward(images, targets)
            outputs = result.get("outputs", {})
            method.criterion(outputs, targets)
            triplet_indices = None
            if hasattr(method.criterion, 'indices') and method.criterion.indices is not None:
                triplet_indices = [
                    (src.cpu().clone(), tgt.cpu().clone())
                    for src, tgt in method.criterion.indices[1]
                ]
            from src.core.metrics import _evaluate_predcls_batch
            _evaluate_predcls_batch(
                outputs=outputs, targets=targets,
                evaluators=all_evaluators,
                mr_evaluators=all_mr_evaluators,
                rel_nums=rel_nums,
                triplet_match_indices=triplet_indices,
            )
            processed += 1
            if processed % 200 == 0:
                print(f"  batch {batch_idx}: processed={processed}", flush=True)

    from src.core.metrics import _collect_main_recall, _collect_mean_recall
    all_results = {}
    all_results.update(_collect_main_recall(all_evaluators))
    all_results.update(_collect_mean_recall(all_mr_evaluators))

    summary = {
        "checkpoint": args.ckpt_path,
        "method": args.method_name,
        "test_dataset_size": args.test_dataset_size,
        "batches_processed": processed,
        "metrics": {k: float(v) for k, v in all_results.items() if not (isinstance(v, float) and v != v)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "official_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
