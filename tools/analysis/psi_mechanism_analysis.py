#!/usr/bin/env python3
"""PSI mechanism analysis for Primitive-RelTR.

Runs Primitive-RelTR in eval mode on a test subset, extracts primitive
evidence tokens, computes PSI vs Delta, and performs token ablation/replacement.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

_ON_PARENT = 31
_ON_FINE = {24, 26, 28, 35, 40, 41, 46}
_FINE_NAME_TO_ID = {
    "laying on": 24, "lying on": 26, "mounted on": 28,
    "parked on": 35, "sitting on": 40, "standing on": 41, "walking on": 46,
}


def build_config(test_size, batch_size, num_workers, device):
    from utils.parser import create_parser, default_parser
    from utils.main_utils import load_config, update_config
    args = create_parser().parse_args([])
    cfg = args.__dict__
    cfg.update({
        "method": "reltr_primitive", "dataname": "VisualGenome", "test": True,
        "test_dataset_size": test_size if test_size and test_size > 0 else None,
        "val_batch_size": batch_size, "num_workers": num_workers,
        "device": device, "no_display_method_info": True,
        "ex_name": "psi_analysis", "overwrite": True,
    })
    loaded = load_config("configs/VisualGenome/RelTR_Primitive.py")
    cfg = update_config(cfg, loaded, exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"])
    for k, v in default_parser().items():
        if cfg.get(k) is None:
            cfg[k] = v
    cfg.update({
        "method": "reltr_primitive", "dataname": "VisualGenome", "test": True,
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


def evidence_score(tokens):
    """Compute scalar evidence scores from primitive evidence tokens."""
    return tokens.norm(dim=-1)  # [K] L2 norm per primitive


def compute_psi(ev_scores, comp, fine_id):
    """Compute PSI for a given GT=fine sample.

    Shared channels: those with C[parent] > 0.5
    Private channels: those with C[fine] >> C[parent]
    """
    c = torch.sigmoid(comp)
    parent = c[_ON_PARENT]
    fine = c[fine_id]
    shared_weight = parent.clone()
    private_weight = (fine - parent).clamp(min=0)
    e_shared = (ev_scores * shared_weight).sum() / (shared_weight.sum() + 1e-8)
    e_private = (ev_scores * private_weight).sum() / (private_weight.sum() + 1e-8)
    psi = (e_shared - e_private) / (e_shared + e_private + 1e-8)
    return float(psi.cpu()), float(e_shared.cpu()), float(e_private.cpu())


def main():
    parser = argparse.ArgumentParser(description="PSI mechanism analysis")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_dataset_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=2000)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from src.methods import method_maps
    from utils.main_utils import get_dataset

    cfg = build_config(args.test_dataset_size, args.batch_size, args.num_workers, device)
    _, _, test_loader = get_dataset("VisualGenome", cfg)
    method = method_maps["reltr_primitive"](steps_per_epoch=1, save_dir=args.output_dir, **cfg)
    load_checkpoint(method, args.ckpt_path)
    method.to(torch.device(device))
    method.eval()

    rows = []  # PSI, Delta, collapse flag
    ablation_rows = []  # token ablation records
    replacement_rows = []  # token replacement records

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed >= args.max_batches:
                break
            images, targets = method._split_batch(batch)
            targets = method._move_targets_to_device(targets)

            # Normal forward
            result = method.forward(images, targets)
            outputs = result["outputs"]
            aux = method._primitive_aux
            if aux is None:
                processed += 1
                continue

            comp = method.primitive_head.comp
            base_logits = outputs["base_rel_logits"]
            prim_logits = outputs["primitive_rel_logits"]

            # Hungarian matching
            method.criterion(outputs, targets)
            triplet_indices = method.criterion.indices[1]

            for i, target in enumerate(targets):
                rels = target.get("rel_annotations")
                if rels is None or rels.numel() == 0 or i >= len(triplet_indices):
                    continue
                if rels.dim() == 1:
                    rels = rels.reshape(-1, 3)
                src_idx, tgt_idx = triplet_indices[i]
                for src, tgt in zip(src_idx, tgt_idx):
                    gt_id = int(rels[int(tgt.item()), 2].item())
                    if gt_id not in _ON_FINE:
                        continue
                    q = int(src.item())
                    ev = evidence_score(aux["evidence"][i * outputs["rel_logits"].shape[1] + q])
                    psi, e_shared, e_private = compute_psi(ev, comp, gt_id)
                    p_base = base_logits[i, q].softmax(-1)
                    p_prim_full = outputs["rel_logits"][i, q].softmax(-1)
                    delta = float(p_prim_full[_ON_PARENT].cpu() - p_prim_full[gt_id].cpu())
                    pred_id = int(p_prim_full.argmax(-1).item())
                    collapsed = int(pred_id == _ON_PARENT)
                    rows.append({
                        "gt_id": gt_id, "q": q,
                        "psi": psi, "e_shared": e_shared, "e_private": e_private,
                        "delta": delta, "collapsed": collapsed,
                        "base_on_prob": float(p_base[_ON_PARENT].cpu()),
                        "prim_on_prob": float(p_prim_full[_ON_PARENT].cpu()),
                        "base_fine_prob": float(p_base[gt_id].cpu()),
                        "prim_fine_prob": float(p_prim_full[gt_id].cpu()),
                    })

                    # Token ablation: zero out private-dominant primitive tokens
                    c = torch.sigmoid(comp)
                    private_weight = (c[gt_id] - c[_ON_PARENT]).clamp(min=0)
                    private_mask = (private_weight > 0.5 * private_weight.max()).float()
                    ablated_ev = ev * (1 - private_mask)
                    abl_e_shared, abl_e_private = (
                        (ablated_ev * c[_ON_PARENT]).sum() / (c[_ON_PARENT].sum() + 1e-8),
                        (ablated_ev * private_weight * (1 - private_mask)).sum() / (private_weight.sum() + 1e-8),
                    )
                    ablation_rows.append({
                        "gt_id": gt_id, "q": q,
                        "psi_before": psi,
                        "e_shared_before": e_shared, "e_private_before": e_private,
                        "e_shared_after": float(abl_e_shared.cpu()),
                        "e_private_after": float(abl_e_private.cpu()),
                    })

            processed += 1
            if processed % 200 == 0:
                print(f"  batch {batch_idx}: samples={len(rows)}", flush=True)

    # Compute summary statistics
    import numpy as np
    psi_vals = np.array([r["psi"] for r in rows])
    delta_vals = np.array([r["delta"] for r in rows])
    collapsed = np.array([r["collapsed"] for r in rows], dtype=bool)

    # Pearson r
    if len(psi_vals) > 2:
        r = np.corrcoef(psi_vals, delta_vals)[0, 1]
    else:
        r = float("nan")

    # PSI by collapse
    psi_collapse = psi_vals[collapsed].mean() if collapsed.any() else float("nan")
    psi_correct = psi_vals[~collapsed].mean() if (~collapsed).any() else float("nan")

    # AUC: PSI predicting collapse (manual implementation, no sklearn dependency)
    if len(set(collapsed.astype(int))) > 1 and len(psi_vals) > 1:
        pos = psi_vals[collapsed]
        neg = psi_vals[~collapsed]
        # Mann-Whitney U based AUC
        correct = 0
        total = 0
        for p in pos:
            for n in neg:
                if p > n:
                    correct += 1
                elif p == n:
                    correct += 0.5
                total += 1
        auc = correct / total if total > 0 else 0.5
    else:
        auc = 0.5

    summary = {
        "n_samples": len(rows),
        "n_collapse": int(collapsed.sum()),
        "n_correct": int((~collapsed).sum()),
        "pearson_r": float(r),
        "psi_mean_collapse": float(psi_collapse),
        "psi_mean_correct": float(psi_correct),
        "auc_psi_predicting_collapse": float(auc),
        "delta_mean_collapse": float(delta_vals[collapsed].mean()) if collapsed.any() else float("nan"),
        "delta_mean_correct": float(delta_vals[~collapsed].mean()) if (~collapsed).any() else float("nan"),
        "ablation": {
            "n": len(ablation_rows),
            "psi_before_mean": float(np.mean([r["psi_before"] for r in ablation_rows])),
            "e_private_before_mean": float(np.mean([r["e_private_before"] for r in ablation_rows])),
            "e_private_after_mean": float(np.mean([r["e_private_after"] for r in ablation_rows])),
        },
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "psi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "psi_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
