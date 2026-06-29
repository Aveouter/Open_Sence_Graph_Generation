#!/usr/bin/env python3
"""Phase 0 relation feature extraction for primitive discovery.

Extracts ON-family GT-aligned relation features from RelTR or Motifs.
Outputs a tensor file and metadata JSONL for downstream primitive discovery.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ON_FAMILY = {
    "on",
    "sitting on",
    "standing on",
    "lying on",
    "laying on",
    "walking on",
    "parked on",
    "mounted on",
}


def load_predicate_names():
    from tools.analysis.export_relation_predictions import load_predicate_names
    return load_predicate_names()


def scalar(v, default=0):
    if v is None:
        return default
    if torch.is_tensor(v):
        v = v.detach().cpu()
        if v.numel() == 1:
            return int(v.item())
    try:
        return int(v)
    except Exception:
        return default


def move_images_to_device(images, device):
    if torch.is_tensor(images):
        return images.to(device)
    if isinstance(images, (list, tuple)):
        return [img.to(device) for img in images]
    if hasattr(images, "tensors") and torch.is_tensor(images.tensors):
        images.tensors = images.tensors.to(device)
        if hasattr(images, "mask") and images.mask is not None:
            images.mask = images.mask.to(device)
        return images
    return images


def reltr_forward_with_features(model, samples):
    """Copy RelTR.forward while returning relation pre-logit features."""
    from utils.misc import nested_tensor_from_tensor_list
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    features, pos = model.backbone(samples)
    src, mask = features[-1].decompose()
    assert mask is not None
    hs, hs_t, so_masks, memory = model.transformer(
        model.input_proj(src), mask, model.entity_embed.weight,
        model.triplet_embed.weight, pos[-1], model.so_embed.weight)
    so_masks = so_masks.detach()
    so_masks = model.so_mask_conv(
        so_masks.view(-1, 2, src.shape[-2], src.shape[-1])
    ).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2], -1)
    so_masks = model.so_mask_fc(so_masks)

    hs_sub, hs_obj = torch.split(hs_t, model.hidden_dim, dim=-1)

    outputs_class = model.entity_class_embed(hs)
    outputs_coord = model.entity_bbox_embed(hs).sigmoid()
    outputs_class_sub = model.sub_class_embed(hs_sub)
    outputs_coord_sub = model.sub_bbox_embed(hs_sub).sigmoid()
    outputs_class_obj = model.obj_class_embed(hs_obj)
    outputs_coord_obj = model.obj_bbox_embed(hs_obj).sigmoid()
    rel_pre = torch.cat((hs_sub, hs_obj, so_masks), dim=-1)
    outputs_class_rel = model.rel_class_embed(rel_pre)

    out = {
        "pred_logits": outputs_class[-1],
        "pred_boxes": outputs_coord[-1],
        "sub_logits": outputs_class_sub[-1],
        "sub_boxes": outputs_coord_sub[-1],
        "obj_logits": outputs_class_obj[-1],
        "obj_boxes": outputs_coord_obj[-1],
        "rel_logits": outputs_class_rel[-1],
    }
    feat = {
        "hs_sub": hs_sub[-1],
        "hs_obj": hs_obj[-1],
        "so_masks": so_masks[-1],
        "rel_pre_feature": rel_pre[-1],
        "memory": memory,
    }
    return out, feat


def box_geom(sub_box, obj_box):
    # boxes are cxcywh normalized
    sx, sy, sw, sh = sub_box.unbind(-1)
    ox, oy, ow, oh = obj_box.unbind(-1)
    eps = 1e-6
    return torch.stack([
        sx - ox,
        sy - oy,
        torch.log((sw + eps) / (ow + eps)),
        torch.log((sh + eps) / (oh + eps)),
        sw * sh,
        ow * oh,
        torch.abs(sx - ox),
        torch.abs(sy - oy),
    ], dim=-1)


def extract_reltr(args, predicate_names, device):
    from src.exp import BaseExperiment
    from src.methods.reltr_method import RelTR_Method
    from tools.analysis.export_reltr_predictions import _build_reltr_config
    from utils.main_utils import get_dataset

    _, config = _build_reltr_config(args.test_dataset_size, args.val_batch_size,
                                    args.num_workers, device)
    _, _, test_loader = get_dataset("VisualGenome", config)
    method = RelTR_Method(steps_per_epoch=1, save_dir=str(args.output_dir), **config)
    method.to(torch.device(device))
    method.eval()
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    BaseExperiment._adapt_state_dict(state_dict, method.model)
    method.to(torch.device(device))
    method.eval()

    feats = []
    meta = []
    coverage_total = 0
    coverage_matched = 0
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed >= args.max_batches:
                break
            images, targets = method._split_batch(batch)
            images = move_images_to_device(images, device)
            targets = method._move_targets_to_device(targets)
            samples = method._to_nested_tensor(images)
            outputs, hidden = reltr_forward_with_features(method.model, samples)
            method.criterion(outputs, targets)
            triplet_indices = method.criterion.indices[1] if getattr(method.criterion, "indices", None) is not None else None
            rel_pre = hidden["rel_pre_feature"]
            sub_boxes = outputs["sub_boxes"]
            obj_boxes = outputs["obj_boxes"]
            for i, target in enumerate(targets):
                rels = target.get("rel_annotations")
                if rels is None or rels.numel() == 0:
                    continue
                if rels.dim() == 1:
                    rels = rels.reshape(-1, 3)
                labels = target["labels"].detach().cpu().long()
                image_id = scalar(target.get("image_id", i), i)
                match_map = {}
                if triplet_indices is not None and i < len(triplet_indices):
                    src_idx, tgt_idx = triplet_indices[i]
                    for src, tgt in zip(src_idx.detach().cpu().tolist(), tgt_idx.detach().cpu().tolist()):
                        match_map[int(tgt)] = int(src)
                for r, rel in enumerate(rels.detach().cpu()):
                    gt_id = int(rel[2].item())
                    gt_name = predicate_names[gt_id]
                    if gt_name not in ON_FAMILY:
                        continue
                    coverage_total += 1
                    q = match_map.get(r)
                    if q is None:
                        continue
                    coverage_matched += 1
                    geom = box_geom(sub_boxes[i, q].detach(), obj_boxes[i, q].detach())
                    feat = torch.cat([rel_pre[i, q].detach().cpu(), geom.detach().cpu()], dim=0)
                    feats.append(feat.float())
                    s_idx, o_idx = int(rel[0].item()), int(rel[1].item())
                    meta.append({
                        "row_id": len(meta),
                        "batch_idx": batch_idx,
                        "image_id": image_id,
                        "subject_idx": s_idx,
                        "object_idx": o_idx,
                        "subject_class_id": int(labels[s_idx].item()),
                        "object_class_id": int(labels[o_idx].item()),
                        "gt_predicate_id": gt_id,
                        "gt_predicate_name": gt_name,
                        "query_idx": q,
                        "matched_pair_found": True,
                        "feature_source": "rel_pre_feature+pred_box_geom",
                        "model": "RelTR",
                    })
            processed += 1
            if processed % 10 == 0:
                print(f"batch {batch_idx}: features={len(feats)}", flush=True)
    return feats, meta, coverage_total, coverage_matched


def motifs_forward_with_features(method, images, targets, device):
    # Mirrors Motifs_Method.forward, but returns per-image hidden relation features.
    return_obj_preds = getattr(method.hparams, "eval_mode", "predcls") == "sgcls"
    boxes_list = [t["boxes"] for t in targets]
    labels_list = [t["labels"] for t in targets]
    image_sizes = [t.get("size", t.get("orig_size")) for t in targets]
    visual_feats_list = method._extract_visual_features(images, boxes_list, labels_list, image_sizes)
    outputs = []
    hidden = []
    model = method.model
    with torch.no_grad():
        for i, (box, lab, vis) in enumerate(zip(boxes_list, labels_list, visual_feats_list)):
            extra_kwargs = method._extra_model_kwargs(targets[i], box, lab, return_obj_preds)
            num_objects = vis.size(0)
            pairs = model._generate_pairs(num_objects, vis.device)
            motif_visual_feats = model.input_visual_proj(vis)
            out = model(vis, box, lab, return_obj_preds=return_obj_preds, **extra_kwargs)
            if num_objects == 0 or pairs.numel() == 0:
                outputs.append(out); hidden.append({}); continue
            obj_logits, obj_preds, edge_ctx = model.context_layer(
                motif_visual_feats, box, lab, return_obj_preds=return_obj_preds,
                obj_dists=extra_kwargs.get("obj_dists"))
            edge_rep = model.post_emb(edge_ctx).view(num_objects, 2, model.hidden_dim)
            head_rep = edge_rep[:, 0].contiguous()
            tail_rep = edge_rep[:, 1].contiguous()
            prod_context = torch.cat((head_rep[pairs[:, 0]], tail_rep[pairs[:, 1]]), dim=-1)
            prod_context = model.post_cat(prod_context)
            if model.use_vision:
                edge_visual_feats = (model._project_union_features(extra_kwargs.get("union_feats"))
                                     if extra_kwargs.get("union_feats") is not None
                                     else model._fallback_union_features(motif_visual_feats, pairs))
                prod_final = prod_context * edge_visual_feats
            else:
                edge_visual_feats = prod_context.new_zeros(prod_context.shape)
                prod_final = prod_context
            if model.use_tanh:
                prod_final = torch.tanh(prod_final)
            outputs.append(out)
            hidden.append({
                "pairs": pairs.detach(),
                "head_rep": head_rep[pairs[:, 0]].detach(),
                "tail_rep": tail_rep[pairs[:, 1]].detach(),
                "prod_context": prod_context.detach(),
                "edge_visual_feats": edge_visual_feats.detach(),
                "prod_final": prod_final.detach(),
            })
    return outputs, hidden


def extract_motifs(args, predicate_names, device):
    from src.exp import BaseExperiment
    from src.methods import method_maps
    from tools.analysis.export_relation_predictions import _build_opensgg_config
    from utils.main_utils import get_dataset

    _, config = _build_opensgg_config(args.ckpt_path, args.test_dataset_size,
                                      args.val_batch_size, args.num_workers,
                                      device, method="Motifs")
    _, _, test_loader = get_dataset("VisualGenome", config)
    method_cls = method_maps["motifs"]
    method = method_cls(steps_per_epoch=1, save_dir=str(args.output_dir), **config)
    state_dict = BaseExperiment._load_checkpoint_state_dict(args.ckpt_path)
    BaseExperiment._adapt_state_dict(state_dict, method.model)
    method.to(torch.device(device))
    method.eval()
    feats, meta = [], []
    coverage_total = 0
    coverage_matched = 0
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and processed >= args.max_batches:
                break
            images, targets = method._split_batch(batch)
            images = move_images_to_device(images, device)
            targets = method._move_targets_to_device(targets)
            outputs, hidden = motifs_forward_with_features(method, images, targets, device)
            for i, target in enumerate(targets):
                rels = target.get("rel_annotations")
                if rels is None or rels.numel() == 0 or i >= len(hidden) or not hidden[i]:
                    continue
                if rels.dim() == 1:
                    rels = rels.reshape(-1, 3)
                labels = target["labels"].detach().cpu().long()
                image_id = scalar(target.get("image_id", i), i)
                pairs = hidden[i]["pairs"].detach().cpu()
                pair_to_idx = {(int(pairs[j, 0]), int(pairs[j, 1])): j for j in range(pairs.shape[0])}
                for rel in rels.detach().cpu():
                    gt_id = int(rel[2].item())
                    gt_name = predicate_names[gt_id]
                    if gt_name not in ON_FAMILY:
                        continue
                    coverage_total += 1
                    s_idx, o_idx = int(rel[0].item()), int(rel[1].item())
                    p_idx = pair_to_idx.get((s_idx, o_idx))
                    if p_idx is None:
                        continue
                    coverage_matched += 1
                    feat = hidden[i]["prod_final"][p_idx].detach().cpu().float()
                    feats.append(feat)
                    meta.append({
                        "row_id": len(meta),
                        "batch_idx": batch_idx,
                        "image_id": image_id,
                        "subject_idx": s_idx,
                        "object_idx": o_idx,
                        "subject_class_id": int(labels[s_idx].item()),
                        "object_class_id": int(labels[o_idx].item()),
                        "gt_predicate_id": gt_id,
                        "gt_predicate_name": gt_name,
                        "query_idx": p_idx,
                        "matched_pair_found": True,
                        "feature_source": "prod_final",
                        "model": "Motifs",
                    })
            processed += 1
            if processed % 10 == 0:
                print(f"batch {batch_idx}: features={len(feats)}", flush=True)
    return feats, meta, coverage_total, coverage_matched


def save_outputs(feats, meta, coverage_total, coverage_matched, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if feats:
        features = torch.stack(feats, dim=0)
    else:
        features = torch.empty(0)
    torch.save({"features": features}, output_dir / "features.pt")
    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "num_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]) if features.dim() == 2 and features.shape[0] else 0,
        "coverage_total": coverage_total,
        "coverage_matched": coverage_matched,
        "coverage": coverage_matched / coverage_total if coverage_total else 0.0,
    }
    with open(output_dir / "coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 0 relation features")
    parser.add_argument("--model", required=True, choices=["RelTR", "Motifs"])
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_dataset_size", type=int, default=5000)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    predicate_names = load_predicate_names()

    if args.model == "RelTR":
        feats, meta, total, matched = extract_reltr(args, predicate_names, device)
    else:
        feats, meta, total, matched = extract_motifs(args, predicate_names, device)
    save_outputs(feats, meta, total, matched, args.output_dir)


if __name__ == "__main__":
    import argparse
    main()
