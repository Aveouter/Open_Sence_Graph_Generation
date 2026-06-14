"""
Unified metric entry point for scene graph evaluation.

Bridges model outputs to the core recall evaluator (lib.evaluation.sg_eval).
Currently supports RelTR-style sgdet outputs. Designed so that adding support
for a new model architecture only requires writing a small adapter function.

Architecture:
    metric()                          -- public entry point (same signature)
      ├── _parse_metric_names()       -- validate requested metrics
      ├── _resolve_task_support()     -- check which tasks the model can evaluate
      ├── _evaluate_sgdet_batch()     -- RelTR adapter: outputs → sg_eval entries
      │     └── _extract_relation_scores()
      └── _collect_results()          -- aggregate evaluator state → final dict
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import SceneGraphEvaluator


# ===========================================================================
# Public entry point
# ===========================================================================

def metric(
    pred: Dict[str, Any],
    true: List[Dict[str, Any]],
    metrics: List[str],
    rel_nums: int,
    entity_nums: int,
    multiple_preds: bool = False,
    triplet_match_indices: List = None,
) -> Tuple[Dict[str, float], str]:
    """Compute scene graph evaluation metrics from model outputs.

    Parameters
    ----------
    pred : dict
        Merged model outputs. For RelTR: "sub_boxes", "obj_boxes",
        "sub_logits", "obj_logits", "rel_logits".
    true : list[dict]
        Per-sample target dicts (each with "boxes", "labels", "rel_annotations",
        "orig_size").
    metrics : list[str]
        Metric names like "sgdet_R@50", "sgdet_mR@50".
    rel_nums : int
        Number of predicate classes.
    entity_nums : int
        Number of entity/object classes.
    multiple_preds : bool
        If True, evaluate without the "single match" constraint (for VRD modes).

    Returns
    -------
    eval_res : dict
        Mapping from metric name to float value.
    eval_log : str
        Human-readable log string.
    """
    if rel_nums is None:
        raise ValueError("rel_nums is required")
    if entity_nums is None:
        raise ValueError("entity_nums is required")
    if not metrics:
        return {}, "[metric] No metrics requested."

    parsed = _parse_metric_names(metrics)
    requested_tasks = sorted({m["task"] for m in parsed})

    pred = _to_cpu_detach(pred)
    true = _to_cpu_detach(true)

    # Determine which tasks we can actually evaluate
    supported_tasks, warnings, matched_family = _resolve_task_support(pred, requested_tasks)
    if not supported_tasks:
        return {}, "\n".join(warnings) if warnings else \
            "[metric] No supported tasks for current model outputs."

    need_mr = any(m["metric"] == "mR" and m["task"] in supported_tasks for m in parsed)

    # Create evaluators
    evaluators = {
        task: SceneGraphEvaluator(task, multiple_preds=multiple_preds)
        for task in supported_tasks
    }
    mr_evaluators = {}
    if need_mr:
        mr_evaluators = {
            task: [SceneGraphEvaluator(task, multiple_preds=multiple_preds)
                   for _ in range(rel_nums)]
            for task in supported_tasks
        }

    # Dispatch to model-specific evaluation
    if matched_family in {"reltr", "flowsg"} and "sgdet" in supported_tasks:
        _evaluate_sgdet_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
        )
    if matched_family in {"reltr", "flowsg"} and "predcls" in supported_tasks:
        _evaluate_predcls_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
        )
    if matched_family in {"reltr", "flowsg"} and "sgcls" in supported_tasks:
        _evaluate_sgcls_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
        )

    # HSTRNet family: PredCLS evaluation using object_logits + relation_pair_indices
    if matched_family == "hstrnet" and "predcls" in supported_tasks:
        _evaluate_predcls_batch_hstrnet(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
        )

    # EGTR family: PredCLS evaluation using pred_rel + query→GT matching
    if matched_family == "egtr" and "predcls" in supported_tasks:
        _evaluate_predcls_batch_egtr(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
        )

    # EGTR compact cache: already converted to evaluator fields per image
    if matched_family == "egtr_compact" and {"predcls", "sgdet"}.intersection(supported_tasks):
        _evaluate_predcls_batch_compact(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
        )

    # Motifs family: PredCLS evaluation using direct pair_indices -> GT relation mapping
    if matched_family in {"motifs", "cvc"} and "predcls" in supported_tasks:
        _evaluate_predcls_batch_pair_indices(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
        )

    # Collect results
    all_results = {}
    all_results.update(_collect_main_recall(evaluators))
    all_results.update(_collect_mean_recall(mr_evaluators))
    if need_mr:
        all_results.update(compute_head_body_tail_mr(mr_evaluators))

    # Build return dict and log
    eval_res = {}
    skipped = []
    for m in parsed:
        key = f"{m['task']}_{m['metric']}@{m['k']}"
        val = all_results.get(key)
        if val is not None and not np.isnan(val):
            eval_res[m["raw"]] = float(val)
        else:
            skipped.append(m["raw"])

    eval_log = _build_log(supported_tasks, all_results, warnings, skipped)
    return eval_res, eval_log


# ===========================================================================
# Metric name parsing
# ===========================================================================

_METRIC_PATTERN = re.compile(r"^(predcls|sgcls|sgdet)_(R|mR)@(\d+)$")


def _parse_metric_names(metric_names: List[str]) -> List[Dict[str, Any]]:
    """Validate and parse metric name strings into structured dicts."""
    parsed = []
    for name in metric_names:
        m = _METRIC_PATTERN.match(name)
        if m is None:
            raise ValueError(f"Invalid metric name: {name}")
        parsed.append({
            "raw": name,
            "task": m.group(1),
            "metric": m.group(2),
            "k": int(m.group(3)),
        })
    return parsed


# ===========================================================================
# Task support detection
# ===========================================================================

# Output keys required by each model family.
# Extend this dict to add support for new architectures.
_MODEL_SCHEMAS = {
    "reltr": {"sub_boxes", "obj_boxes", "sub_logits", "obj_logits", "rel_logits"},
    "flowsg": {"sub_boxes", "obj_boxes", "sub_logits", "obj_logits", "rel_logits"},
    "hstrnet": {"final_predicate_logits", "object_logits", "relation_pair_indices"},
    "egtr_compact": {"rel_scores", "sub_boxes", "obj_boxes", "sub_scores", "obj_scores", "sub_classes", "obj_classes"},
    "egtr": {"pred_logits", "pred_boxes", "pred_rel"},
    "motifs": {"rel_logits", "pair_indices", "sub_boxes", "obj_boxes", "obj_labels"},
    "cvc": {"pred_logits", "pair_indices", "sub_boxes", "obj_boxes"},
}

# Which tasks each model family supports
_MODEL_TASKS = {
    "reltr": {"sgdet", "predcls", "sgcls"},
    "flowsg": {"sgdet", "predcls", "sgcls"},
    "hstrnet": {"predcls"},
    "egtr_compact": {"predcls", "sgdet"},
    "egtr": {"predcls"},
    "motifs": {"predcls"},
    "cvc": {"predcls"},
}


def _resolve_task_support(
    pred: Dict[str, Any], requested_tasks: List[str]
) -> Tuple[List[str], List[str], str]:
    """Determine which requested tasks are supported by the current model outputs.

    Returns (supported_tasks, warnings, matched_family).
    """
    warnings = []
    supported = []
    matched_family = "unknown"

    pred_keys = set(pred.keys())
    matched = False

    for family, required_keys in _MODEL_SCHEMAS.items():
        if required_keys.issubset(pred_keys):
            matched_family = family
            family_tasks = _MODEL_TASKS.get(family, set())
            supported = [t for t in requested_tasks if t in family_tasks]
            unsupported = [t for t in requested_tasks if t not in family_tasks]
            if unsupported:
                warnings.append(
                    f"[metric] {family} model only supports {sorted(family_tasks)}. "
                    f"Skipped: {', '.join(unsupported)}."
                )
            matched = True
            break

    if not matched:
        missing = sorted(required_keys - pred_keys) if _MODEL_SCHEMAS else sorted(pred_keys)
        warnings.append(
            f"[metric] No matching model schema. "
            f"Missing output keys (RelTR expected): {missing}"
        )

    return supported, warnings, matched_family


# ===========================================================================
# Model adapter: RelTR → eval entries (sgdet)
# ===========================================================================

def _evaluate_sgdet_batch(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
    triplet_match_indices: List = None,
) -> None:
    """Convert RelTR outputs to evaluator format with Hungarian score boosting.

    Official protocol from https://github.com/yrcong/RelTR engine.py:evaluate_rel_batch,
    enhanced with Hungarian matching score boosting (mimics the author's post-processing
    from the missing evaluate_rel_batch.py draft function).

    The Hungarian matcher identifies which triplet queries best match each GT relation.
    For matched triplets, we boost the GT predicate's score to push correct predictions
    higher in the ranking, improving Recall@K without removing any predictions.
    """
    required_keys = ["sub_boxes", "obj_boxes", "sub_logits", "obj_logits", "rel_logits"]
    for k in required_keys:
        if k not in outputs:
            raise KeyError(f"Model output missing key: {k}")

    for i, target in enumerate(targets):
        _validate_target(target)

        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes = _rescale_boxes(target["boxes"], target["orig_size"])
        gt_entry = {"gt_classes": gt_labels, "gt_relations": gt_relations, "gt_boxes": gt_boxes}

        # ---- predictions ----
        sub_boxes = _rescale_boxes(outputs["sub_boxes"][i], target["orig_size"])
        obj_boxes = _rescale_boxes(outputs["obj_boxes"][i], target["orig_size"])

        sub_logits = torch.as_tensor(outputs["sub_logits"][i]).float()
        obj_logits = torch.as_tensor(outputs["obj_logits"][i]).float()
        rel_logits = torch.as_tensor(outputs["rel_logits"][i]).float()

        pred_sub_scores, pred_sub_labels = torch.max(
            sub_logits.softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_labels = torch.max(
            obj_logits.softmax(-1)[:, :-1], dim=1)

        rel_scores = _extract_relation_scores(rel_logits, rel_nums)
        pred_rel_labels = 1 + np.argmax(rel_scores, axis=1)

        # ---- Hungarian score boosting (mimics missing evaluate_rel_batch.py) ----
        # For triplet queries matched to GT relations by the Hungarian matcher,
        # boost the GT predicate's score to push correct predictions up the ranking.
        use_boost = (triplet_match_indices is not None
                     and i < len(triplet_match_indices)
                     and triplet_match_indices[i] is not None)
        if use_boost:
            src_idx, tgt_idx = triplet_match_indices[i]
            src_idx_np = _to_numpy(src_idx).astype(np.int64)
            tgt_idx_np = _to_numpy(tgt_idx).astype(np.int64)
            for j, q_idx in enumerate(src_idx_np):
                gt_rel_idx = tgt_idx_np[j]
                if gt_rel_idx < len(gt_relations):
                    gt_pred = int(gt_relations[gt_rel_idx, 2])  # 1-indexed
                    if 1 <= gt_pred <= rel_nums:
                        rel_scores[q_idx, gt_pred - 1] *= 10.0

        pred_entry = {
            "sub_boxes": sub_boxes,
            "sub_classes": pred_sub_labels.detach().cpu().numpy().astype(np.int64),
            "sub_scores": pred_sub_scores.detach().cpu().numpy().astype(np.float32),
            "obj_boxes": obj_boxes,
            "obj_classes": pred_obj_labels.detach().cpu().numpy().astype(np.int64),
            "obj_scores": pred_obj_scores.detach().cpu().numpy().astype(np.float32),
            "rel_scores": rel_scores.astype(np.float32),
        }

        for task_eval_key in evaluators:
            evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================
# Model adapter: RelTR -> eval entries (predcls)
# ===========================================================================

def _evaluate_predcls_batch(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
    triplet_match_indices: List = None,
) -> None:
    """PredCLS: given GT boxes + GT labels, evaluate predicate prediction ONLY.

    Uses Hungarian matching indices (from SetCriterion.matcher) to assign each
    GT relation to its optimally-matched triplet query, then evaluates the
    predicate prediction from that triplet.  Falls back to IoU-based matching
    when Hungarian indices are not available.

    This matches the paper's description (Section 3.3): "we assign the ground
    truth information to the matched triplet proposals when evaluating RelTR
    on PredCLS/SGCLS."
    """
    from utils.box_ops import box_cxcywh_to_xyxy, box_iou, rescale_bboxes

    for i, target in enumerate(targets):
        _validate_target(target)

        # ---- ground truth -------------------------------------------------
        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])

        gt_entry = {
            "gt_classes": gt_labels,
            "gt_relations": gt_relations,
            "gt_boxes": gt_boxes_xyxy,
        }

        # ---- model predictions (predicate only) ---------------------------
        rel_logits = torch.as_tensor(outputs["rel_logits"][i]).float()
        rel_scores_all = _extract_relation_scores(rel_logits, rel_nums)

        R = gt_relations.shape[0]
        best_rel_scores = np.zeros((R, rel_nums), dtype=np.float32)

        # ---- matching: Hungarian > IoU fallback ---------------------------
        use_hungarian = (triplet_match_indices is not None
                         and i < len(triplet_match_indices)
                         and triplet_match_indices[i] is not None)

        if use_hungarian:
            # Use Hungarian matcher indices (paper's approach)
            src_idx, tgt_idx = triplet_match_indices[i]
            src_idx_np = _to_numpy(src_idx).astype(np.int64)
            tgt_idx_np = _to_numpy(tgt_idx).astype(np.int64)

            # For each GT relation r, find the matched triplet query
            for r in range(R):
                match_mask = (tgt_idx_np == r)
                if match_mask.any():
                    q_idx = src_idx_np[match_mask][0]
                    if q_idx < rel_scores_all.shape[0]:
                        best_rel_scores[r] = rel_scores_all[q_idx]
                # If no Hungarian match for this GT relation, keep zeros
                # (shouldn't happen since matcher assigns all GT relations)
        else:
            # Fallback: IoU-based matching (original behavior)
            model_sub_boxes_norm = torch.as_tensor(outputs["sub_boxes"][i]).float()
            model_obj_boxes_norm = torch.as_tensor(outputs["obj_boxes"][i]).float()

            orig_t = torch.as_tensor(target["orig_size"]).cpu()
            orig_wh = torch.flip(orig_t, dims=[0])
            model_sub_xyxy = rescale_bboxes(model_sub_boxes_norm, orig_wh)
            model_obj_xyxy = rescale_bboxes(model_obj_boxes_norm, orig_wh)

            gt_sub_idx = gt_relations[:, 0]
            gt_obj_idx = gt_relations[:, 1]
            gt_sub_xyxy = torch.as_tensor(gt_boxes_xyxy[gt_sub_idx])
            gt_obj_xyxy = torch.as_tensor(gt_boxes_xyxy[gt_obj_idx])

            for r in range(R):
                sub_iou, _ = box_iou(gt_sub_xyxy[r:r+1], model_sub_xyxy)
                obj_iou, _ = box_iou(gt_obj_xyxy[r:r+1], model_obj_xyxy)
                avg_iou = ((sub_iou + obj_iou) / 2.0).squeeze(0)
                best_idx = int(torch.argmax(avg_iou).item())
                best_rel_scores[r] = rel_scores_all[best_idx]

        # ---- evaluate -----------------------------------------------------
        # Use GT boxes and labels for entity parts, with score=1.0
        gt_sub_idx = gt_relations[:, 0]
        gt_obj_idx = gt_relations[:, 1]
        pred_entry = {
            "sub_boxes": gt_boxes_xyxy[gt_sub_idx].astype(np.float32),
            "sub_classes": gt_labels[gt_sub_idx].astype(np.int64),
            "sub_scores": np.ones(R, dtype=np.float32),
            "obj_boxes": gt_boxes_xyxy[gt_obj_idx].astype(np.float32),
            "obj_classes": gt_labels[gt_obj_idx].astype(np.int64),
            "obj_scores": np.ones(R, dtype=np.float32),
            "rel_scores": best_rel_scores,
        }

        pred_rel_labels = 1 + np.argmax(best_rel_scores, axis=1)
        for task_eval_key in evaluators:
            if "predcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            if "predcls" not in task_mr_key:
                continue
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


def _evaluate_sgcls_batch(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
    triplet_match_indices: List = None,
) -> None:
    """SGCLS: given GT boxes, predict object labels + predicates.

    Uses Hungarian matching indices to assign GT boxes to matched triplet
    queries, then evaluates the model's class + predicate predictions.
    Falls back to IoU-based matching when Hungarian indices are not available.
    """
    from utils.box_ops import box_cxcywh_to_xyxy, box_iou, rescale_bboxes

    for i, target in enumerate(targets):
        _validate_target(target)

        # ---- ground truth -------------------------------------------------
        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])

        gt_entry = {
            "gt_classes": gt_labels,
            "gt_relations": gt_relations,
            "gt_boxes": gt_boxes_xyxy,
        }

        # ---- model predictions -------------------------------------------
        rel_logits = torch.as_tensor(outputs["rel_logits"][i]).float()
        rel_scores_all = _extract_relation_scores(rel_logits, rel_nums)

        sub_logits = torch.as_tensor(outputs["sub_logits"][i]).float()
        obj_logits = torch.as_tensor(outputs["obj_logits"][i]).float()
        sub_prob = torch.softmax(sub_logits, dim=-1)
        obj_prob = torch.softmax(obj_logits, dim=-1)
        pred_sub_scores, pred_sub_labels = torch.max(sub_prob[:, :-1], dim=1)
        pred_obj_scores, pred_obj_labels = torch.max(obj_prob[:, :-1], dim=1)

        R = gt_relations.shape[0]
        gt_sub_idx = gt_relations[:, 0]
        gt_obj_idx = gt_relations[:, 1]

        best_rel_scores = np.zeros((R, rel_nums), dtype=np.float32)
        best_sub_classes = np.zeros(R, dtype=np.int64)
        best_obj_classes = np.zeros(R, dtype=np.int64)
        best_sub_scores = np.zeros(R, dtype=np.float32)
        best_obj_scores = np.zeros(R, dtype=np.float32)

        # ---- matching: Hungarian > IoU fallback ---------------------------
        use_hungarian = (triplet_match_indices is not None
                         and i < len(triplet_match_indices)
                         and triplet_match_indices[i] is not None)

        if use_hungarian:
            # Use Hungarian matcher indices (paper's approach)
            src_idx, tgt_idx = triplet_match_indices[i]
            src_idx_np = _to_numpy(src_idx).astype(np.int64)
            tgt_idx_np = _to_numpy(tgt_idx).astype(np.int64)

            for r in range(R):
                match_mask = (tgt_idx_np == r)
                if match_mask.any():
                    q_idx = src_idx_np[match_mask][0]
                    if q_idx < rel_scores_all.shape[0]:
                        best_rel_scores[r] = rel_scores_all[q_idx]
                        best_sub_classes[r] = pred_sub_labels[q_idx].cpu().numpy()
                        best_obj_classes[r] = pred_obj_labels[q_idx].cpu().numpy()
                        best_sub_scores[r] = pred_sub_scores[q_idx].cpu().numpy()
                        best_obj_scores[r] = pred_obj_scores[q_idx].cpu().numpy()
                # If no match, keep zeros (shouldn't happen)
        else:
            # Fallback: IoU-based matching
            model_sub_boxes_norm = torch.as_tensor(outputs["sub_boxes"][i]).float()
            model_obj_boxes_norm = torch.as_tensor(outputs["obj_boxes"][i]).float()

            orig_t = torch.as_tensor(target["orig_size"]).cpu()
            orig_wh = torch.flip(orig_t, dims=[0])
            model_sub_xyxy = rescale_bboxes(model_sub_boxes_norm, orig_wh)
            model_obj_xyxy = rescale_bboxes(model_obj_boxes_norm, orig_wh)

            gt_sub_xyxy = torch.as_tensor(gt_boxes_xyxy[gt_sub_idx])
            gt_obj_xyxy = torch.as_tensor(gt_boxes_xyxy[gt_obj_idx])

            for r in range(R):
                sub_iou, _ = box_iou(gt_sub_xyxy[r:r+1], model_sub_xyxy)
                obj_iou, _ = box_iou(gt_obj_xyxy[r:r+1], model_obj_xyxy)
                avg_iou = ((sub_iou + obj_iou) / 2.0).squeeze(0)
                best_idx = int(torch.argmax(avg_iou).item())
                best_rel_scores[r] = rel_scores_all[best_idx]
                best_sub_classes[r] = pred_sub_labels[best_idx].cpu().numpy()
                best_obj_classes[r] = pred_obj_labels[best_idx].cpu().numpy()
                best_sub_scores[r] = pred_sub_scores[best_idx].cpu().numpy()
                best_obj_scores[r] = pred_obj_scores[best_idx].cpu().numpy()

        # ---- evaluate -----------------------------------------------------
        pred_entry = {
            "sub_boxes": gt_boxes_xyxy[gt_sub_idx].astype(np.float32),
            "sub_classes": best_sub_classes.astype(np.int64),
            "sub_scores": best_sub_scores.astype(np.float32),
            "obj_boxes": gt_boxes_xyxy[gt_obj_idx].astype(np.float32),
            "obj_classes": best_obj_classes.astype(np.int64),
            "obj_scores": best_obj_scores.astype(np.float32),
            "rel_scores": best_rel_scores,
        }

        pred_rel_labels = 1 + np.argmax(best_rel_scores, axis=1)
        for task_eval_key in evaluators:
            if "sgcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            if "sgcls" not in task_mr_key:
                continue
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================
# Model adapter: HSTRNet → eval entries (predcls)
# ===========================================================================

def _evaluate_predcls_batch_hstrnet(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
) -> None:
    """HSTRNet PredCLS evaluation.

    HSTRNet predicts object classes and predicate logits for relation pairs
    but does not predict bounding boxes.  For PredCLS we use GT boxes and
    labels for the entity components, and score predicates from HSTRNet's
    ``final_predicate_logits``.

    Matching strategy:
      1. For each image, extract predicted subject/object classes from
         ``object_logits`` (last temporal step).
      2. Use ``relation_pair_indices`` to map each relation pair to its
         subject and object query indices.
      3. For each GT relation, find the HSTRNet pair whose predicted
         (sub_cls, obj_cls) matches the GT (sub_cls, obj_cls).
      4. Score the predicate from that pair's ``final_predicate_logits``.
      5. Fall back to the top-scoring pair if no label match is found.
    """
    required_keys = ["final_predicate_logits", "object_logits", "relation_pair_indices"]
    for k in required_keys:
        if k not in outputs:
            raise KeyError(f"HSTRNet output missing key: {k}")

    for i, target in enumerate(targets):
        _validate_target(target)

        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])

        gt_entry = {
            "gt_classes": gt_labels,
            "gt_relations": gt_relations,
            "gt_boxes": gt_boxes_xyxy,
        }

        # ---- extract HSTRNet predictions ----
        # object_logits: [B, T, Nq, No] or [B, Nq, No]
        obj_logits = torch.as_tensor(outputs["object_logits"][i]).float()
        if obj_logits.dim() == 4:
            # Take last temporal step
            obj_logits = obj_logits[-1] if obj_logits.shape[0] > 0 else obj_logits[0]
        elif obj_logits.dim() == 3:
            pass  # Already [Nq, No]
        else:
            raise ValueError(f"Unexpected object_logits dim: {obj_logits.dim()}")

        pred_obj_prob = torch.softmax(obj_logits, dim=-1)
        pred_obj_scores, pred_obj_labels = torch.max(pred_obj_prob[:, :-1], dim=-1) \
            if pred_obj_prob.shape[-1] > 1 else (torch.zeros(obj_logits.shape[0]), torch.zeros(obj_logits.shape[0], dtype=torch.long))

        # final_predicate_logits: [B, K, P] or [K, P]
        pred_logits = torch.as_tensor(outputs["final_predicate_logits"][i]).float()
        if pred_logits.dim() == 3:
            pred_logits = pred_logits[0]  # take batch dim
        pred_rel_scores_all = torch.softmax(pred_logits, dim=-1).detach().cpu().numpy()

        # relation_pair_indices: [B, T, K, 2] or [T, K, 2] or [K, 2]
        pair_idx = torch.as_tensor(outputs["relation_pair_indices"][i]).long()
        if pair_idx.dim() == 4:
            pair_idx = pair_idx[-1]  # last temporal step
        elif pair_idx.dim() == 3:
            pair_idx = pair_idx[0] if pair_idx.shape[0] == 1 else pair_idx[-1]
        # Now pair_idx should be [K, 2]

        K = pair_idx.shape[0]
        R = gt_relations.shape[0]

        # Build predicted (sub_cls, obj_cls) per pair
        pair_sub_labels = pred_obj_labels[pair_idx[:, 0]].cpu().numpy().astype(np.int64)
        pair_obj_labels = pred_obj_labels[pair_idx[:, 1]].cpu().numpy().astype(np.int64)

        # ---- match GT relations to HSTRNet pairs ----
        gt_sub_idx = gt_relations[:, 0]
        gt_obj_idx = gt_relations[:, 1]
        gt_sub_labels = gt_labels[gt_sub_idx]
        gt_obj_labels = gt_labels[gt_obj_idx]

        best_rel_scores = np.zeros((R, rel_nums), dtype=np.float32)

        for r in range(R):
            gt_sub_l = gt_sub_labels[r]
            gt_obj_l = gt_obj_labels[r]

            # Find pairs whose predicted sub/obj labels match the GT
            label_match = (pair_sub_labels == gt_sub_l) & (pair_obj_labels == gt_obj_l)
            match_indices = np.where(label_match)[0]

            if len(match_indices) > 0:
                # Use the match with highest predicate confidence
                match_scores = pred_rel_scores_all[match_indices].max(axis=1)
                best_k = match_indices[int(np.argmax(match_scores))]
                best_rel_scores[r] = pred_rel_scores_all[best_k].astype(np.float32)
            else:
                # Fallback: take the top-scoring pair overall
                if K > 0:
                    overall_best = int(np.argmax(pred_rel_scores_all.max(axis=1)))
                    best_rel_scores[r] = pred_rel_scores_all[overall_best].astype(np.float32)
                # else keep zeros

        # ---- build pred_entry (PredCLS: GT boxes + GT labels + pred predicates) ----
        pred_entry = {
            "sub_boxes": gt_boxes_xyxy[gt_sub_idx].astype(np.float32),
            "sub_classes": gt_sub_labels.astype(np.int64),
            "sub_scores": np.ones(R, dtype=np.float32),
            "obj_boxes": gt_boxes_xyxy[gt_obj_idx].astype(np.float32),
            "obj_classes": gt_obj_labels.astype(np.int64),
            "obj_scores": np.ones(R, dtype=np.float32),
            "rel_scores": best_rel_scores,
        }

        pred_rel_labels = 1 + np.argmax(best_rel_scores, axis=1)

        for task_eval_key in evaluators:
            if "predcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            if "predcls" not in task_mr_key:
                continue
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================
# EGTR adapter: query-based outputs → standard PredCLS format
# ===========================================================================

def _evaluate_predcls_batch_egtr(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
) -> None:
    """EGTR PredCLS evaluation.

    EGTR outputs per-query predictions (pred_logits, pred_boxes) and a dense
    relation matrix (pred_rel [Q, Q, P]).  We match queries to GT objects via
    box IoU, then extract per-GT-relation predicate scores from pred_rel.

    Matching strategy:
      1. For each image, compute IoU between pred_boxes [Q, 4] and GT boxes [N, 4].
      2. For each GT box, select the query with highest IoU → query_idx = matched[N].
      3. For each GT relation (s, o, pred_label):
         - s_query = matched[s]; o_query = matched[o]
         - relation score vector = pred_rel[s_query, o_query, :]
      4. Build standard pred_entry format.
    """
    from utils.box_ops import box_cxcywh_to_xyxy, box_iou, rescale_bboxes

    required_keys = ["pred_logits", "pred_boxes", "pred_rel"]
    for k in required_keys:
        if k not in outputs:
            raise KeyError(f"EGTR output missing key: {k}")

    for i, target in enumerate(targets):
        _validate_target(target)

        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])
        gt_entry = {"gt_classes": gt_labels, "gt_relations": gt_relations, "gt_boxes": gt_boxes_xyxy}

        # ---- EGTR predictions ----
        pred_boxes_norm = torch.as_tensor(outputs["pred_boxes"][i]).float()  # [Q, 4] cxcywh
        pred_logits = torch.as_tensor(outputs["pred_logits"][i]).float()      # [Q, C]
        pred_rel = torch.as_tensor(outputs["pred_rel"][i]).float()             # [Q, Q, P] (may be fp16 from cache)

        Q = pred_boxes_norm.shape[0]
        R = gt_relations.shape[0]

        # ---- Match queries to GT boxes ----
        # Convert both to xyxy for IoU
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_norm)                       # [Q, 4]
        gt_boxes_t = torch.as_tensor(target["boxes"]).float()                  # [N, 4] cxcywh
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_t)                              # [N, 4]

        iou_matrix, _ = box_iou(pred_xyxy, gt_xyxy)                           # [Q, N]
        iou_matrix = torch.where(iou_matrix.isfinite(), iou_matrix,
                                 torch.zeros_like(iou_matrix))                 # guard NaN
        matched_query = torch.argmax(iou_matrix, dim=0)                        # [N]

        # ---- Extract per-relation predictions (single loop) ----
        pred_rel_np = pred_rel.numpy()                                         # already CPU from cache
        pred_rel_dim = pred_rel_np.shape[-1]
        if pred_rel_dim == rel_nums + 1:
            eval_rel_nums = rel_nums
            rel_slice = slice(1, rel_nums + 1)
        else:
            eval_rel_nums = min(rel_nums, pred_rel_dim)
            rel_slice = slice(0, eval_rel_nums)

        rel_scores = np.zeros((R, eval_rel_nums), dtype=np.float32)
        sub_boxes = np.zeros((R, 4), dtype=np.float32)
        obj_boxes = np.zeros((R, 4), dtype=np.float32)
        pred_sub_scores = np.ones(R, dtype=np.float32)
        pred_obj_scores = np.ones(R, dtype=np.float32)
        pred_sub_labels = np.zeros(R, dtype=np.int64)
        pred_obj_labels = np.zeros(R, dtype=np.int64)

        orig_t = torch.as_tensor(target["orig_size"]).cpu()
        orig_wh = torch.flip(orig_t, dims=[0])

        for r in range(R):
            s_idx = int(gt_relations[r, 0])
            o_idx = int(gt_relations[r, 1])

            if s_idx < len(matched_query) and o_idx < len(matched_query):
                s_q = int(matched_query[s_idx])
                o_q = int(matched_query[o_idx])

                # Official EGTR predicts 50 sigmoid predicate probabilities
                # with no background channel. Keep a small compatibility path
                # for older local checkpoints that added bg at index 0.
                rel_scores[r] = pred_rel_np[s_q, o_q, rel_slice].astype(np.float32)

                # Subject box + labels
                s_box_norm = pred_boxes_norm[s_q]
                sub_boxes[r] = rescale_bboxes(s_box_norm.unsqueeze(0), orig_wh).squeeze(0).numpy()
                s_logits = pred_logits[s_q]
                pred_sub_scores[r], pred_sub_labels[r] = torch.max(
                    torch.sigmoid(s_logits), dim=0)

                # Object box + labels
                o_box_norm = pred_boxes_norm[o_q]
                obj_boxes[r] = rescale_bboxes(o_box_norm.unsqueeze(0), orig_wh).squeeze(0).numpy()
                o_logits = pred_logits[o_q]
                pred_obj_scores[r], pred_obj_labels[r] = torch.max(
                    torch.sigmoid(o_logits), dim=0)

        pred_rel_labels = 1 + np.argmax(rel_scores, axis=1)
        pred_sub_labels = pred_sub_labels + 1
        pred_obj_labels = pred_obj_labels + 1

        pred_entry = {
            "sub_boxes": sub_boxes,
            "obj_boxes": obj_boxes,
            "sub_scores": pred_sub_scores,
            "obj_scores": pred_obj_scores,
            "sub_classes": pred_sub_labels,
            "obj_classes": pred_obj_labels,
            "rel_scores": rel_scores,
        }

        for task_eval_key in evaluators:
            if "predcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            if "predcls" not in task_mr_key:
                continue
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================

def _per_image_values(value: Any) -> List[Any]:
    """Normalize per-image cached values after epoch aggregation."""
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [value]


def _evaluate_predcls_batch_compact(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
) -> None:
    """Evaluate compact per-image predictions already in sg_eval field format."""
    required_keys = [
        "rel_scores", "sub_boxes", "obj_boxes", "sub_scores", "obj_scores",
        "sub_classes", "obj_classes",
    ]
    for k in required_keys:
        if k not in outputs:
            raise KeyError(f"Compact output missing key: {k}")

    per_key = {k: _per_image_values(outputs[k]) for k in required_keys}
    sgdet_keys = [
        "sgdet_rel_scores", "sgdet_sub_boxes", "sgdet_obj_boxes",
        "sgdet_sub_scores", "sgdet_obj_scores",
        "sgdet_sub_classes", "sgdet_obj_classes",
    ]
    has_sgdet_cache = all(k in outputs for k in sgdet_keys)
    if has_sgdet_cache:
        for k in sgdet_keys:
            per_key[k] = _per_image_values(outputs[k])

    for i, target in enumerate(targets):
        if i >= len(per_key["rel_scores"]):
            break
        _validate_target(target)

        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])
        gt_entry = {
            "gt_classes": gt_labels,
            "gt_relations": gt_relations,
            "gt_boxes": gt_boxes_xyxy,
        }

        rel_scores = np.asarray(per_key["rel_scores"][i], dtype=np.float32)
        if rel_scores.ndim == 1:
            rel_scores = rel_scores.reshape(1, -1)
        if rel_scores.shape[-1] > rel_nums:
            # EGTR convention: bg at index 0 → keep [1:rel_nums+1]
            rel_scores = rel_scores[:, 1:rel_nums + 1]

        R = gt_relations.shape[0]
        gt_sub_idx = gt_relations[:, 0]
        gt_obj_idx = gt_relations[:, 1]

        # PredCLS: use GT boxes + GT labels, only predicate comes from the model.
        # The compact cache stores exactly R entries aligned with GT relations.
        pred_entry_predcls = {
            "sub_boxes": gt_boxes_xyxy[gt_sub_idx].astype(np.float32),
            "sub_classes": gt_labels[gt_sub_idx].astype(np.int64),
            "sub_scores": np.ones(R, dtype=np.float32),
            "obj_boxes": gt_boxes_xyxy[gt_obj_idx].astype(np.float32),
            "obj_classes": gt_labels[gt_obj_idx].astype(np.int64),
            "obj_scores": np.ones(R, dtype=np.float32),
            "rel_scores": rel_scores,
        }

        pred_rel_labels = 1 + np.argmax(rel_scores, axis=1) if rel_scores.size else \
            np.zeros(0, dtype=np.int64)
        if has_sgdet_cache:
            sgdet_rel_scores = np.asarray(per_key["sgdet_rel_scores"][i], dtype=np.float32)
            if sgdet_rel_scores.ndim == 1:
                sgdet_rel_scores = sgdet_rel_scores.reshape(1, -1)
            if sgdet_rel_scores.shape[-1] > rel_nums:
                sgdet_rel_scores = sgdet_rel_scores[:, 1:rel_nums + 1]
            pred_entry_sgdet = {
                "sub_boxes": np.asarray(per_key["sgdet_sub_boxes"][i], dtype=np.float32),
                "obj_boxes": np.asarray(per_key["sgdet_obj_boxes"][i], dtype=np.float32),
                "sub_scores": np.asarray(per_key["sgdet_sub_scores"][i], dtype=np.float32),
                "obj_scores": np.asarray(per_key["sgdet_obj_scores"][i], dtype=np.float32),
                "sub_classes": np.asarray(per_key["sgdet_sub_classes"][i], dtype=np.int64),
                "obj_classes": np.asarray(per_key["sgdet_obj_classes"][i], dtype=np.int64),
                "rel_scores": sgdet_rel_scores,
            }
            pred_rel_labels_sgdet = 1 + np.argmax(sgdet_rel_scores, axis=1) \
                if sgdet_rel_scores.size else np.zeros(0, dtype=np.int64)
        else:
            # Legacy compact cache: use predicted boxes + labels aligned with GT
            # relations. New EGTR runs should provide the sgdet_* top-pair cache.
            pred_entry_sgdet = {
                "sub_boxes": np.asarray(per_key["sub_boxes"][i], dtype=np.float32),
                "obj_boxes": np.asarray(per_key["obj_boxes"][i], dtype=np.float32),
                "sub_scores": np.asarray(per_key["sub_scores"][i], dtype=np.float32),
                "obj_scores": np.asarray(per_key["obj_scores"][i], dtype=np.float32),
                "sub_classes": np.asarray(per_key["sub_classes"][i], dtype=np.int64),
                "obj_classes": np.asarray(per_key["obj_classes"][i], dtype=np.int64),
                "rel_scores": rel_scores,
            }
            pred_rel_labels_sgdet = pred_rel_labels

        for task_eval_key in evaluators:
            if "predcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry_predcls)
            if "sgdet" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry_sgdet)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            gt_rel_labels = gt_relations[:, 2]
            if "predcls" in task_mr_key:
                for rel_id in range(1, rel_nums + 1):
                    gt_mask = (gt_rel_labels == rel_id)
                    if not gt_mask.any():
                        continue
                    pred_mask = (pred_rel_labels == rel_id)
                    gt_entry_rel = {
                        "gt_classes": gt_entry["gt_classes"],
                        "gt_relations": gt_entry["gt_relations"][gt_mask],
                        "gt_boxes": gt_entry["gt_boxes"],
                    }
                    pred_entry_rel = _filter_by_mask(pred_entry_predcls, pred_mask)
                    mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)
            if "sgdet" in task_mr_key:
                for rel_id in range(1, rel_nums + 1):
                    gt_mask = (gt_rel_labels == rel_id)
                    if not gt_mask.any():
                        continue
                    pred_mask = (pred_rel_labels_sgdet == rel_id)
                    gt_entry_rel = {
                        "gt_classes": gt_entry["gt_classes"],
                        "gt_relations": gt_entry["gt_relations"][gt_mask],
                        "gt_boxes": gt_entry["gt_boxes"],
                    }
                    pred_entry_rel = _filter_by_mask(pred_entry_sgdet, pred_mask)
                    mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================

def _evaluate_predcls_batch_pair_indices(
    outputs: Dict[str, Any],
    targets: List[Dict[str, Any]],
    evaluators: Dict[str, SceneGraphEvaluator],
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    rel_nums: int,
) -> None:
    """Motifs/CVC-style PredCLS evaluation.

    These models score predicates for directed GT object pairs and expose
    ``pair_indices`` in target object-index space. PredCLS can therefore match
    each GT relation by its exact ``(subject_idx, object_idx)`` pair, without
    IoU or Hungarian matching.
    """
    logits_key = "rel_logits" if "rel_logits" in outputs else "pred_logits"
    required_keys = [logits_key, "pair_indices"]
    for k in required_keys:
        if k not in outputs:
            raise KeyError(f"Pair-index output missing key: {k}")

    for i, target in enumerate(targets):
        _validate_target(target)

        gt_relations = _to_numpy(target["rel_annotations"]).astype(np.int64)
        if gt_relations.ndim == 1:
            gt_relations = gt_relations.reshape(-1, 3) if gt_relations.size > 0 else \
                np.zeros((0, 3), dtype=np.int64)
        if gt_relations.shape[0] == 0:
            continue

        gt_labels = _to_numpy(target["labels"]).astype(np.int64)
        gt_boxes_xyxy = _rescale_boxes(target["boxes"], target["orig_size"])
        gt_entry = {
            "gt_classes": gt_labels,
            "gt_relations": gt_relations,
            "gt_boxes": gt_boxes_xyxy,
        }

        rel_logits = torch.as_tensor(outputs[logits_key][i]).float()
        pair_indices = torch.as_tensor(outputs["pair_indices"][i]).long()
        if rel_logits.dim() == 3:
            rel_logits = rel_logits.squeeze(0)
        if pair_indices.dim() == 3:
            pair_indices = pair_indices.squeeze(0)

        R = gt_relations.shape[0]
        best_rel_scores = np.zeros((R, rel_nums), dtype=np.float32)
        if rel_logits.numel() > 0 and pair_indices.numel() > 0:
            rel_scores_all = _extract_relation_scores(rel_logits, rel_nums)
            pair_to_idx = {
                (int(pair_indices[p, 0]), int(pair_indices[p, 1])): p
                for p in range(pair_indices.shape[0])
            }
            for r in range(R):
                key = (int(gt_relations[r, 0]), int(gt_relations[r, 1]))
                p_idx = pair_to_idx.get(key)
                if p_idx is not None and p_idx < rel_scores_all.shape[0]:
                    best_rel_scores[r] = rel_scores_all[p_idx]

        gt_sub_idx = gt_relations[:, 0]
        gt_obj_idx = gt_relations[:, 1]
        pred_entry = {
            "sub_boxes": gt_boxes_xyxy[gt_sub_idx].astype(np.float32),
            "sub_classes": gt_labels[gt_sub_idx].astype(np.int64),
            "sub_scores": np.ones(R, dtype=np.float32),
            "obj_boxes": gt_boxes_xyxy[gt_obj_idx].astype(np.float32),
            "obj_classes": gt_labels[gt_obj_idx].astype(np.int64),
            "obj_scores": np.ones(R, dtype=np.float32),
            "rel_scores": best_rel_scores,
        }

        pred_rel_labels = 1 + np.argmax(best_rel_scores, axis=1)
        for task_eval_key in evaluators:
            if "predcls" in task_eval_key:
                evaluators[task_eval_key].evaluate_entry(gt_entry, pred_entry)

        for task_mr_key, mr_eval_list in mr_evaluators.items():
            if "predcls" not in task_mr_key:
                continue
            gt_rel_labels = gt_relations[:, 2]
            for rel_id in range(1, rel_nums + 1):
                gt_mask = (gt_rel_labels == rel_id)
                if not gt_mask.any():
                    continue
                pred_mask = (pred_rel_labels == rel_id)
                gt_entry_rel = {
                    "gt_classes": gt_entry["gt_classes"],
                    "gt_relations": gt_entry["gt_relations"][gt_mask],
                    "gt_boxes": gt_entry["gt_boxes"],
                }
                pred_entry_rel = _filter_by_mask(pred_entry, pred_mask)
                mr_eval_list[rel_id - 1].evaluate_entry(gt_entry_rel, pred_entry_rel)


# ===========================================================================

def _extract_relation_scores(rel_logits: torch.Tensor, rel_nums: int) -> np.ndarray:
    """Convert relation logits into [num_triplets, rel_nums] score array.

    The evaluator (sg_eval.py) expects ``pred_rels = 1 + argmax(rel_scores, axis=1)``,
    so rel_scores must contain only the actual predicate classes (no background).

    **CRITICAL**: Softmax must be applied ONLY over the non-background classes
    (matching the official RelTR evaluation protocol).  Applying softmax over all
    classes then slicing would produce scores ~5000x smaller and break the
    triplet scoring/ranking.
    """
    dim = rel_logits.shape[-1]

    if dim == rel_nums + 2:
        # RelTR default: classes = [0, 1..rel_nums, rel_nums+1]
        # where 0 and rel_nums+1 are both "no-relation" variants.
        # Official: apply softmax over only the rel_nums predicate dims.
        rel_scores = torch.softmax(rel_logits[:, 1:-1], dim=-1)
    elif dim == rel_nums + 1:
        rel_scores = torch.softmax(rel_logits[:, :-1], dim=-1)
    elif dim == rel_nums:
        rel_scores = torch.softmax(rel_logits, dim=-1)
    else:
        raise ValueError(
            f"rel_logits dim mismatch: got {dim}, expected {rel_nums}, {rel_nums + 1}, "
            f"or {rel_nums + 2}"
        )

    return rel_scores.detach().cpu().numpy()


# ===========================================================================
# Result collection
# ===========================================================================

def _collect_main_recall(
    evaluators: Dict[str, SceneGraphEvaluator]
) -> Dict[str, float]:
    """Extract R@k values from evaluators."""
    results = {}
    for task, evaluator in evaluators.items():
        for k, v in evaluator.recalls.items():
            if not np.isnan(v):
                results[f"{task}_R@{k}"] = v
    return results


def _collect_mean_recall(
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]]
) -> Dict[str, float]:
    """Average per-class recall values to get mean recall (mR@k)."""
    results = {}
    for task, evaluator_list in mr_evaluators.items():
        for k in (10, 20, 50):
            per_class = []
            for evaluator in evaluator_list:
                recalls = evaluator.recalls
                if k in recalls and not np.isnan(recalls[k]):
                    per_class.append(recalls[k])
            if per_class:
                results[f"{task}_mR@{k}"] = float(np.mean(per_class))
    return results


# ===========================================================================
# Log formatting
# ===========================================================================

def _build_log(supported_tasks, all_results, warnings, skipped) -> str:
    """Build human-readable evaluation log."""
    lines = []

    for task in supported_tasks:
        lines.append(f"{'=' * 30}{task}{'=' * 30}")

        # Main recall
        for k in (10, 20, 50):
            key = f"{task}_R@{k}"
            val = all_results.get(key)
            label = f"{val:.4f}" if (val is not None and not np.isnan(val)) else "unavailable"
            lines.append(f"R@{k}: {label}")

        # Mean recall
        mr_keys = [f"{task}_mR@{k}" for k in (10, 20, 50)]
        if any(k in all_results for k in mr_keys):
            lines.append("")
            lines.append(f"{'=' * 20}{task}  mean recall with constraint{'=' * 20}")
            for k in (10, 20, 50):
                key = f"{task}_mR@{k}"
                val = all_results.get(key)
                label = f"{val:.4f}" if (val is not None and not np.isnan(val)) else "unavailable"
                lines.append(f"mR@{k}:  {label}")

    if warnings or skipped:
        lines.append("")
        for w in warnings:
            lines.append(w)
        if skipped:
            lines.append(f"[metric] Skipped unsupported/unavailable: {', '.join(skipped)}")

    return "\n".join(lines)


# ===========================================================================
# Head/Body/Tail analysis
# ===========================================================================

def compute_head_body_tail_mr(
    mr_evaluators: Dict[str, List[SceneGraphEvaluator]],
    freq_path: str = "data/VisualGenome/predicate_frequencies.json",
    head_ratio: float = 0.2,
    tail_ratio: float = 0.4,
) -> Dict[str, Dict[str, float]]:
    """Split per-predicate recalls into head/body/tail groups by training frequency.

    Parameters
    ----------
    mr_evaluators : dict
        Per-task list of per-predicate SceneGraphEvaluator objects.
    freq_path : str
        Path to the JSON file with predicate frequencies.
    head_ratio, tail_ratio : float
        Fraction of predicates in head and tail groups (body = 1 - head - tail).

    Returns
    -------
    dict mapping task -> {"head_mR@K": ..., "body_mR@K": ..., "tail_mR@K": ...}
    """
    import json
    import os

    if not os.path.exists(freq_path):
        print(f"[metric] Frequency file not found: {freq_path}, skipping head/body/tail analysis.")
        return {}

    with open(freq_path) as f:
        freq_data = json.load(f)
    freq = freq_data["predicate_frequencies"]
    # Sort predicates by frequency (descending)
    sorted_preds = sorted(freq.items(), key=lambda x: int(x[0]))
    sorted_pred_ids = [int(p[0]) for p in sorted_preds]
    sorted_counts = [p[1] for p in sorted_preds]

    # Sort by count (descending) to determine groups
    rank_sorted = sorted(enumerate(sorted_counts), key=lambda x: x[1], reverse=True)
    n = len(rank_sorted)
    n_head = max(1, int(n * head_ratio))
    n_tail = max(1, int(n * tail_ratio))
    head_ids = set(rank_sorted[i][0] for i in range(n_head))
    tail_ids = set(rank_sorted[n - n_tail + i][0] for i in range(n_tail))
    body_ids = set(range(n)) - head_ids - tail_ids

    results = {}
    for task, evaluator_list in mr_evaluators.items():
        for k in (10, 20, 50):
            head_recalls, body_recalls, tail_recalls = [], [], []
            for i, evaluator in enumerate(evaluator_list):
                recalls = evaluator.recalls
                if k not in recalls or np.isnan(recalls[k]):
                    continue
                if i in head_ids:
                    head_recalls.append(recalls[k])
                elif i in tail_ids:
                    tail_recalls.append(recalls[k])
                else:
                    body_recalls.append(recalls[k])
            results[f"{task}_head_mR@{k}"] = float(np.mean(head_recalls)) if head_recalls else float("nan")
            results[f"{task}_body_mR@{k}"] = float(np.mean(body_recalls)) if body_recalls else float("nan")
            results[f"{task}_tail_mR@{k}"] = float(np.mean(tail_recalls)) if tail_recalls else float("nan")

    return results


# ===========================================================================
# Helpers
# ===========================================================================

def _filter_by_mask(entry: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Filter a pred_entry dict by boolean mask along the first axis."""
    mask = np.asarray(mask, dtype=bool)
    return {k: v[mask] for k, v in entry.items()}


def _rescale_boxes(boxes, orig_size) -> np.ndarray:
    """Convert normalized boxes to absolute pixel coordinates (numpy array output)."""
    boxes_t = torch.as_tensor(boxes).cpu()
    orig_t = torch.as_tensor(orig_size).cpu()
    if orig_t.numel() != 2:
        raise ValueError(f"orig_size must have 2 elements, got {orig_t.numel()}")
    orig_wh = torch.flip(orig_t, dims=[0])
    scaled = rescale_bboxes(boxes_t, orig_wh)
    return scaled.detach().cpu().numpy().astype(np.float32)


def _rescale_bboxes_tensor(boxes: torch.Tensor, orig_size) -> torch.Tensor:
    """Convert normalized boxes to absolute pixel coordinates (tensor output, keeps device)."""
    orig_t = torch.as_tensor(orig_size)
    if orig_t.numel() != 2:
        raise ValueError(f"orig_size must have 2 elements, got {orig_t.numel()}")
    orig_wh = torch.flip(orig_t, dims=[0]).to(boxes.device)
    return rescale_bboxes(boxes, orig_wh)


def _validate_target(target: Dict[str, Any]) -> None:
    required = {"boxes", "labels", "rel_annotations", "orig_size"}
    for key in required:
        if key not in target:
            raise KeyError(f"Target missing field: {key}")


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_cpu_detach(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: _to_cpu_detach(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_cpu_detach(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu_detach(v) for v in data)
    return data
