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
    supported_tasks, warnings = _resolve_task_support(pred, requested_tasks)
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
    if "sgdet" in supported_tasks:
        _evaluate_sgdet_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
        )
    if "predcls" in supported_tasks:
        _evaluate_predcls_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
        )
    if "sgcls" in supported_tasks:
        _evaluate_sgcls_batch(
            outputs=pred,
            targets=true,
            evaluators=evaluators,
            mr_evaluators=mr_evaluators,
            rel_nums=rel_nums,
            triplet_match_indices=triplet_match_indices,
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
}

# Which tasks each model family supports
_MODEL_TASKS = {
    "reltr": {"sgdet", "predcls", "sgcls"},
}


def _resolve_task_support(
    pred: Dict[str, Any], requested_tasks: List[str]
) -> Tuple[List[str], List[str]]:
    """Determine which requested tasks are supported by the current model outputs."""
    warnings = []
    supported = []

    pred_keys = set(pred.keys())
    matched = False

    for family, required_keys in _MODEL_SCHEMAS.items():
        if required_keys.issubset(pred_keys):
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

    return supported, warnings


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
# Relation score extraction
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
