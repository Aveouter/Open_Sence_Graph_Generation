"""
Scene Graph Recall Evaluation.

Core recall computation adapted from Danfei Xu's scene graph benchmark.
Evaluates scene graph predictions by matching (subject, predicate, object)
triplets to ground truth using exact-label matching followed by IoU-based
bounding box filtering.

Supports: sgdet, sgcls, predcls, preddet, phrdet
"""

from functools import reduce

import numpy as np
from lib.pytorch_misc import intersect_2d
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps


# ===========================================================================
# Public API
# ===========================================================================

class SceneGraphEvaluator:
    """Accumulates per-sample recall results and reports Recall@K statistics.

    Usage:
        evaluator = SceneGraphEvaluator(mode="sgdet")
        for gt, pred in zip(gt_entries, pred_entries):
            evaluator.evaluate_entry(gt, pred)
        print(evaluator.recalls)  # {20: 0.15, 50: 0.28, 100: 0.35}
    """

    def __init__(self, mode, multiple_preds=False):
        self.mode = mode
        self.multiple_preds = multiple_preds
        self.result_dict = {f"{mode}_recall": {10: [], 20: [], 50: [], 100: []}}

    # ---- factories --------------------------------------------------------

    @classmethod
    def all_modes(cls, **kwargs):
        return {m: cls(mode=m, **kwargs) for m in ("sgdet", "sgcls", "predcls")}

    @classmethod
    def vrd_modes(cls, **kwargs):
        return {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ("preddet", "phrdet")}

    # ---- per-sample evaluation -------------------------------------------

    def evaluate_entry(self, gt_entry, pred_entry, iou_thresh=0.5):
        """Evaluate a single image and append its Recall@K to internal storage."""
        _evaluate_single_entry(
            gt_entry, pred_entry, self.mode, self.result_dict,
            iou_thresh=iou_thresh, multiple_preds=self.multiple_preds,
        )

    # ---- result access ---------------------------------------------------

    @property
    def recalls(self):
        """Return {k: mean_recall} for k in {20, 50, 100} that have data."""
        recall_dict = self.result_dict.get(f"{self.mode}_recall", {})
        return {
            k: float(np.mean(v))
            for k, v in recall_dict.items()
            if len(v) > 0
        }

    def save(self, path):
        np.save(path, self.result_dict)


# ===========================================================================
# Per-sample evaluation (private)
# ===========================================================================

def _evaluate_single_entry(gt_entry, pred_entry, mode, result_dict,
                           multiple_preds=False, iou_thresh=0.5):
    """Process one gt/pred pair and append Recall@K values to result_dict."""
    gt_rels = gt_entry["gt_relations"]
    gt_boxes = gt_entry["gt_boxes"].astype(float)
    gt_classes = gt_entry["gt_classes"]

    rel_scores = pred_entry["rel_scores"]
    pred_rel_labels = 1 + rel_scores.argmax(axis=1)
    pred_rel_scores = rel_scores.max(axis=1)

    pred_to_gt, _, _ = evaluate_recall(
        gt_rels=gt_rels,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        pred_rels=pred_rel_labels,
        sub_boxes=pred_entry["sub_boxes"],
        obj_boxes=pred_entry["obj_boxes"],
        sub_scores=pred_entry["sub_scores"],
        obj_scores=pred_entry["obj_scores"],
        pred_scores=pred_rel_scores,
        sub_labels=pred_entry["sub_classes"],
        obj_labels=pred_entry["obj_classes"],
        iou_thresh=iou_thresh,
        phrdet=(mode == "phrdet"),
    )

    recall_bucket = result_dict[f"{mode}_recall"]
    for k in recall_bucket:
        top_k = pred_to_gt[:k]
        if len(top_k) == 0:
            matched = np.array([], dtype=np.int64)
        else:
            matched = reduce(np.union1d, top_k)
        recall_bucket[k].append(float(len(matched)) / gt_rels.shape[0])


# ===========================================================================
# Core recall computation
# ===========================================================================

def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, sub_boxes, obj_boxes,
                    sub_scores, obj_scores, pred_scores,
                    sub_labels, obj_labels,
                    iou_thresh=0.5, phrdet=False):
    """Match predicted relation triplets to ground truth.

    Returns per-prediction matching indices sorted by descending confidence
    (product of subject, object, and relation scores).

    Parameters
    ----------
    gt_rels : ndarray [M, 3] int64
        GT relations as (subject_box_idx, object_box_idx, relation_label).
    gt_boxes : ndarray [B, 4] float32
        GT bounding boxes.
    gt_classes : ndarray [B] int64
        GT class labels.
    pred_rels : ndarray [N] int64
        Predicted relation labels (1-indexed).
    sub_boxes : ndarray [N, 4] float32
    obj_boxes : ndarray [N, 4] float32
    sub_scores : ndarray [N] float32
    obj_scores : ndarray [N] float32
    pred_scores : ndarray [N] float32
        Confidence score for each predicted relation.
    sub_labels : ndarray [N] int64
    obj_labels : ndarray [N] int64
    iou_thresh : float
    phrdet : bool
        Phrase detection mode (union-box IoU).

    Returns
    -------
    pred_to_gt : list[list[int]]
        pred_to_gt[i] = list of GT indices matched to prediction i.
    _ : None
        Compatibility placeholder.
    rel_scores : ndarray [N, 3]
        Stacked (sub_score, obj_score, pred_score).
    """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    assert gt_rels.shape[0] > 0, "gt_rels must be non-empty"

    # ---- build GT triplets ------------------------------------------------
    gt_triplets, gt_triplet_boxes, _ = _build_triplets(
        relation_labels=gt_rels[:, 2],
        sub_obj_pairs=gt_rels[:, :2],
        class_labels=gt_classes,
        boxes=gt_boxes,
    )

    # ---- build predicted triplets -----------------------------------------
    pred_triplets = np.column_stack((sub_labels, pred_rels, obj_labels))
    pred_triplet_boxes = np.column_stack((sub_boxes, obj_boxes))
    rel_scores = np.column_stack((sub_scores, obj_scores, pred_scores))

    # sort by product of three confidence scores (descending)
    sort_idx = rel_scores.prod(axis=1).argsort()[::-1]
    pred_triplets = pred_triplets[sort_idx]
    pred_triplet_boxes = pred_triplet_boxes[sort_idx]
    rel_scores = rel_scores[sort_idx]

    # ---- match ------------------------------------------------------------
    pred_to_gt = _match_predictions(
        gt_triplets, pred_triplets,
        gt_triplet_boxes, pred_triplet_boxes,
        iou_thresh, phrdet=phrdet,
    )

    return pred_to_gt, None, rel_scores


# ===========================================================================
# Triplet building
# ===========================================================================

def _build_triplets(relation_labels, sub_obj_pairs, class_labels, boxes,
                    rel_scores=None, class_scores=None):
    """Convert raw relation annotations into (sub_cls, rel, obj_cls) triplets.

    Parameters
    ----------
    relation_labels : ndarray [R]
    sub_obj_pairs : ndarray [R, 2]
        Each row is (subject_idx, object_idx) into class_labels / boxes.
    class_labels : ndarray [B]
    boxes : ndarray [B, 4]
    rel_scores : ndarray [R] or None
    class_scores : ndarray [B] or None

    Returns
    -------
    triplets : ndarray [R, 3]  (sub_cls, rel_label, obj_cls)
    triplet_boxes : ndarray [R, 8]  (sub_box, obj_box)
    triplet_scores : ndarray [R, 3] or None  (sub_cls_score, obj_cls_score, rel_score)
    """
    assert relation_labels.shape[0] == sub_obj_pairs.shape[0]

    sub_obj_classes = class_labels[sub_obj_pairs[:, :2]]
    triplets = np.column_stack((
        sub_obj_classes[:, 0],
        relation_labels,
        sub_obj_classes[:, 1],
    ))
    triplet_boxes = np.column_stack((
        boxes[sub_obj_pairs[:, 0]],
        boxes[sub_obj_pairs[:, 1]],
    ))

    triplet_scores = None
    if rel_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_obj_pairs[:, 0]],
            class_scores[sub_obj_pairs[:, 1]],
            rel_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


# ===========================================================================
# Pred-to-GT matching
# ===========================================================================

def _match_predictions(gt_triplets, pred_triplets,
                       gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """Match each predicted triplet to zero or more GT triplets.

    A match requires:
      1. Exact match on (sub_class, relation_label, obj_class), AND
      2. IoU(sub_box, gt_sub_box) >= thresh AND IoU(obj_box, gt_obj_box) >= thresh
         (or union-box IoU in phrdet mode).

    Parameters
    ----------
    gt_triplets : ndarray [M, 3]
    pred_triplets : ndarray [N, 3]  (sorted by confidence, descending)
    gt_boxes : ndarray [M, 8]
    pred_boxes : ndarray [N, 8]
    iou_thresh : float
    phrdet : bool

    Returns
    -------
    pred_to_gt : list[list[int]]  length N, each inner list contains GT indices
    """
    # [M, N] boolean: which (gt, pred) pairs have exact triplet match
    exact_matches = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = exact_matches.any(axis=1)

    pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]

    for gt_idx, gt_box, match_mask in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        exact_matches[gt_has_match],
    ):
        candidate_boxes = pred_boxes[match_mask]
        candidate_indices = np.where(match_mask)[0]

        if phrdet:
            ious = _compute_union_box_iou(gt_box, candidate_boxes)
        else:
            ious = _compute_sub_obj_min_iou(gt_box, candidate_boxes)

        for pred_idx in candidate_indices[ious >= iou_thresh]:
            pred_to_gt[pred_idx].append(int(gt_idx))

    return pred_to_gt


def _compute_union_box_iou(gt_box, pred_boxes):
    """IoU between the GT union box and each predicted union box.

    The union box is the bounding box that tightly encloses both subject and
    object boxes: [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)].
    """
    gt_union = gt_box.reshape(2, 4)
    gt_union = np.concatenate((gt_union.min(axis=0)[:2], gt_union.max(axis=0)[2:]), axis=0)

    pred_union = pred_boxes.reshape(-1, 2, 4)
    pred_union = np.concatenate(
        (pred_union.min(axis=1)[:, :2], pred_union.max(axis=1)[:, 2:]),
        axis=1,
    )

    return bbox_overlaps(gt_union[None], pred_union)[0]


def _compute_sub_obj_min_iou(gt_box, pred_boxes):
    """Minimum of subject-IoU and object-IoU for each prediction.

    Both subject and object boxes must satisfy the IoU threshold, so we take
    the smaller (more restrictive) of the two IoU values.
    """
    sub_iou = bbox_overlaps(gt_box[None, :4], pred_boxes[:, :4])[0]
    obj_iou = bbox_overlaps(gt_box[None, 4:], pred_boxes[:, 4:])[0]
    return np.minimum(sub_iou, obj_iou)


# ===========================================================================
# Mean Recall aggregation
# ===========================================================================

def aggregate_mean_recall(evaluators_by_rel, mode):
    """Average Recall@K across per-relationship evaluators.

    Parameters
    ----------
    evaluators_by_rel : list of (rel_id, rel_name, {mode: SceneGraphEvaluator})
    mode : str

    Returns
    -------
    dict with keys "R@20", "R@50", "R@100"
    """
    mR = {"R@20": 0.0, "R@50": 0.0, "R@100": 0.0}
    valid = 0

    for _rel_id, _rel_name, eval_dict in evaluators_by_rel:
        recalls = eval_dict[mode].recalls
        if any(np.isnan(recalls.get(k, np.nan)) for k in ("R@20", "R@50", "R@100")):
            continue
        for k in mR:
            mR[k] += recalls.get(k, 0.0)
        valid += 1

    if valid > 0:
        for k in mR:
            mR[k] /= valid

    return mR
