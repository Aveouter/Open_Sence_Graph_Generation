from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.evaluation.sg_eval import SceneGraphEvaluator

# Importing ``utils.box_ops`` normally executes utils/__init__.py, which eagerly
# imports the optional Lightning stack. Metric protocol tests do not need it.
if "utils" not in sys.modules:
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(Path(__file__).resolve().parents[2] / "utils")]
    sys.modules["utils"] = utils_package

from src.core.metrics import (
    _collect_mean_recall,
    _extract_relation_scores,
    _inclusive_iou_rows,
    _relation_nms_pair_scores,
    metric,
)
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps


class RASGGPredClsProtocolTest(unittest.TestCase):
    @staticmethod
    def _brute_relation_nms(boxes, classes, pairs, scores,
                            nms_threshold=0.6, l21_threshold=0.7):
        """Literal NumPy transcription of the upstream PE-Net evaluator."""
        ious = bbox_overlaps(boxes, boxes)
        sub_ious = ious[pairs[:, 0]][:, pairs[:, 0]]
        obj_ious = ious[pairs[:, 1]][:, pairs[:, 1]]
        rel_ious = np.minimum(sub_ious, obj_ious)
        sub_labels = classes[pairs[:, 0]]
        obj_labels = classes[pairs[:, 1]]
        l21 = np.sqrt(
            np.square(scores[:, None, :]) + np.square(scores[None, :, :])
        ).sum(axis=-1)
        overlap = (
            (rel_ious >= nms_threshold)
            & (sub_labels[:, None] == sub_labels[None, :])
            & (obj_labels[:, None] == obj_labels[None, :])
            & (l21 > l21_threshold)
        )
        overlap = np.repeat(overlap[:, :, None], scores.shape[1], axis=2)

        working = scores.copy()
        working[:, 0] = 0.0
        labels = np.zeros(len(pairs), dtype=np.int64)
        for _ in range(len(pairs)):
            row, predicate = np.unravel_index(working.argmax(), working.shape)
            if labels[row] == 0:
                labels[row] = predicate
            working[overlap[row, :, predicate], predicate] = 0.0
            working[row] = -1.0
        selected_scores = scores[np.arange(len(pairs)), labels]
        return labels, selected_scores

    @staticmethod
    def _synthetic_all_pair_case():
        """Two GT predicates; the second correct prediction is global rank 21."""
        num_objects = 22
        labels = torch.arange(1, num_objects + 1, dtype=torch.long)

        boxes = []
        for idx in range(num_objects):
            row, col = divmod(idx, 5)
            boxes.append([0.08 + col * 0.18, 0.08 + row * 0.18, 0.04, 0.04])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        pairs = torch.tensor(
            [(s, o) for s in range(num_objects) for o in range(num_objects) if s != o],
            dtype=torch.long,
        )
        pair_to_row = {tuple(pair): idx for idx, pair in enumerate(pairs.tolist())}

        # Layout is [background, predicate 1, predicate 2]. Background pairs
        # retain tiny foreground probabilities after the full 3-way softmax.
        logits = torch.full((len(pairs), 3), -10.0)
        logits[:, 0] = 10.0

        gt_one = pair_to_row[(0, 1)]
        gt_two = pair_to_row[(2, 3)]
        logits[gt_one] = torch.tensor([0.0, 6.0, -10.0])

        distractors = [
            row
            for pair, row in pair_to_row.items()
            if pair not in {(0, 1), (2, 3)}
        ][:19]
        for offset, row in enumerate(distractors):
            logits[row] = torch.tensor([0.0, 5.0 - offset * 0.01, -10.0])

        # Nineteen distractors plus predicate-1 GT occupy ranks 1..20.
        logits[gt_two] = torch.tensor([0.0, -10.0, 1.0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "rel_annotations": torch.tensor([[0, 1, 1], [2, 3, 2]]),
            "orig_size": torch.tensor([100, 100]),
            "size": torch.tensor([100, 100]),
        }
        output = {
            "model_family": "motifs",
            "rel_logits": [logits],
            "pair_indices": [pairs],
            "sub_boxes": [boxes[pairs[:, 0]]],
            "obj_boxes": [boxes[pairs[:, 1]]],
            "obj_labels": [labels],
            "predicate_bg_index": "first",
            "relation_softmax_scope": "all",
        }
        return output, target

    def test_all_pairs_and_predicates_share_global_topk(self) -> None:
        output, target = self._synthetic_all_pair_case()

        result, _ = metric(
            pred=output,
            true=[target],
            metrics=[
                "predcls_R@20",
                "predcls_R@50",
                "predcls_mR@20",
                "predcls_mR@50",
            ],
            rel_nums=2,
            entity_nums=23,
        )

        self.assertAlmostEqual(result["predcls_R@20"], 0.5)
        self.assertAlmostEqual(result["predcls_R@50"], 1.0)
        self.assertAlmostEqual(result["predcls_mR@20"], 0.5)
        self.assertAlmostEqual(result["predcls_mR@50"], 1.0)

    def test_empty_predictions_record_zero_recall(self) -> None:
        output, target = self._synthetic_all_pair_case()
        output["rel_logits"] = [torch.empty((0, 3))]
        output["pair_indices"] = [torch.empty((0, 2), dtype=torch.long)]

        result, _ = metric(
            pred=output,
            true=[target],
            metrics=["predcls_R@20", "predcls_mR@20"],
            rel_nums=2,
            entity_nums=23,
        )

        self.assertEqual(result["predcls_R@20"], 0.0)
        self.assertEqual(result["predcls_mR@20"], 0.0)

    def test_dual_model_and_evaluator_targets_match_official_vg_loading(self) -> None:
        # Official VG removes a degenerate box from the model's proposal list
        # but keeps it (and its relations) in evaluator GT. Model pair indices
        # therefore index ``boxes`` while GT relations index ``eval_boxes``.
        eval_boxes = torch.tensor([
            [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.0, 0.1],  # removed from model input
            [0.8, 0.8, 0.1, 0.1],
        ])
        model_boxes = eval_boxes[[0, 2]]
        pairs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        logits = torch.tensor([
            [0.0, 8.0, -8.0],
            [8.0, 0.0, 0.0],
        ])
        target = {
            "boxes": model_boxes,
            "labels": torch.tensor([1, 3]),
            "rel_annotations": torch.tensor([[0, 1, 1]]),
            "eval_boxes": eval_boxes,
            "eval_labels": torch.tensor([1, 2, 3]),
            "eval_rel_annotations": torch.tensor([
                [0, 2, 1],  # matchable through remapped model pair 0 -> 1
                [0, 1, 2],  # impossible: evaluator-only degenerate object
            ]),
            "orig_size": torch.tensor([100, 100]),
            "size": torch.tensor([100, 100]),
        }
        output = {
            "model_family": "motifs",
            "rel_logits": [logits],
            "pair_indices": [pairs],
            "sub_boxes": [model_boxes[pairs[:, 0]]],
            "obj_boxes": [model_boxes[pairs[:, 1]]],
            "obj_labels": [target["labels"]],
            "predicate_bg_index": "first",
            "relation_softmax_scope": "all",
        }

        result, _ = metric(
            pred=output,
            true=[target],
            metrics=["predcls_R@20", "predcls_mR@20"],
            rel_nums=2,
            entity_nums=4,
        )
        self.assertAlmostEqual(result["predcls_R@20"], 0.5)
        self.assertAlmostEqual(result["predcls_mR@20"], 0.5)

    def test_mean_recall_has_fixed_foreground_denominator(self) -> None:
        evaluators = [SceneGraphEvaluator("predcls") for _ in range(3)]
        for evaluator in evaluators[:2]:
            evaluator.result_dict["predcls_recall"][20].append(1.0)

        result = _collect_mean_recall({"predcls": evaluators})

        self.assertAlmostEqual(result["predcls_mR@20"], 2.0 / 3.0)

    def test_bg_first_scores_softmax_before_background_removal(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0]])

        scores = _extract_relation_scores(
            logits,
            rel_nums=2,
            predicate_bg_index="first",
            softmax_scope="all",
        )

        expected = torch.softmax(logits, dim=-1)[:, 1:].numpy()
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-7)

    def test_hardcoded_vg_names_match_dataset_dictionary(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "src/models/ra_sgg.py").read_text())
        assignments = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "VG_OBJECT_NAMES",
                "VG_PREDICATE_NAMES",
            }:
                assignments[target.id] = ast.literal_eval(node.value)

        # Digests were generated from the ordered idx_to_label and
        # idx_to_predicate arrays in the official VG-SGG dictionary. Keeping
        # the fixture compact makes this test runnable in a clean checkout,
        # where the full Visual Genome dataset is intentionally gitignored.
        object_digest = hashlib.sha256(json.dumps(
            assignments["VG_OBJECT_NAMES"],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        predicate_digest = hashlib.sha256(json.dumps(
            assignments["VG_PREDICATE_NAMES"],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()).hexdigest()

        self.assertEqual(
            object_digest,
            "38f7134833b31e5d31b2c3b12d3eda35a4b3d1ccf247c43d8638eac9356233e1",
        )
        self.assertEqual(
            predicate_digest,
            "62d60f06e4242b54302d9b2c7f1c4fd24daae422b407d6013102c980101553c4",
        )

    def test_checkpoint_eval_does_not_require_torchvision_pretrain(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse(
            (root / "configs/VisualGenome/RA_SGG.py").read_text()
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "backbone_pretrained"
            )
        }
        self.assertIs(assignments["backbone_pretrained"], False)

    def test_relation_nms_reassigns_suppressed_pair(self) -> None:
        # The two pairs have the same entity labels and fully overlapping boxes.
        # Pair zero wins predicate 1, so pair one must fall back to predicate 2.
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]] * 2, dtype=np.float32)
        scores = np.array([
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.6],
        ], dtype=np.float32)

        result = _relation_nms_pair_scores(
            sub_boxes=boxes,
            obj_boxes=boxes,
            sub_classes=np.array([1, 1]),
            obj_classes=np.array([2, 2]),
            full_rel_scores=scores,
            nms_threshold=0.6,
            l21_threshold=0.7,
        )

        np.testing.assert_allclose(result, [[0.8, 0.0], [0.0, 0.6]])

    def test_vectorized_inclusive_iou_matches_legacy_evaluator(self) -> None:
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 4.0, 12.0, 16.0],
            [20.0, 20.0, 20.0, 20.0],
        ], dtype=np.float32)
        np.testing.assert_allclose(
            _inclusive_iou_rows(boxes, boxes),
            bbox_overlaps(boxes, boxes),
            rtol=0.0,
            atol=0.0,
        )

    def test_optimized_relation_nms_matches_upstream_bruteforce(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(12):
            object_count = 6
            xy = rng.uniform(0.0, 40.0, size=(object_count, 2))
            wh = rng.uniform(3.0, 25.0, size=(object_count, 2))
            boxes = np.column_stack((xy, xy + wh)).astype(np.float32)
            classes = rng.integers(1, 5, size=object_count, dtype=np.int64)
            pairs = np.array([
                (s, o) for s in range(object_count)
                for o in range(object_count) if s != o
            ], dtype=np.int64)
            logits = rng.normal(size=(len(pairs), 6)).astype(np.float32)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            scores = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            expected_labels, expected_scores = self._brute_relation_nms(
                boxes, classes, pairs, scores
            )
            _, actual_labels, actual_scores = _relation_nms_pair_scores(
                sub_boxes=boxes[pairs[:, 0]],
                obj_boxes=boxes[pairs[:, 1]],
                sub_classes=classes[pairs[:, 0]],
                obj_classes=classes[pairs[:, 1]],
                full_rel_scores=scores,
                object_boxes=boxes,
                pair_indices=pairs,
                return_selection=True,
            )
            np.testing.assert_array_equal(actual_labels, expected_labels)
            np.testing.assert_array_equal(actual_scores, expected_scores)

    def test_relation_nms_preserves_possible_background_assignment(self) -> None:
        # Four identical relation pairs compete for only three foreground
        # predicates. Upstream NMS assigns background to the final pair and
        # retains that row's original background probability for ranking.
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]] * 4, dtype=np.float32)
        scores = np.array([[0.1, 0.9, 0.8, 0.7]] * 4, dtype=np.float32)
        _, labels, selected_scores = _relation_nms_pair_scores(
            sub_boxes=boxes,
            obj_boxes=boxes,
            sub_classes=np.ones(4, dtype=np.int64),
            obj_classes=np.full(4, 2, dtype=np.int64),
            full_rel_scores=scores,
            return_selection=True,
        )
        np.testing.assert_array_equal(labels, [1, 2, 3, 0])
        np.testing.assert_allclose(selected_scores, [0.9, 0.8, 0.7, 0.1])


if __name__ == "__main__":
    unittest.main()
