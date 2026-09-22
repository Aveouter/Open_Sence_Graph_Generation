"""Tests for probe architectures and geometry features.

Imports torch, so this module skips itself in the dependency-free CI job.
"""

from __future__ import annotations

import unittest

from tests._optional import require_modules

require_modules("torch")

import torch

from tools.ontology_probe.probe_models import (
    GEOMETRY_DIM,
    LABEL_DIM,
    PROBE_INPUTS,
    build_probe,
    geometry_from_boxes,
    normalized_cxcywh,
)


class NormalizedCxcywhTest(unittest.TestCase):
    def test_hand_computed_conversion(self) -> None:
        # Image 200 wide x 100 tall. Subject spans x 20..60, y 10..30.
        sub = torch.tensor([[20.0, 10.0, 60.0, 30.0]])
        obj = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
        size = torch.tensor([[100.0, 200.0]])  # (height, width)

        boxes = normalized_cxcywh(sub, obj, size)
        self.assertEqual(boxes.shape, (2, 4))
        # cx = 40/200, cy = 20/100, w = 40/200, h = 20/100
        self.assertAlmostEqual(boxes[0, 0].item(), 0.20)
        self.assertAlmostEqual(boxes[0, 1].item(), 0.20)
        self.assertAlmostEqual(boxes[0, 2].item(), 0.20)
        self.assertAlmostEqual(boxes[0, 3].item(), 0.20)

    def test_height_and_width_are_not_transposed(self) -> None:
        """A transposed conversion still yields plausible geometry, so pin it."""
        sub = torch.tensor([[0.0, 0.0, 100.0, 10.0]])  # wide, short box
        obj = sub.clone()
        size = torch.tensor([[50.0, 500.0]])  # height 50, width 500
        boxes = normalized_cxcywh(sub, obj, size)
        self.assertAlmostEqual(boxes[0, 2].item(), 100.0 / 500.0)  # w uses width
        self.assertAlmostEqual(boxes[0, 3].item(), 10.0 / 50.0)    # h uses height

    def test_subjects_come_before_objects(self) -> None:
        sub = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        obj = torch.tensor([[90.0, 90.0, 100.0, 100.0]])
        size = torch.tensor([[100.0, 100.0]])
        boxes = normalized_cxcywh(sub, obj, size)
        self.assertLess(boxes[0, 0].item(), boxes[1, 0].item())


class GeometryTest(unittest.TestCase):
    def test_geometry_has_the_documented_width(self) -> None:
        sub = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 25.0]])
        obj = torch.tensor([[20.0, 20.0, 40.0, 40.0], [0.0, 0.0, 5.0, 5.0]])
        size = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
        geometry = geometry_from_boxes(sub, obj, size)
        self.assertEqual(geometry.shape, (2, GEOMETRY_DIM))
        self.assertTrue(torch.isfinite(geometry).all())

    def test_identical_boxes_give_unit_iou_and_zero_offset(self) -> None:
        box = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        size = torch.tensor([[100.0, 100.0]])
        geometry = geometry_from_boxes(box, box.clone(), size)[0]
        dx, dy, _log_wr, _log_hr, area_ratio, center_dist, iou, _union_area = geometry
        self.assertAlmostEqual(dx.item(), 0.0)
        self.assertAlmostEqual(dy.item(), 0.0)
        self.assertAlmostEqual(area_ratio.item(), 1.0, places=4)
        self.assertAlmostEqual(center_dist.item(), 0.0)
        self.assertAlmostEqual(iou.item(), 1.0, places=4)

    def test_disjoint_boxes_give_zero_iou(self) -> None:
        sub = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        obj = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
        size = torch.tensor([[100.0, 100.0]])
        geometry = geometry_from_boxes(sub, obj, size)[0]
        self.assertAlmostEqual(geometry[6].item(), 0.0)

    def test_mismatched_box_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            geometry_from_boxes(
                torch.zeros(2, 4), torch.zeros(3, 4), torch.zeros(2, 2)
            )
        with self.assertRaises(ValueError):
            geometry_from_boxes(
                torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 2)
            )

    def test_geometry_is_differentiable(self) -> None:
        sub = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        obj = torch.tensor([[20.0, 20.0, 40.0, 40.0]], requires_grad=True)
        size = torch.tensor([[100.0, 100.0]])
        geometry_from_boxes(sub, obj, size).sum().backward()
        self.assertIsNotNone(sub.grad)
        self.assertTrue(torch.isfinite(sub.grad).all())


class ProbeModelTest(unittest.TestCase):
    def _features(self, n: int, visual_dim: int = 32) -> dict[str, torch.Tensor]:
        return {
            "labels_s": torch.randint(1, 151, (n,)),
            "labels_o": torch.randint(1, 151, (n,)),
            "geometry": torch.randn(n, GEOMETRY_DIM),
            "visual_s": torch.randn(n, visual_dim),
            "visual_o": torch.randn(n, visual_dim),
            "visual_u": torch.randn(n, visual_dim),
        }

    def test_every_registered_probe_builds_and_forwards(self) -> None:
        for name in PROBE_INPUTS:
            with self.subTest(probe=name):
                model = build_probe(name, n_classes=7, visual_dim=32)
                logits = model(self._features(5))
                self.assertEqual(logits.shape, (5, 7))
                self.assertTrue(torch.isfinite(logits).all())

    def test_input_width_matches_the_declared_blocks(self) -> None:
        label_blocks = 0
        for name, inputs in PROBE_INPUTS.items():
            with self.subTest(probe=name):
                model = build_probe(name, n_classes=3, visual_dim=32)
                expected = 0
                for block in inputs:
                    expected += {
                        "label_s": LABEL_DIM,
                        "label_o": LABEL_DIM,
                        "geometry": GEOMETRY_DIM,
                        "visual_s": 32,
                        "visual_o": 32,
                        "visual_u": 32,
                    }[block]
                first_linear = next(
                    m for m in model.mlp.net if isinstance(m, torch.nn.Linear)
                )
                self.assertEqual(first_linear.in_features, expected)
                label_blocks += sum(1 for b in inputs if b.startswith("label"))
        self.assertGreater(label_blocks, 0)

    def test_b3_uses_no_labels_at_all(self) -> None:
        model = build_probe("B3", n_classes=4, visual_dim=16)
        self.assertIsNone(model.label_embed_s)
        self.assertIsNone(model.label_embed_o)
        self.assertFalse(any("label" in name for name in model.inputs))

    def test_b4_consumes_every_block(self) -> None:
        model = build_probe("B4", n_classes=4, visual_dim=16)
        self.assertEqual(
            set(model.inputs),
            {"label_s", "label_o", "geometry", "visual_s", "visual_o", "visual_u"},
        )

    def test_label_embeddings_receive_gradients(self) -> None:
        model = build_probe("B1_add", n_classes=4)
        features = self._features(6)
        logits = model(features)
        logits.sum().backward()
        grad = model.label_embed_s.weight.grad  # type: ignore[union-attr]
        self.assertIsNotNone(grad)
        # Only the labels actually present should have non-zero gradient.
        present = features["labels_s"].unique()
        self.assertGreater(grad[present].abs().sum().item(), 0.0)

    def test_out_of_range_labels_do_not_crash(self) -> None:
        """Label ids are clamped, so a stray id degrades rather than raising."""
        model = build_probe("B1_add", n_classes=3)
        features = self._features(2)
        features["labels_s"] = torch.tensor([0, 999])
        self.assertEqual(model(features).shape, (2, 3))

    def test_visual_blocks_accept_extra_leading_dims(self) -> None:
        model = build_probe("B3", n_classes=3, visual_dim=16)
        features = self._features(4, visual_dim=16)
        features["visual_s"] = features["visual_s"].unsqueeze(1)
        self.assertEqual(model(features).shape, (4, 3))

    def test_unknown_probe_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_probe("B9", n_classes=3)

    def test_b1_add_has_far_fewer_parameters_than_a_flat_lookup(self) -> None:
        """The reason B1_add is a fair compositional control, not a bigger model."""
        model = build_probe("B1_add", n_classes=50)
        flat_table = 151 * 151 * 50
        self.assertLess(model.describe()["n_parameters"], flat_table)


if __name__ == "__main__":
    unittest.main()
