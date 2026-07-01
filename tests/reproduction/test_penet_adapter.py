from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.methods import method_maps
from src.models.penet import PENetContext, build_penet


class PENetAdapterTest(unittest.TestCase):
    def test_method_is_registered(self) -> None:
        self.assertIn("penet", method_maps)
        self.assertEqual(method_maps["penet"].__name__, "PENet_Method")

    def test_builder_uses_opensgg_config_fields(self) -> None:
        model = build_penet(
            SimpleNamespace(
                entity_nums=151,
                rel_nums=51,
                visual_dim=16,
                hidden_dim=32,
                penet_embed_dim=8,
                use_freq_bias=False,
                freq_bias_eps=1e-12,
                penet_textual_only=False,
                penet_visual_only=False,
                dropout=0.0,
            )
        )

        self.assertIsInstance(model, PENetContext)
        self.assertEqual(model.num_classes, 151)
        self.assertEqual(model.num_predicates, 51)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.embed_dim, 8)

    def test_forward_returns_motifs_compatible_schema(self) -> None:
        torch.manual_seed(11)
        model = PENetContext(
            num_classes=151,
            num_predicates=51,
            visual_dim=16,
            hidden_dim=32,
            embed_dim=8,
            use_freq_bias=False,
            dropout=0.0,
        )
        visual_feats = torch.randn(3, 16)
        boxes = torch.rand(3, 4)
        labels = torch.tensor([1, 2, 3])

        outputs = model(visual_feats, boxes, labels)

        self.assertEqual(outputs["rel_logits"].shape, (6, 51))
        self.assertEqual(outputs["pair_indices"].shape, (6, 2))
        self.assertEqual(outputs["obj_labels"].shape, (3,))
        self.assertEqual(outputs["sub_boxes"].shape, (6, 4))
        self.assertEqual(outputs["obj_boxes"].shape, (6, 4))

    def test_forward_handles_no_relation_pairs(self) -> None:
        model = PENetContext(
            num_classes=151,
            num_predicates=51,
            visual_dim=16,
            hidden_dim=32,
            embed_dim=8,
            use_freq_bias=False,
            dropout=0.0,
        )
        visual_feats = torch.randn(1, 16)
        boxes = torch.rand(1, 4)
        labels = torch.tensor([1])

        outputs = model(visual_feats, boxes, labels)

        self.assertEqual(outputs["rel_logits"].shape, (0, 51))
        self.assertEqual(outputs["pair_indices"].shape, (0, 2))
        self.assertEqual(outputs["sub_boxes"].shape, (0, 4))
        self.assertEqual(outputs["obj_boxes"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
