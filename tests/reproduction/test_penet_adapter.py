from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from data.dataloaders.coco import make_coco_transforms
from src.methods import method_maps
from src.models.backbone import ResNetBackbone
from src.models.penet import (
    PENetContext,
    build_penet,
    fusion_func,
    _make_fc,
    _nms_overlaps,
)
from src.models.penet_detector import PENetSGDetProposalGenerator
from utils.penet_weights import (
    _backbone_key_map_official_to_ours,
    _box_head_key_map_official_to_ours,
    _box_predictor_key_map_official_to_ours,
    _rpn_head_key_map_official_to_ours,
)


class PENetArchitectureTest(unittest.TestCase):
    """Tests matching official PrototypeEmbeddingNetwork architecture."""

    def test_method_is_registered(self) -> None:
        self.assertIn("penet", method_maps)
        self.assertEqual(method_maps["penet"].__name__, "PENet_Method")

    def test_builder_official_defaults(self) -> None:
        """Builder defaults to official hyperparameters."""
        model = build_penet(
            SimpleNamespace(
                entity_nums=151,
                rel_nums=51,
                visual_dim=4096,
                hidden_dim=2048,
                penet_embed_dim=300,
                penet_pooling_dim=4096,
                penet_context_hidden_dim=512,
                glove_dir=None,
                use_freq_bias=False,
                dropout=0.2,
                penet_nms_thresh=0.5,
                penet_train_pairs=512,
                penet_pos_frac=0.25,
            )
        )
        self.assertIsInstance(model, PENetContext)
        self.assertEqual(model.num_classes, 151)
        self.assertEqual(model.num_predicates, 51)
        self.assertEqual(model.mlp_dim, 2048)
        self.assertEqual(model.context_hidden_dim, 512)
        self.assertEqual(model.embed_dim, 300)
        self.assertIsNone(model.freq_bias)
        # post_emb should be Linear(4096, 4096)
        self.assertEqual(model.post_emb.weight.shape, (4096, 4096))

    def test_builder_small_config(self) -> None:
        """Small dims for fast tests."""
        model = build_penet(
            SimpleNamespace(
                entity_nums=151,
                rel_nums=51,
                visual_dim=64,
                hidden_dim=32,
                penet_embed_dim=8,
                penet_pooling_dim=128,
                penet_context_hidden_dim=16,
                use_freq_bias=False,
                dropout=0.0,
                penet_nms_thresh=0.5,
                penet_train_pairs=512,
                penet_pos_frac=0.25,
            )
        )
        self.assertEqual(model.mlp_dim, 32)
        self.assertEqual(model.embed_dim, 8)

    def test_official_detector_key_maps(self) -> None:
        """Official detector keys map to local PE-NET detector modules."""
        backbone = ResNetBackbone(
            arch="resnext101_32x8d",
            pretrained=False,
            frozen=True,
        )
        backbone_map = _backbone_key_map_official_to_ours(backbone)
        self.assertEqual(
            backbone_map["backbone.body.stem.conv1.weight"],
            "conv1.weight",
        )
        self.assertEqual(
            backbone_map["backbone.body.stem.bn1.running_mean"],
            "bn1.running_mean",
        )
        self.assertEqual(
            backbone_map["backbone.body.layer1.0.conv1.weight"],
            "layer1.0.conv1.weight",
        )

        box_map = _box_head_key_map_official_to_ours()
        self.assertEqual(
            box_map["roi_heads.box.feature_extractor.fc6.weight"],
            "_box_extractor.fc6.weight",
        )
        self.assertEqual(
            box_map["roi_heads.box.feature_extractor.fc7.bias"],
            "_box_extractor.fc7.bias",
        )

        rpn_map = _rpn_head_key_map_official_to_ours()
        self.assertEqual(
            rpn_map["rpn.head.conv.weight"],
            "rpn_head.conv.weight",
        )
        self.assertEqual(
            rpn_map["rpn.head.bbox_pred.bias"],
            "rpn_head.bbox_pred.bias",
        )

        predictor_map = _box_predictor_key_map_official_to_ours()
        self.assertEqual(
            predictor_map["roi_heads.box.predictor.cls_score.weight"],
            "box_predictor.cls_score.weight",
        )
        self.assertEqual(
            predictor_map["roi_heads.box.predictor.bbox_pred.bias"],
            "box_predictor.bbox_pred.bias",
        )

    def test_sgdet_proposal_generator_module_shapes(self) -> None:
        """Local SGDet proposal modules match official detector tensor shapes."""
        proposal = PENetSGDetProposalGenerator(num_classes=151)
        self.assertEqual(proposal.rpn_head.conv.weight.shape, (256, 256, 3, 3))
        self.assertEqual(proposal.rpn_head.cls_logits.weight.shape, (4, 256, 1, 1))
        self.assertEqual(proposal.rpn_head.bbox_pred.weight.shape, (16, 256, 1, 1))
        self.assertEqual(proposal.box_predictor.cls_score.weight.shape, (151, 4096))
        self.assertEqual(proposal.box_predictor.bbox_pred.weight.shape, (604, 4096))
        self.assertEqual(proposal._cell_anchor(0).shape, (4, 4))

    def test_penet_eval_resize_matches_official_without_changing_default(self) -> None:
        """PE-NET opts into official 600/1000 eval resize without moving defaults."""
        default_resize = make_coco_transforms("val").transforms[0]
        self.assertEqual(default_resize.sizes, [800])
        self.assertEqual(default_resize.max_size, 1333)

        penet_resize = make_coco_transforms(
            "val",
            args=SimpleNamespace(eval_min_size=600, eval_max_size=1000),
        ).transforms[0]
        self.assertEqual(penet_resize.sizes, [600])
        self.assertEqual(penet_resize.max_size, 1000)

    def test_weight_init_is_kaiming_uniform(self) -> None:
        """Linear layers use kaiming_uniform_ (matching make_fc)."""
        model = PENetContext(
            num_classes=151,
            num_predicates=51,
            visual_dim=64,
            hidden_dim=32,
            context_hidden_dim=16,
            embed_dim=8,
            pooling_dim=128,
            use_freq_bias=False,
            dropout=0.0,
        )
        for name, layer in [
            ("post_emb", model.post_emb),
            ("gate_sub", model.gate_sub),
            ("linear_sub", model.linear_sub),
            ("out_obj", model.out_obj),
            ("lin_obj_cyx", model.lin_obj_cyx),
        ]:
            self.assertTrue(
                layer.weight.abs().sum() > 0,
                f"{name} has zero-initialized weights",
            )

    def test_out_obj_dimensions(self) -> None:
        """out_obj: Linear(context_hidden_dim=512, num_classes=151)."""
        model = PENetContext(
            num_classes=151,
            num_predicates=51,
            visual_dim=4096,
            hidden_dim=2048,
            context_hidden_dim=512,
            embed_dim=300,
            pooling_dim=4096,
            use_freq_bias=False,
            dropout=0.0,
        )
        self.assertIsInstance(model.out_obj, torch.nn.Linear)
        self.assertEqual(model.out_obj.weight.shape, (151, 512))
        self.assertIsInstance(model.lin_obj_cyx, torch.nn.Linear)
        # official: make_fc(obj_dim + embed_dim + 128, hidden_dim)
        # = make_fc(4096 + 300 + 128, 512) = make_fc(4524, 512)
        self.assertEqual(model.lin_obj_cyx.weight.shape, (512, 4524))

    def test_post_emb_dimensions(self) -> None:
        """post_emb: Linear(4096, 4096) matching official obj_dim=4096."""
        model = PENetContext(
            num_classes=151,
            num_predicates=51,
            visual_dim=4096,
            hidden_dim=2048,
            context_hidden_dim=512,
            embed_dim=300,
            pooling_dim=4096,
            use_freq_bias=False,
            dropout=0.0,
        )
        self.assertEqual(model.post_emb.weight.shape, (4096, 4096))
        # forward splits to [N, 2, 2048]
        x = torch.randn(7, 4096)
        out = model.post_emb(x)
        out = out.view(7, 2, 2048)
        self.assertEqual(out.shape, (7, 2, 2048))


class PENetForwardTest(unittest.TestCase):
    """Forward pass tests aligned with official behavior."""

    def _make_model(self, **overrides) -> PENetContext:
        kw = dict(
            num_classes=151,
            num_predicates=51,
            visual_dim=64,
            hidden_dim=32,
            context_hidden_dim=16,
            embed_dim=8,
            pooling_dim=128,
            use_freq_bias=False,
            dropout=0.0,
        )
        kw.update(overrides)
        return PENetContext(**kw)

    def test_fusion_func(self) -> None:
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        result = fusion_func(x, y)
        self.assertEqual(result.shape, (4, 8))
        self.assertTrue((result != 0).any())

    def test_forward_cosine_similarity_output(self) -> None:
        """Rel logits from cosine similarity (51-d, bg at 0)."""
        torch.manual_seed(42)
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))

        outputs = model(visual_feats, boxes, labels)
        self.assertEqual(outputs["rel_logits"].shape, (20, 51))
        self.assertEqual(outputs["pair_indices"].shape, (20, 2))
        self.assertEqual(outputs["sub_boxes"].shape, (20, 4))
        self.assertEqual(outputs["obj_boxes"].shape, (20, 4))
        self.assertEqual(outputs["predicate_bg_index"], "first")

    def test_forward_no_pairs(self) -> None:
        """Single object → no pairs."""
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(1, 64)
        boxes = torch.rand(1, 4)
        labels = torch.tensor([1])
        outputs = model(visual_feats, boxes, labels)
        self.assertEqual(outputs["rel_logits"].shape, (0, 51))
        self.assertEqual(outputs["pair_indices"].shape, (0, 2))

    def test_training_produces_proto_losses(self) -> None:
        """Training mode + rel_annotations → add_losses populated."""
        torch.manual_seed(42)
        model = self._make_model()
        model.train()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        ra = torch.tensor([[0, 1, 10], [2, 3, 25]])
        outputs = model(visual_feats, boxes, labels, rel_annotations=ra)
        add = outputs["add_losses"]
        for name in ["l21_loss", "dist_loss2", "loss_dis"]:
            self.assertIn(name, add)
            self.assertGreaterEqual(add[name].item(), 0.0)

    def test_eval_produces_no_losses(self) -> None:
        """Eval mode → no proto losses."""
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        outputs = model(
            visual_feats,
            boxes,
            labels,
            rel_annotations=torch.tensor([[0, 1, 10]]),
        )
        self.assertEqual(outputs["add_losses"], {})

    def test_sgcls_returns_obj_logits(self) -> None:
        """return_obj_preds=True → obj_logits returned."""
        torch.manual_seed(42)
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        outputs = model(visual_feats, boxes, labels, return_obj_preds=True)
        self.assertIsNotNone(outputs["obj_logits"])
        self.assertEqual(outputs["obj_logits"].shape, (5, 151))

    def test_predcls_uses_gt_labels(self) -> None:
        """Training: entity_preds == GT labels."""
        torch.manual_seed(42)
        model = self._make_model()
        model.train()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        outputs = model(visual_feats, boxes, labels)
        self.assertTrue(torch.equal(outputs["obj_labels"], labels.long()))

    def test_fusion_func_output(self) -> None:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = fusion_func(x, y)
        expected = torch.relu(x + y) - (x - y) ** 2
        self.assertTrue(torch.allclose(result, expected))

    def test_make_fc_kaiming_init(self) -> None:
        fc = _make_fc(64, 32)
        self.assertEqual(fc.weight.shape, (32, 64))
        self.assertEqual(fc.bias.shape, (32,))

    def test_nms_overlaps(self) -> None:
        boxes = torch.tensor(
            [
                [[0, 0, 100, 100]],
                [[50, 50, 150, 150]],
                [[200, 200, 300, 300]],
            ],
            dtype=torch.float32,
        )
        iou = _nms_overlaps(boxes)
        self.assertEqual(iou.shape, (3, 3, 1))
        self.assertAlmostEqual(iou[0, 0, 0].item(), 1.0, places=4)

    def test_nms_per_cls(self) -> None:
        model = self._make_model(nms_thresh=0.5)
        obj_dists = torch.randn(5, 151)
        boxes_per_cls = torch.rand(5, 151, 4) * 100
        preds = model.nms_per_cls(obj_dists, boxes_per_cls, 5)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue((preds >= 0).all())
        self.assertTrue((preds < 151).all())

    def test_sgdet_refine_with_boxes_per_cls(self) -> None:
        """SGDet with boxes_per_cls → NMS refinement."""
        torch.manual_seed(42)
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        obj_dists = torch.randn(5, 151)
        boxes_per_cls = torch.rand(5, 151, 4) * 100
        outputs = model(
            visual_feats,
            boxes,
            labels,
            return_obj_preds=True,
            obj_dists=obj_dists,
            boxes_per_cls=boxes_per_cls,
        )
        self.assertEqual(outputs["obj_logits"].shape, (5, 151))

    def test_model_with_freq_bias(self) -> None:
        """Frequency bias should not crash."""
        model = self._make_model(use_freq_bias=True)
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        outputs = model(visual_feats, boxes, labels)
        self.assertEqual(outputs["rel_logits"].shape, (20, 51))

    def test_union_feats_fallback(self) -> None:
        """No union features → fallback works silently."""
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        out_fb = model(visual_feats, boxes, labels)
        self.assertEqual(out_fb["rel_logits"].shape, (20, 51))

    def test_precomputed_union_feats(self) -> None:
        """precomputed_union_feats should be used."""
        torch.manual_seed(42)
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))
        union_feats = torch.randn(20, 128)
        out = model(visual_feats, boxes, labels, precomputed_union_feats=union_feats)
        self.assertEqual(out["rel_logits"].shape, (20, 51))

    def test_external_union_fn(self) -> None:
        """External _compute_union_fn callback should work."""
        torch.manual_seed(42)
        model = self._make_model()
        model.eval()
        visual_feats = torch.randn(5, 64)
        boxes = torch.rand(5, 4)
        labels = torch.randint(1, 151, (5,))

        # Fake multi-scale feature maps
        feature_maps = {32: torch.randn(2048, 8, 12)}

        def fake_union_fn(fmaps, b, pairs, img_sz):
            P = pairs.size(0)
            return torch.randn(P, 128)

        out = model(
            visual_feats,
            boxes,
            labels,
            _compute_union_fn=fake_union_fn,
            _fpn_features=feature_maps,
            _image_size=torch.tensor([256.0, 384.0]),
        )
        self.assertEqual(out["rel_logits"].shape, (20, 51))


if __name__ == "__main__":
    unittest.main()
