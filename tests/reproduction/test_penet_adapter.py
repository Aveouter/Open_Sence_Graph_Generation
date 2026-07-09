from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch

from src.core.metrics import metric
from src.methods import method_maps
from src.models.backbone import PENetBoxFeatureExtractor
from src.models.penet_detector import PENetSGDetProposalGenerator
from src.models.penet import (
    PENetContext,
    build_penet,
    fusion_func,
    _filter_sgdet_overlap_pairs,
    _make_fc,
    _nms_overlaps,
)
from utils.penet_weights import (
    load_detector_box_feature_extractor_from_state_dict,
    load_relation_box_feature_extractor_from_state_dict,
)


OFFICIAL_PENET_SGDET_CKPT = (
    "/workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/"
    "PE-NET_SGDet/model_final.pth"
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
                penet_sgdet_eval_topk=100,
                penet_sgdet_require_overlap=True,
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
                penet_sgdet_eval_topk=100,
                penet_sgdet_require_overlap=True,
            )
        )
        self.assertEqual(model.mlp_dim, 32)
        self.assertEqual(model.embed_dim, 8)

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

    def test_sgdet_overlap_pair_filter_matches_official_test_pairs(self) -> None:
        pairs = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        boxes = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.25, 0.25, 0.20, 0.20],
                [0.80, 0.80, 0.10, 0.10],
            ],
            dtype=torch.float32,
        )
        filtered = _filter_sgdet_overlap_pairs(
            pairs,
            boxes,
            torch.tensor([100.0, 100.0]),
        )
        self.assertTrue(torch.equal(filtered, torch.tensor([[0, 1], [1, 0]])))

    def test_sgdet_overlap_pair_filter_keeps_official_placeholder(self) -> None:
        pairs = torch.tensor([[0, 1], [1, 0]])
        boxes = torch.tensor(
            [
                [0.10, 0.10, 0.10, 0.10],
                [0.90, 0.90, 0.10, 0.10],
            ],
            dtype=torch.float32,
        )
        filtered = _filter_sgdet_overlap_pairs(
            pairs,
            boxes,
            torch.tensor([100.0, 100.0]),
        )
        self.assertTrue(torch.equal(filtered, torch.tensor([[0, 0]])))

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

    def test_sgdet_outputs_are_top100_sorted_cache(self) -> None:
        """SGDet cache is capped for official R@100/mR@100 evaluation."""
        torch.manual_seed(42)
        model = self._make_model(sgdet_eval_topk=100)
        model.eval()
        num_boxes = 12
        visual_feats = torch.randn(num_boxes, 64)
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] = boxes[:, 2:].clamp_min(0.05)
        labels = torch.randint(1, 151, (num_boxes,))
        obj_dists = torch.randn(num_boxes, 151)
        base_xyxy = torch.rand(num_boxes, 4)
        xy1 = torch.minimum(base_xyxy[:, :2], base_xyxy[:, 2:]) * 100
        xy2 = torch.maximum(base_xyxy[:, :2], base_xyxy[:, 2:]) * 100 + 1
        boxes_xyxy = torch.cat([xy1, xy2], dim=1)
        boxes_per_cls = boxes_xyxy[:, None, :].expand(num_boxes, 151, 4).clone()

        outputs = model(
            visual_feats,
            boxes,
            labels,
            return_obj_preds=True,
            obj_dists=obj_dists,
            boxes_per_cls=boxes_per_cls,
        )
        self.assertIn("sgdet_rel_scores", outputs)
        self.assertLessEqual(outputs["sgdet_rel_scores"].shape[0], 100)
        self.assertEqual(outputs["sgdet_rel_scores"].shape[-1], 51)
        self.assertEqual(outputs["sgdet_box_space"], "resized_xyxy")

    def test_sgdet_proposal_generator_module_shapes(self) -> None:
        proposal = PENetSGDetProposalGenerator(num_classes=151)
        self.assertEqual(proposal.rpn_head.conv.weight.shape, (256, 256, 3, 3))
        self.assertEqual(proposal.rpn_head.cls_logits.weight.shape, (4, 256, 1, 1))
        self.assertEqual(proposal.rpn_head.bbox_pred.weight.shape, (16, 256, 1, 1))
        self.assertEqual(proposal.box_predictor.cls_score.weight.shape, (151, 4096))
        self.assertEqual(proposal.box_predictor.bbox_pred.weight.shape, (604, 4096))

    @unittest.skipUnless(
        os.path.isfile(OFFICIAL_PENET_SGDET_CKPT),
        "official PE-NET SGDet checkpoint is not available locally",
    )
    def test_official_sgdet_relation_and_detector_box_tensor_parity(self) -> None:
        ckpt = torch.load(OFFICIAL_PENET_SGDET_CKPT, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        stripped = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        relation_extractor = PENetBoxFeatureExtractor()
        detector_extractor = PENetBoxFeatureExtractor()
        self.assertEqual(
            load_relation_box_feature_extractor_from_state_dict(
                relation_extractor,
                state_dict,
            ),
            4,
        )
        self.assertEqual(
            load_detector_box_feature_extractor_from_state_dict(
                detector_extractor,
                state_dict,
            ),
            4,
        )

        rel_state = relation_extractor.state_dict()
        det_state = detector_extractor.state_dict()
        for suffix in ("fc6.weight", "fc6.bias", "fc7.weight", "fc7.bias"):
            self.assertTrue(
                torch.equal(
                    rel_state[suffix],
                    stripped[f"roi_heads.relation.box_feature_extractor.{suffix}"],
                ),
                suffix,
            )
            self.assertTrue(
                torch.equal(
                    det_state[suffix],
                    stripped[f"roi_heads.box.feature_extractor.{suffix}"],
                ),
                suffix,
            )
        self.assertFalse(torch.equal(rel_state["fc7.weight"], det_state["fc7.weight"]))

    @unittest.skipUnless(
        os.path.isfile(OFFICIAL_PENET_SGDET_CKPT),
        "official PE-NET SGDet checkpoint is not available locally",
    )
    def test_official_sgdet_relation_predictor_tensor_parity(self) -> None:
        ckpt = torch.load(OFFICIAL_PENET_SGDET_CKPT, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"]
        stripped = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        model = PENetContext()
        remapped = model.remap_external_state_dict(state_dict)
        self.assertEqual(len(remapped), 63)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        self.assertFalse(unexpected)
        self.assertGreater(len(missing), 0)

        local_state = model.state_dict()
        for key in (
            "post_emb.weight",
            "vis2sem.0.weight",
            "out_obj.weight",
            "lin_obj_cyx.weight",
        ):
            official_key = f"roi_heads.relation.predictor.{key}"
            self.assertTrue(torch.equal(local_state[key], stripped[official_key]), key)

    def test_penet_sgdet_metric_accepts_50_or_51_rel_nums(self) -> None:
        """PE-NET SGDet keeps bg at column 0 even if epoch-end passes 50."""
        rel_scores = torch.zeros(1, 51)
        rel_scores[0, 20] = 1.0
        outputs = {
            "model_family": "penet_sgdet",
            "sgdet_rel_scores": [rel_scores],
            "sgdet_sub_boxes": [torch.tensor([[10.0, 10.0, 30.0, 30.0]])],
            "sgdet_obj_boxes": [torch.tensor([[50.0, 50.0, 80.0, 80.0]])],
            "sgdet_sub_scores": [torch.tensor([0.9])],
            "sgdet_obj_scores": [torch.tensor([0.8])],
            "sgdet_sub_classes": [torch.tensor([1])],
            "sgdet_obj_classes": [torch.tensor([2])],
        }
        target = {
            "boxes": torch.tensor(
                [
                    [0.20, 0.20, 0.20, 0.20],
                    [0.65, 0.65, 0.30, 0.30],
                ]
            ),
            "labels": torch.tensor([1, 2]),
            "rel_annotations": torch.tensor([[0, 1, 20]]),
            "orig_size": torch.tensor([100, 100]),
            "size": torch.tensor([100, 100]),
        }
        for rel_nums in (50, 51):
            result, _ = metric(
                outputs,
                [target],
                ["sgdet_R@100", "sgdet_mR@100"],
                rel_nums=rel_nums,
                entity_nums=151,
            )
            self.assertEqual(result["sgdet_R@100"], 1.0)
            self.assertGreater(result["sgdet_mR@100"], 0.0)

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
