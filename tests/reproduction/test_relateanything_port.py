"""Public behavior of the RelateAnything source port, not baseline results."""
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests._optional import skip_unless


class RelateAnythingPortTest(unittest.TestCase):
    def test_checkpoint_configuration_preserves_architecture_and_offline_loading(self):
        from src.modules.relateanything.config import config_from_args

        config = config_from_args(SimpleNamespace(
            d_model=512, text_dim=512, final_budget=500,
            backbone_model="local-backbone", dropout=None,
        ), backbone_pretrained=False)
        self.assertEqual(config.final_budget, 500)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.backbone_model, "local-backbone")
        self.assertFalse(config.backbone_pretrained)
        self.assertEqual(config.dropout, 0.2)

    @skip_unless("numpy")
    def test_pair_logits_contribute_before_the_shared_calibrated_sigmoid(self):
        import numpy as np

        from src.modules.relateanything.scoring import ScoreContract, graph_constrained

        contract = ScoreContract(calib_a=0.5, calib_b=-1.0, pair_weight=1.0)
        scores = contract.scores(np.array([[2.0, 0.0], [0.0, 4.0]]), np.array([2.0, -2.0]))
        np.testing.assert_allclose(scores, 1 / (1 + np.exp(-np.array([[1.0, 0.0], [-2.0, 0.0]]))))
        np.testing.assert_array_equal(graph_constrained(scores, np.array([True, False])),
                                      [[True, False], [False, False]])

    @skip_unless("numpy", "torch")
    def test_a1_evaluator_uses_image_recall_and_observed_class_mean(self):
        import torch

        from src.modules.relateanything.eval.evaluator import SGClsEvaluator

        evaluator = SGClsEvaluator(topk=[1], score_mode="sigmoid", graph_constraint=True)
        evaluator.update({
            "logits": torch.tensor([[[5.0, -5.0]], [[5.0, -5.0]]]),
            "pair_logits": torch.zeros(2, 1),
            "sub_idx": torch.tensor([[0], [0]]), "obj_idx": torch.tensor([[1], [1]]),
            "valid_mask": torch.ones(2, 1, dtype=torch.bool),
        }, [{"relations": torch.tensor([[0, 1, 0]])},
            {"relations": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 2, 1]])}])
        metrics = evaluator.compute()
        self.assertAlmostEqual(metrics["R@1"], 2 / 3)
        self.assertAlmostEqual(metrics["mR@1"], 0.5)
        self.assertAlmostEqual(metrics["F1@1"], 4 / 7)

    def test_complete_model_predicts_directed_pairs_without_object_labels(self):
        try:
            if tuple(map(int, importlib.metadata.version("transformers").split(".")[:2])) < (5, 14):
                self.skipTest("complete-model tests require the isolated RelateAnything runtime")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("complete-model tests require transformers >=5.14")
        import torch

        from src.modules.relateanything import RelSGG, RelSGGConfig

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({
                "model_type": "dinov3_vit", "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 2, "patch_size": 16,
                "num_register_tokens": 1, "image_size": 64,
            }))
            model = RelSGG(RelSGGConfig(
                backbone_model=directory, backbone_pretrained=False,
                d_model=32, text_dim=16, geo_budget=6, final_budget=6,
                n_self_layers=1, n_cross_layers=1, n_dep_layers=1, n_gnd_layers=1,
                n_heads=2, deformable_points=2, deformable_heads=2,
                deformable_nulls=1, dropout=0.0,
            )).eval()
            model.vocab_head.set_vocabulary_matrix(["on", "under"], torch.randn(2, 16))
            model.reparameterize()
            with torch.no_grad():
                output = model(torch.rand(1, 3, 64, 64),
                               torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]]),
                               box_counts=torch.tensor([2]), targets=None)
            valid = output["valid_mask"]
            self.assertTrue(valid.any())
            self.assertEqual(output["logits"].shape[-1], 2)
            self.assertTrue(torch.isfinite(output["logits"][valid]).all())
            self.assertTrue((output["sub_idx"][valid] != output["obj_idx"][valid]).all())
            self.assertNotIn("loss", output)

    def test_public_adapter_rejects_missing_checkpoint_without_random_fallback(self):
        from src.relateanything import RelateAnythingModel

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "model.pth"):
                RelateAnythingModel.from_checkpoint(Path(directory, "model.pth"), device="cpu")

    @skip_unless("torch")
    def test_ci_smoke_respects_the_checkpoint_only_region_runtime(self):
        result = subprocess.run([
            sys.executable, "tools/ci_smoke_test.py", "--methods", "relateanything",
        ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("pipeline_smoke_only", result.stdout)
        self.assertIn("checkpoint-backed parity requires external assets", result.stdout)

    def test_registered_region_method_can_be_resolved_without_legacy_dependencies(self):
        result = subprocess.run([
            sys.executable, "-S", "-c",
            "from src.methods import method_maps; "
            "assert method_maps['relateanything'].__name__ == 'RelateAnything_Method'; "
            "assert 'penet' in method_maps; assert len(method_maps) >= 18",
        ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_main_cli_selects_relateanything_and_requires_a_checkpoint(self):
        result = subprocess.run([sys.executable, "train.py", "--method", "RelateAnything", "--test"],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("RelateAnything requires --ckpt_path", result.stderr)

    def test_cli_validates_pixel_region_input_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            regions = Path(directory, "regions.json")
            regions.write_text(json.dumps({"boxes": [[0, 0, 10, 10]], "labels": [1]}))
            result = subprocess.run([
                sys.executable, "train.py", "--method", "RelateAnything", "--device", "cpu",
                "--ckpt_path", str(Path(directory, "model.pth")), "--image", "image.jpg",
                "--regions", str(regions),
            ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("regions must be an array of pixel xyxy boxes", result.stderr)

    def test_release_checkpoint_predicts_the_same_triplets_as_official_api(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        official_root = os.environ.get("RELATEANYTHING_REFERENCE_ROOT")
        if not checkpoint or not official_root:
            self.skipTest("requires externally stored official source and release snapshot")
        import numpy as np
        from PIL import Image

        from src.relateanything import RelateAnythingModel

        sys.path.insert(0, official_root)
        self.addCleanup(sys.path.remove, official_root)
        from relsgg import RelateAnything as OfficialAPI

        predicates = ["on", "under", "holding"]
        reference = OfficialAPI.from_checkpoint(checkpoint, predicates=predicates, device="cpu", strict=True)
        local = RelateAnythingModel.from_checkpoint(checkpoint, predicates=predicates, device="cpu")
        image = Image.fromarray(np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3))
        boxes = [[2, 2, 30, 30], [34, 16, 75, 60], [20, 4, 56, 44]]
        expected = reference.predict(image, boxes, topk=6, max_boxes=100)
        actual = local.predict(image, boxes, topk=6, max_boxes=100)
        self.assertEqual([(t.subject_idx, t.object_idx, t.predicate) for t in actual],
                         [(t.subject_idx, t.object_idx, t.predicate) for t in expected])
        np.testing.assert_allclose([t.score for t in actual], [t.score for t in expected], rtol=1e-6, atol=1e-7)
        self.assertTrue(actual, "comparison must not pass vacuously on empty predictions")
        masks = np.zeros((3, 64, 80), dtype=np.uint8)
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            masks[index, y1:y2, x1:x2] = 1
        for decompose in (False, True):
            actual = local.predict(image, boxes, masks=masks, box_scores=[0.8, 0.9, 0.7],
                                   topk=6, decompose=decompose)
            expected = reference.predict(image, boxes, masks=masks, box_scores=[0.8, 0.9, 0.7],
                                         topk=6, decompose=decompose)
            streams = actual if decompose else {"all": actual}
            expected_streams = expected if decompose else {"all": expected}
            for stream, triplets in streams.items():
                reference_triplets = expected_streams[stream]
                self.assertEqual([(t.subject_idx, t.object_idx, t.predicate) for t in triplets],
                                 [(t.subject_idx, t.object_idx, t.predicate) for t in reference_triplets])
                np.testing.assert_allclose([t.score for t in triplets], [t.score for t in reference_triplets],
                                           rtol=1e-6, atol=1e-7)
        local.set_vocabulary(["holding", "on"])
        reference.set_vocabulary(["holding", "on"])
        actual = local.predict(image, boxes, topk=6, max_boxes=100)
        expected = reference.predict(image, boxes, topk=6, max_boxes=100)
        self.assertEqual([(t.subject_idx, t.object_idx, t.predicate) for t in actual],
                         [(t.subject_idx, t.object_idx, t.predicate) for t in expected])
        np.testing.assert_allclose([t.score for t in actual], [t.score for t in expected], rtol=1e-6, atol=1e-7)

    def test_release_full_vocabulary_preserves_the_sidecar_order(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        if not checkpoint:
            self.skipTest("requires externally stored release snapshot")
        import numpy as np
        import torch
        from PIL import Image

        from src.modules.relateanything.eval.evaluator import (
            SoftSGClsEvaluator,
            build_cross_match_matrix,
        )
        from src.relateanything import RelateAnythingModel
        from src.relateanything_evaluation import (
            encode_benchmark_predicates,
            evaluate_batches,
            load_benchmark_bank,
        )

        with np.load(Path(checkpoint).parent / "predicate_embeddings.npz", allow_pickle=True) as bank:
            names = bank["names"].tolist()
        model = RelateAnythingModel.from_checkpoint(checkpoint, full_vocabulary=True)
        self.assertEqual(model.predictor.predicates, names)
        self.assertEqual(len(names), 19103)
        embeddings = load_benchmark_bank(checkpoint, model.predictor.predicates)
        self.assertEqual(embeddings.shape[0], len(names))
        self.assertTrue(np.isfinite(embeddings).all())
        predictions = model.predict(Image.new("RGB", (80, 64), "gray"),
                                    [[2, 2, 30, 30], [34, 16, 75, 60]], topk=2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(item.predicate in names for item in predictions))
        target_embeddings = encode_benchmark_predicates(["on", "under"], model.predictor.text_student)
        matrix = build_cross_match_matrix(names, ["on", "under"], embeddings,
                                          target_embeddings.numpy(), tau_eval=0.72)
        self.assertTrue(matrix.any(dim=0).all())
        images = torch.full((1, 3, 448, 448), 128 / 255)
        boxes = torch.tensor([[[0.25, 0.25, 0.3, 0.3], [0.7, 0.7, 0.3, 0.3]]])
        counts = torch.tensor([2])
        targets = [{"relations": torch.tensor([[0, 1, 0], [1, 0, 1]])}]
        with torch.inference_mode():
            outputs = model.forward_regions(images, boxes, counts)
        reference = SoftSGClsEvaluator(match_matrix=matrix, group_of=torch.arange(2),
                                        topk=[20, 50, 100], graph_constraint=True, score_mode="sigmoid")
        reference.update(outputs, targets)
        report = evaluate_batches(model, [(images, boxes, counts, targets)], num_predicates=2,
                                  protocol="A3", match_matrix=matrix)
        self.assertEqual(report["metrics"], {f"regions_A3_{key}": value
                                           for key, value in reference.compute().items()})

    def test_a3_target_text_space_uses_official_training_templates(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        if not checkpoint:
            self.skipTest("requires externally stored release snapshot")
        import torch

        from src.modules.relateanything.text.student import encode_texts_student
        from src.modules.relateanything.vocabulary import TRAIN_TEMPLATES
        from src.relateanything_evaluation import encode_benchmark_predicates

        student = str(Path(checkpoint).parent / "text_student.pt")
        expected = encode_texts_student(["on", "under"], student,
                                        templates=TRAIN_TEMPLATES, device="cpu")
        actual = encode_benchmark_predicates(["on", "under"], student, device="cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_main_cli_writes_checkpoint_backed_predictions(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        if not checkpoint:
            self.skipTest("requires externally stored release snapshot")
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (80, 64), "gray").save(root / "image.png")
            (root / "regions.json").write_text(json.dumps([[2, 2, 30, 30], [34, 16, 75, 60]]))
            (root / "predicates.json").write_text(json.dumps(["on", "under"]))
            result = subprocess.run([
                sys.executable, "train.py", "--method", "RelateAnything", "--test",
                "--ckpt_path", checkpoint, "--device", "cpu", "--image", str(root / "image.png"),
                "--regions", str(root / "regions.json"), "--predicate_vocabulary", str(root / "predicates.json"),
                "--relation_output", str(root / "prediction.json"), "--relation_topk", "2",
            ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "prediction.json").read_text())
            self.assertFalse(report["reproduced"])
            self.assertEqual(report["task"], "supplied_regions")
            self.assertEqual(report["predicates"], ["on", "under"])
            self.assertEqual(len(report["predictions"]), 2)
            self.assertTrue(all(item["subject_idx"] != item["object_idx"] for item in report["predictions"]))

    def test_released_model_region_loop_matches_official_metrics_and_restores_budget(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        if not checkpoint:
            self.skipTest("requires externally stored release snapshot")
        import torch

        from src.modules.relateanything.eval.evaluator import SGClsEvaluator
        from src.relateanything import RelateAnythingModel
        from src.relateanything_evaluation import evaluate_batches

        model = RelateAnythingModel.from_checkpoint(checkpoint, predicates=["on", "under"],
                                                    calibration=False)
        images = torch.arange(3 * 448 * 448, dtype=torch.float32).reshape(1, 3, 448, 448) % 256 / 255
        boxes = torch.tensor([[[0.25, 0.25, 0.3, 0.3], [0.7, 0.7, 0.3, 0.3]]])
        counts = torch.tensor([2])
        targets = [{"relations": torch.tensor([[0, 1, 0], [1, 0, 1]])}]
        budget = model.predictor.model.sampler.final_budget
        with torch.inference_mode():
            outputs = model.forward_regions(images, boxes, counts)
        official = SGClsEvaluator(topk=[20, 50, 100], num_predicates=2,
                                  graph_constraint=True, score_mode="sigmoid")
        official.update(outputs, targets)
        report = evaluate_batches(model, [(images, boxes, counts, targets)], num_predicates=2)
        self.assertEqual(report["seen_images"], 1)
        self.assertEqual(report["scored_relations"], 2)
        self.assertEqual(report["metrics"], {f"regions_A1_{key}": value
                                           for key, value in official.compute().items()})
        self.assertEqual(model.predictor.model.sampler.final_budget, budget)
        with self.assertRaisesRegex(ValueError, "one region target"):
            evaluate_batches(model, [(images, boxes, counts, [])], num_predicates=2, eval_budget=1)
        self.assertEqual(model.predictor.model.sampler.final_budget, budget)

    def test_official_pack_loader_feeds_the_released_model_without_label_inputs(self):
        checkpoint = os.environ.get("RELATEANYTHING_TEST_CHECKPOINT")
        if not checkpoint:
            self.skipTest("requires externally stored release snapshot")
        import numpy as np
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader

        from src.modules.relateanything.data import RelationDataset, collate_fn
        from src.relateanything import RelateAnythingModel
        from src.relateanything_evaluation import evaluate_batches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "test"
            split.mkdir()
            Image.new("RGB", (80, 64), "gray").save(root / "image.png")
            (split / "meta.json").write_text(json.dumps({
                "img_dir": str(root), "categories": ["ignored_object_label"],
                "predicates": ["on", "under"],
            }))
            (split / "file_names.json").write_text(json.dumps(["image.png"]))
            np.save(split / "img_meta.npy", np.array([[0, 80, 64, 0, 2, 0, 2]], dtype=np.int64))
            np.save(split / "boxes.npy", np.array([[0.25, 0.25, 0.3, 0.3],
                                                   [0.7, 0.7, 0.3, 0.3]], dtype=np.float32))
            np.save(split / "box_cats.npy", np.array([0, 0], dtype=np.int64))
            np.save(split / "rels.npy", np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 1]], dtype=np.int64))
            dataset = RelationDataset(root, "test", resolution=448, max_objects=100)
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
            images, _, counts, targets = next(iter(loader))
            self.assertEqual(images.shape, (1, 3, 448, 448))
            torch.testing.assert_close(counts, torch.tensor([2]))
            self.assertIn("entity_labels", targets[0])
            model = RelateAnythingModel.from_checkpoint(checkpoint,
                                                        predicates=dataset.predicate_names, calibration=False)
            report = evaluate_batches(model, loader, num_predicates=2)
            self.assertEqual(report["seen_images"], 1)
            self.assertEqual(report["scored_relations"], 2)
            self.assertEqual(report["empty_predictions"], 0)
            del loader, dataset
            gc.collect()  # Release the official loader's memmaps before Windows cleanup.

    @skip_unless("numpy", "torch")
    def test_region_evaluation_keeps_its_namespace_and_rejects_lost_denominators(self):
        import torch

        from src.relateanything_evaluation import RegionEvaluator

        evaluator = RegionEvaluator(num_predicates=2, topk=[1])
        evaluator.update({"logits": torch.tensor([[[5.0, -5.0]]]),
                          "pair_logits": torch.zeros(1, 1), "sub_idx": torch.tensor([[0]]),
                          "obj_idx": torch.tensor([[1]]), "valid_mask": torch.tensor([[False]])},
                         [{"relations": torch.tensor([[0, 1, 0]])}])
        with self.assertRaisesRegex(RuntimeError, "denominator"):
            evaluator.compute()

    @skip_unless("numpy", "torch")
    def test_a3_region_evaluator_preserves_synonym_matching_and_metric_identity(self):
        import torch

        from src.relateanything_evaluation import RegionEvaluator

        evaluator = RegionEvaluator(num_predicates=2, protocol="A3", topk=[1],
                                    match_matrix=torch.tensor([[True], [True]]))
        evaluator.update({"logits": torch.tensor([[[-5.0, 5.0]]]),
                          "pair_logits": torch.zeros(1, 1), "sub_idx": torch.tensor([[0]]),
                          "obj_idx": torch.tensor([[1]]), "valid_mask": torch.tensor([[True]])},
                         [{"relations": torch.tensor([[0, 1, 0]])}])
        self.assertEqual(evaluator.compute()["regions_A3_SoftF1@1"], 1.0)

    def test_evaluation_cli_requires_full_input_evidence_before_loading_weights(self):
        result = subprocess.run([
            sys.executable, "train.py", "--method", "RelateAnything", "--test",
            "--ckpt_path", "unavailable/model.pth", "--data_root", "unavailable/vg150",
        ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("evaluation requires --relation_evidence", result.stderr)

    def test_evaluation_rejects_unaligned_benchmark_settings_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "config.py")
            config.write_text("method='RelateAnything'\ntask='supplied_regions'\nsplit='test'\n"
                              "limit=0\ngraph_constraint=True\nimg_size=224\nmax_objects=100\n"
                              "eval_budget=500\nweights='ema'\n")
            result = subprocess.run([
                sys.executable, "train.py", "--method", "RelateAnything", "--test",
                "--config_file", str(config), "--ckpt_path", "unavailable/model.pth",
                "--relation_evidence", "unavailable/evidence.json",
                "--relation_official_root", "unavailable/source",
            ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("official benchmark settings", result.stderr)

    def test_evaluation_rejects_a_checkpoint_outside_the_audited_release_name(self):
        result = subprocess.run([
            sys.executable, "train.py", "--method", "RelateAnything", "--test",
            "--ckpt_path", "unavailable/other.pth", "--data_root", "unavailable/vg150",
            "--relation_evidence", "unavailable/evidence.json",
            "--relation_official_root", "unavailable/source",
        ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("audited model.pth", result.stderr)

    def test_ported_source_matches_the_apache_git_revision(self):
        official_root = os.environ.get("RELATEANYTHING_REFERENCE_ROOT")
        if not official_root:
            self.skipTest("requires externally stored official source")
        root = Path(__file__).resolve().parents[2] / "src/modules/relateanything"
        manifest = json.loads((root / "UPSTREAM.json").read_text())
        for name in manifest["files"]:
            source = name if name in ("LICENSE", "THIRD_PARTY_NOTICES.md") else "relsgg/" + name
            expected = subprocess.check_output([
                "git", "-C", official_root, "show", manifest["source_commit"] + ":" + source,
            ]).replace(b"\r\n", b"\n")
            actual = (root / name).read_bytes().replace(b"\r\n", b"\n")
            if name == "eval/evaluator.py":
                actual = actual.split(b"\n", 1)[1].replace(
                    b"from ..scoring import ScoreContract", b"from relsgg.scoring import ScoreContract")
            self.assertEqual(actual, expected, name)

    def test_source_identity_detects_a_changed_model_before_checkpoint_loading(self):
        from src.modules.relateanything.provenance import verify_source

        root = Path(__file__).resolve().parents[2] / "src/modules/relateanything"
        verify_source(root)
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory, "port")
            shutil.copytree(root, copy)
            (copy / "model/vocab_head.py").write_text("# modified model\n")
            with self.assertRaisesRegex(RuntimeError, "model/vocab_head.py"):
                verify_source(copy)


if __name__ == "__main__":
    unittest.main()
