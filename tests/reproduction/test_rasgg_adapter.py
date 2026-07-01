from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from src.methods import method_maps
from src.models.ra_sgg import RASGGModel, build_ra_sgg


class RASGGAdapterTest(unittest.TestCase):
    def test_method_is_registered(self) -> None:
        self.assertIn("ra_sgg", method_maps)
        self.assertEqual(method_maps["ra_sgg"].__name__, "RA_SGG_Method")

    def test_builder_creates_no_memory_adapter(self) -> None:
        model = build_ra_sgg(
            SimpleNamespace(
                entity_nums=151,
                rel_nums=51,
                visual_dim=16,
                hidden_dim=32,
                ra_sgg_embed_dim=8,
                use_freq_bias=False,
                dropout=0.0,
                ra_sgg_memory_bank_path=None,
                ra_sgg_retrieval_logit_coef=0.0,
            )
        )

        self.assertIsInstance(model, RASGGModel)

    def test_forward_returns_motifs_compatible_schema(self) -> None:
        torch.manual_seed(7)
        model = RASGGModel(
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
        self.assertEqual(outputs["ra_sgg_adapter_status"], "penet_no_memory")

    def test_invalid_memory_file_warns_and_falls_back(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "corrupt_memory.pt"
            memory_path.write_text("not a torch checkpoint")

            with self.assertWarnsRegex(
                RuntimeWarning,
                "memory distribution could not be loaded",
            ):
                model = RASGGModel(
                    num_classes=151,
                    num_predicates=51,
                    visual_dim=16,
                    hidden_dim=32,
                    embed_dim=8,
                    use_freq_bias=False,
                    memory_bank_path=str(memory_path),
                    retrieval_logit_coef=1.0,
                )

        self.assertEqual(model.memory_distribution.numel(), 0)


if __name__ == "__main__":
    unittest.main()
