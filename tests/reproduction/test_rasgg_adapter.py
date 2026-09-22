from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tests._optional import require_modules

require_modules("numpy", "torch", "lightning")

import numpy as np
import torch

from src.exp import (
    _allow_checkpoint_shape_adaptation,
    _remap_external_checkpoint,
)
from src.methods import method_maps
from src.models.ra_sgg import RASGGModel, build_ra_sgg


class RASGGAdapterTest(unittest.TestCase):
    def test_external_checkpoint_remap_supports_both_signatures(self) -> None:
        state_dict = {"weight": torch.ones(1)}
        extractor = object()

        class LegacyRemapper:
            def remap_external_state_dict(self, state):
                self.state = state
                return {"legacy": state["weight"]}

        class VisualRemapper:
            def remap_external_state_dict(self, state, visual_extractor=None):
                self.state = state
                self.visual_extractor = visual_extractor
                return {"visual": state["weight"]}

        legacy = LegacyRemapper()
        visual = VisualRemapper()
        self.assertIn(
            "legacy",
            _remap_external_checkpoint(legacy, state_dict, extractor),
        )
        self.assertIs(legacy.state, state_dict)
        self.assertIn(
            "visual",
            _remap_external_checkpoint(visual, state_dict, extractor),
        )
        self.assertIs(visual.state, state_dict)
        self.assertIs(visual.visual_extractor, extractor)

    def test_rasgg_checkpoint_shape_adaptation_is_disabled(self) -> None:
        for method_name in ("RA_SGG", "ra_sgg", "RA-SGG", "rasgg"):
            self.assertFalse(_allow_checkpoint_shape_adaptation(method_name))
        self.assertTrue(_allow_checkpoint_shape_adaptation("Motifs"))

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
        model.eval()
        # The official PE-Net predictor consumes the 4096-dimensional output
        # of its FPN box and union feature extractors.
        visual_feats = torch.randn(3, 4096)
        union_feats = torch.randn(6, 4096)
        boxes = torch.rand(3, 4)
        labels = torch.tensor([1, 2, 3])
        rel_labels = torch.zeros(6, dtype=torch.long)

        outputs = model(
            roi_features=visual_feats,
            union_features=union_feats,
            labels=labels,
            boxes=boxes,
            rel_labels=rel_labels,
        )

        self.assertEqual(outputs["rel_logits"].shape, (6, 51))
        self.assertEqual(outputs["pair_indices"].shape, (6, 2))
        self.assertEqual(outputs["obj_labels"].shape, (3,))
        self.assertEqual(outputs["add_losses"], {})
        self.assertEqual(outputs["add_data"], {})

    def test_official_numpy_memory_file_loads(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "featurebank.npy"
            np.save(
                memory_path,
                {
                    "key": np.ones((2, 32), dtype=np.float32),
                    "value": np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64),
                },
                allow_pickle=True,
            )

            model = RASGGModel(
                num_classes=151,
                num_predicates=51,
                visual_dim=16,
                mlp_dim=32,
                hidden_dim=32,
                embed_dim=8,
                use_freq_bias=False,
                memory_bank_path=str(memory_path),
            )

        self.assertEqual(tuple(model.fb_keys.shape), (2, 32))
        self.assertEqual(tuple(model.fb_values.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
