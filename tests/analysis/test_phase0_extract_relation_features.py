from __future__ import annotations

import unittest

import torch

from tools.analysis.phase0_extract_relation_features import final_decoder_features


class FinalDecoderFeaturesTest(unittest.TestCase):
    def test_selects_final_layer_from_layered_tensor(self) -> None:
        features = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

        selected = final_decoder_features(features)

        self.assertEqual(tuple(selected.shape), (3, 4, 5))
        torch.testing.assert_close(selected, features[-1])

    def test_keeps_already_final_tensor(self) -> None:
        features = torch.randn(3, 4, 5)

        selected = final_decoder_features(features)

        self.assertIs(selected, features)

    def test_rejects_unindexable_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "relation features must have shape"):
            final_decoder_features(torch.randn(4, 5))


if __name__ == "__main__":
    unittest.main()
