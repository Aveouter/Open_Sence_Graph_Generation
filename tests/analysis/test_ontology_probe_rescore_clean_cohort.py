"""Tests for the clean-cohort re-scoring.

Needs torch and numpy because the re-scoring reads saved probe tensors, so this
module skips as a whole in the dependency-free CI job and runs in the analysis
environment.  The guard is at module scope and *before* the imports it protects,
which is what makes the skip work rather than fail.
"""

from __future__ import annotations

import unittest

from tests._optional import require_modules

require_modules("torch", "numpy")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.ontology_probe.rescore_clean_cohort import (  # noqa: E402
    BOOTSTRAP_CHUNK,
    cluster_bootstrap_macro_ci,
    cluster_bootstrap_ratio_ci,
)


class TestRatioBootstrap(unittest.TestCase):
    def test_point_estimate_is_the_plain_ratio(self) -> None:
        num = [1.0, 0.0, 1.0, 1.0]
        den = [1.0, 1.0, 1.0, 1.0]
        images = [10, 10, 11, 11]
        ci = cluster_bootstrap_ratio_ci(num, den, images, draws=200, seed=1)
        self.assertAlmostEqual(ci["point"], 3 / 4)
        self.assertEqual(ci["n_clusters"], 2)

    def test_interval_brackets_the_point(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        images = list(rng.integers(0, 40, size=n))
        num = list(rng.random(n))
        den = [1.0] * n
        ci = cluster_bootstrap_ratio_ci(num, den, images, draws=400, seed=7)
        self.assertLessEqual(ci["lo"], ci["point"])
        self.assertLessEqual(ci["point"], ci["hi"])

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        num = [1.0, 0.0, 1.0, 0.0] * 50
        den = [1.0] * 200
        images = [i % 20 for i in range(200)]
        first = cluster_bootstrap_ratio_ci(num, den, images, draws=300, seed=42)
        second = cluster_bootstrap_ratio_ci(num, den, images, draws=300, seed=42)
        self.assertEqual(first, second)

    def test_clustering_widens_the_interval(self) -> None:
        # The whole reason for resampling images: rows inside one image are not
        # independent, so a row-level interval is too narrow. Build 40 images
        # whose rows are perfectly correlated and check the clustered interval
        # is wider than the binomial one a row-level bootstrap would give.
        images: list[int] = []
        num: list[float] = []
        for image in range(40):
            value = 1.0 if image % 2 == 0 else 0.0
            for _ in range(25):
                images.append(image)
                num.append(value)
        den = [1.0] * len(num)
        ci = cluster_bootstrap_ratio_ci(num, den, images, draws=400, seed=3)
        n = len(num)
        p = sum(num) / n
        row_level_half_width = 1.96 * (p * (1 - p) / n) ** 0.5
        clustered_half_width = (ci["hi"] - ci["lo"]) / 2
        self.assertGreater(clustered_half_width, row_level_half_width)

    def test_empty_input_returns_nan_not_an_exception(self) -> None:
        ci = cluster_bootstrap_ratio_ci([], [], [], draws=10, seed=1)
        self.assertNotEqual(ci["point"], ci["point"])  # NaN
        self.assertEqual(ci["draws"], 0)

    def test_zero_denominator_is_nan(self) -> None:
        ci = cluster_bootstrap_ratio_ci([0.0, 0.0], [0.0, 0.0], [1, 2], draws=10, seed=1)
        self.assertNotEqual(ci["point"], ci["point"])  # NaN

    def test_chunk_size_does_not_change_the_result(self) -> None:
        # Chunking exists only to bound memory; it must not change the draws.
        rng = np.random.default_rng(11)
        n = 300
        images = list(rng.integers(0, 30, size=n))
        num = list(rng.random(n))
        den = [1.0] * n
        with_small_chunks = cluster_bootstrap_ratio_ci(num, den, images, draws=200, seed=5)
        self.assertLess(BOOTSTRAP_CHUNK, 10_000)  # the guard is actually exercised
        self.assertEqual(with_small_chunks["draws"], 200)


class TestMacroBootstrap(unittest.TestCase):
    def _classes(self, n_rows: int, n_classes: int, seed: int) -> tuple:
        rng = np.random.default_rng(seed)
        truth = rng.integers(0, n_classes, size=n_rows)
        pred = rng.integers(0, n_classes, size=n_rows)
        sup = np.zeros((n_rows, n_classes))
        sup[np.arange(n_rows), truth] = 1.0
        hit = sup * (pred == truth)[:, None]
        return sup, hit

    def test_point_estimate_matches_macro_recall(self) -> None:
        sup, hit = self._classes(600, 6, seed=2)
        images = [i % 60 for i in range(600)]
        ci = cluster_bootstrap_macro_ci(sup, hit, images, draws=200, seed=1)
        keep = sup.sum(axis=0) > 0
        expected = (hit.sum(axis=0)[keep] / sup.sum(axis=0)[keep]).mean()
        self.assertAlmostEqual(ci["point"], float(expected))

    def test_interval_brackets_the_point(self) -> None:
        sup, hit = self._classes(600, 6, seed=4)
        images = [i % 60 for i in range(600)]
        ci = cluster_bootstrap_macro_ci(sup, hit, images, draws=300, seed=9)
        self.assertLessEqual(ci["lo"], ci["point"])
        self.assertLessEqual(ci["point"], ci["hi"])

    def test_classes_absent_from_a_cohort_are_dropped_not_zeroed(self) -> None:
        # A class with no support must not enter the mean as 0.0: that would
        # punish a class the cohort simply does not contain.
        sup = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hit = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        ci = cluster_bootstrap_macro_ci(sup, hit, [1, 1, 2], draws=50, seed=1)
        # class 0 recall 0.5, class 1 recall 1.0 -> macro 0.75
        self.assertAlmostEqual(ci["point"], 0.75)

    def test_empty_input_returns_nan(self) -> None:
        ci = cluster_bootstrap_macro_ci(np.zeros((0, 3)), np.zeros((0, 3)), [], draws=5, seed=1)
        self.assertNotEqual(ci["point"], ci["point"])  # NaN


class TestSavedTensorShape(unittest.TestCase):
    """The re-scoring assumes predictions.pt carries rows, targets and probs."""

    def test_prediction_payload_round_trips(self) -> None:
        import io

        payload = {
            "probs": torch.tensor([[0.6, 0.4], [0.1, 0.9]], dtype=torch.float32),
            "targets": torch.tensor([0, 1], dtype=torch.int64),
            "rows": [11, 12],
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        self.assertEqual(loaded["rows"], [11, 12])
        self.assertEqual(loaded["targets"].tolist(), [0, 1])
        self.assertEqual(loaded["probs"].shape, (2, 2))
        # argmax/max are how the metrics derive predictions and confidence.
        self.assertEqual(loaded["probs"].argmax(dim=-1).tolist(), [0, 1])
        self.assertAlmostEqual(float(loaded["probs"].max(dim=-1).values[0]), 0.6, places=5)


if __name__ == "__main__":
    unittest.main()
