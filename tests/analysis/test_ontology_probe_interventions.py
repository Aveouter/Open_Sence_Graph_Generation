"""Tests for crop geometry, cache sharding, the M7 shuffle and the M8 rescue rate.

The crop and sharding tests are stdlib-only; the permutation and rescue tests
need torch and skip individually rather than skipping the whole module.
"""

from __future__ import annotations

import unittest

from tests._optional import skip_unless
from tools.ontology_probe.diagnostics_table import VERDICT_RULES, classify
from tools.ontology_probe.extract_features import expanded_crop, union_crop
from tools.ontology_probe.feature_cache import plan_shards


class ExpandedCropTest(unittest.TestCase):
    def test_grows_about_the_centre_by_the_context_scale(self) -> None:
        # 40x20 box centred at (50, 50) in a 200x200 image, scale 1.5.
        crop = expanded_crop((30.0, 40.0, 70.0, 60.0), (200, 200), 1.5)
        self.assertEqual(crop, (20, 35, 80, 65))
        self.assertEqual(crop[2] - crop[0], 60)
        self.assertEqual(crop[3] - crop[1], 30)

    def test_scale_one_leaves_the_box_unchanged(self) -> None:
        crop = expanded_crop((30.0, 40.0, 70.0, 60.0), (200, 200), 1.0)
        self.assertEqual(crop, (30, 40, 70, 60))

    def test_crop_is_clamped_to_the_image(self) -> None:
        crop = expanded_crop((-10.0, -10.0, 20.0, 20.0), (100, 100), 2.0)
        self.assertGreaterEqual(crop[0], 0)
        self.assertGreaterEqual(crop[1], 0)
        self.assertLessEqual(crop[2], 100)
        self.assertLessEqual(crop[3], 100)

    def test_height_and_width_are_respected_separately(self) -> None:
        """A 200-wide, 100-tall image must clamp x and y against different bounds."""
        crop = expanded_crop((50.0, 50.0, 150.0, 60.0), (100, 200), 3.0)
        self.assertLessEqual(crop[2], 200)  # x clamped by width
        self.assertLessEqual(crop[3], 100)  # y clamped by height

    def test_union_crop_spans_both_boxes(self) -> None:
        crop = union_crop((10.0, 10.0, 30.0, 30.0), (70.0, 60.0, 90.0, 80.0), (200, 200), 1.0)
        self.assertEqual(crop, (10, 10, 90, 80))

    def test_union_crop_of_overlapping_boxes_is_the_larger(self) -> None:
        crop = union_crop((10.0, 10.0, 50.0, 50.0), (20.0, 20.0, 40.0, 40.0), (200, 200), 1.0)
        self.assertEqual(crop, (10, 10, 50, 50))


class PlanShardsTest(unittest.TestCase):
    def test_partitions_in_order_without_loss(self) -> None:
        shards = plan_shards(list(range(1, 26)), 10)
        self.assertEqual([len(s) for s in shards], [10, 10, 5])
        self.assertEqual([i for s in shards for i in s], list(range(1, 26)))

    def test_empty_input_gives_no_shards(self) -> None:
        self.assertEqual(plan_shards([], 10), [])

    def test_single_partial_shard(self) -> None:
        self.assertEqual(plan_shards([7, 8], 10), [[7, 8]])


class ClassifyTest(unittest.TestCase):
    def test_low_support_is_degenerate_regardless_of_delta(self) -> None:
        verdict = classify(
            {"support": VERDICT_RULES["min_support"] - 1, "delta_visual": 0.9, "shuffle_drop": 0.9}
        )
        self.assertEqual(verdict, "degenerate_support")

    def test_visual_gain_with_matching_shuffle_drop_is_the_strong_result(self) -> None:
        verdict = classify(
            {
                "support": 1000,
                "delta_visual": VERDICT_RULES["delta_visual_min"],
                "shuffle_drop": VERDICT_RULES["shuffle_drop_min"],
            }
        )
        self.assertEqual(verdict, "vision_generalizes")

    def test_delta_without_a_shuffle_drop_is_flagged_not_accepted(self) -> None:
        """The case the rules exist to catch: gain that is not visual."""
        verdict = classify(
            {"support": 1000, "delta_visual": 0.20, "shuffle_drop": 0.001}
        )
        self.assertEqual(verdict, "delta_without_correspondence")

    def test_non_positive_delta_is_prior_dominated(self) -> None:
        verdict = classify({"support": 1000, "delta_visual": -0.01, "shuffle_drop": 0.0})
        self.assertEqual(verdict, "prior_or_geometry_dominated")

    def test_missing_delta_is_inconclusive(self) -> None:
        self.assertEqual(
            classify({"support": 1000, "delta_visual": None}), "inconclusive"
        )

    def test_delta_without_shuffle_measurement_is_unverified(self) -> None:
        verdict = classify({"support": 1000, "delta_visual": 0.2, "shuffle_drop": None})
        self.assertEqual(verdict, "delta_unverified_against_shuffle")


@skip_unless("torch")
class PermutationTest(unittest.TestCase):
    def test_is_a_permutation(self) -> None:
        from tools.ontology_probe.shuffle_test import permutation

        perm = permutation(100, seed=7)
        self.assertEqual(sorted(perm.tolist()), list(range(100)))

    def test_is_deterministic_for_a_seed(self) -> None:
        from tools.ontology_probe.shuffle_test import permutation

        self.assertEqual(
            permutation(50, seed=3).tolist(), permutation(50, seed=3).tolist()
        )
        self.assertNotEqual(
            permutation(50, seed=3).tolist(), permutation(50, seed=4).tolist()
        )

    def test_nothing_moves_when_every_group_is_a_singleton(self) -> None:
        from tools.ontology_probe.shuffle_test import permutation

        perm = permutation(5, seed=1, groups=[0, 1, 2, 3, 4])
        self.assertEqual(perm.tolist(), [0, 1, 2, 3, 4])

    def test_grouped_permutation_stays_inside_each_group(self) -> None:
        """within_predicate must preserve the predicate of every row."""
        from tools.ontology_probe.shuffle_test import permutation

        groups = [0] * 4 + [1] * 3 + [2] * 5
        perm = permutation(len(groups), seed=11, groups=groups)
        for target, source in enumerate(perm.tolist()):
            self.assertEqual(groups[target], groups[source])

    def test_grouped_permutation_actually_moves_rows(self) -> None:
        from tools.ontology_probe.shuffle_test import permutation

        groups = [0] * 200
        perm = permutation(len(groups), seed=5, groups=groups)
        self.assertNotEqual(perm.tolist(), list(range(200)))

    def test_unpermutable_groups_are_left_alone(self) -> None:
        from tools.ontology_probe.shuffle_test import permutation

        perm = permutation(3, seed=2, groups=[0, 1, 1])
        self.assertEqual(perm[0].item(), 0)


@skip_unless("torch")
class RescueAnalysisTest(unittest.TestCase):
    def _cells(self, base_pred, base_conf, visual_pred, labels):
        return (
            {
                "cell": "base",
                "rows": list(range(len(labels))),
                "labels": labels,
                "predictions": base_pred,
                "confidence": base_conf,
            },
            {
                "cell": "visual",
                "rows": list(range(len(labels))),
                "labels": labels,
                "predictions": visual_pred,
                "confidence": [1.0] * len(labels),
            },
        )

    def _cmap(self, n_classes: int = 4):
        from tools.ontology_probe.canonical_map import identity_map

        names = ["__background__"] + [f"p{i}" for i in range(1, 60)]
        cmap = identity_map(names)
        return cmap if cmap.n_classes >= n_classes else cmap

    def test_rescue_and_harm_on_a_hand_computed_case(self) -> None:
        from tools.ontology_probe.rescue_rate import rescue_analysis

        # 4 relations: base right on 0 and 3; visual fixes 1 but breaks 3.
        labels = [0, 1, 2, 0]
        base_pred = [0, 2, 3, 0]
        visual_pred = [0, 1, 3, 2]
        base, visual = self._cells(base_pred, [0.9] * 4, visual_pred, labels)
        result = rescue_analysis(base, visual, self._cmap(), thresholds=(0.5,))

        # base wrong on {1, 2}; visual correct on 1 only -> VRR = 0.5
        self.assertAlmostEqual(result["vrr"], 0.5)
        # base right on {0, 3}; visual wrong on 3 -> harm = 0.5
        self.assertAlmostEqual(result["harm_rate"], 0.5)
        self.assertAlmostEqual(result["n_base_wrong"], 2)
        # one rescue, one harm -> net zero
        self.assertAlmostEqual(result["net_rescue"], 0.0)

    def test_high_confidence_subset_is_restricted_by_tau(self) -> None:
        from tools.ontology_probe.rescue_rate import rescue_analysis

        labels = [0, 1]
        base_pred = [2, 3]  # both wrong
        visual_pred = [0, 1]  # both rescued
        base, visual = self._cells(base_pred, [0.95, 0.55], visual_pred, labels)
        result = rescue_analysis(base, visual, self._cmap(), thresholds=(0.9, 0.5))
        self.assertEqual(result["by_confidence"]["tau_0.9"]["n_relations"], 1)
        self.assertEqual(result["by_confidence"]["tau_0.5"]["n_relations"], 2)
        self.assertAlmostEqual(result["by_confidence"]["tau_0.9"]["vrr"], 1.0)

    def test_empty_subset_reports_nan_rather_than_zero(self) -> None:
        """No confidently-wrong relations must not read as 'rescued none'."""
        from tools.ontology_probe.rescue_rate import rescue_analysis

        labels = [0, 1]
        base, visual = self._cells([0, 2], [0.6, 0.6], [0, 1], labels)
        result = rescue_analysis(base, visual, self._cmap(), thresholds=(0.99,))
        self.assertEqual(result["by_confidence"]["tau_0.99"]["n_relations"], 0)
        self.assertNotEqual(result["by_confidence"]["tau_0.99"]["vrr"], 0.0)

    def test_misaligned_cells_are_rejected(self) -> None:
        from tools.ontology_probe.rescue_rate import align

        a = {"cell": "a", "rows": [0, 1], "labels": [0, 1]}
        b = {"cell": "b", "rows": [1, 0], "labels": [0, 1]}
        with self.assertRaises(AssertionError):
            align(a, b)

    def test_differing_labels_are_rejected(self) -> None:
        from tools.ontology_probe.rescue_rate import align

        a = {"cell": "a", "rows": [0, 1], "labels": [0, 1]}
        b = {"cell": "b", "rows": [0, 1], "labels": [1, 0]}
        with self.assertRaises(AssertionError):
            align(a, b)


if __name__ == "__main__":
    unittest.main()
