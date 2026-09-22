"""Tests for probe metrics, paired tests, and the prior baselines.

Stdlib only, so this runs in the dependency-free CI `validate` job.
"""

from __future__ import annotations

import math
import unittest

from tools.ontology_probe.canonical_map import identity_map, load_canonical_map
from tools.ontology_probe.prior_baselines import (
    B0Predictor,
    B1LookupPredictor,
    build_prior_table,
    calibrate_prior,
    conflict_mask,
    pair_ambiguity,
)
from tools.ontology_probe.probe_metrics import (
    _norm_ppf,
    accuracy,
    hbt_strata,
    macro_recall,
    mcnemar_test,
    mean_nll,
    mean_recall_at_k,
    micro_metric_warning,
    paired_accuracy_diff,
    paired_mean_diff,
    per_class_report,
    ranked_predictions,
    summarise,
)
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    undirected_pair_key,
)

PREDICATE_NAMES = ["__background__"] + [f"p{i}" for i in range(1, 51)]


def make_table(rows):
    table = RelationTable()
    for image_id, sub_idx, obj_idx, pred_id, c_s, c_o in rows:
        table.image_id.append(image_id)
        table.sub_idx.append(sub_idx)
        table.obj_idx.append(obj_idx)
        table.pred_id.append(pred_id)
        table.c_s.append(c_s)
        table.c_o.append(c_o)
    return table


class MetricMathTest(unittest.TestCase):
    def test_accuracy_and_macro_recall_diverge_as_expected(self) -> None:
        # 3 rows of class 0, 1 row of class 1, all predicted class 0.
        y_true = [0, 0, 0, 1]
        y_pred = [0, 0, 0, 0]
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.75)
        # macro recall = (recall of 0 = 1.0, recall of 1 = 0.0) / 2
        self.assertAlmostEqual(macro_recall(y_true, y_pred), 0.5)

    def test_macro_recall_restricted_to_class_subset(self) -> None:
        y_true = [0, 0, 0, 1]
        y_pred = [0, 0, 0, 0]
        self.assertAlmostEqual(macro_recall(y_true, y_pred, classes=[0]), 1.0)
        self.assertAlmostEqual(macro_recall(y_true, y_pred, classes=[1]), 0.0)

    def test_mr1_equals_macro_recall(self) -> None:
        """Documented identity: mR@1 is macro recall, not a second result."""
        y_true = [0, 1, 2, 2, 1]
        probs = [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
            [0.1, 0.8, 0.1],
        ]
        y_pred = [order[0] for order in ranked_predictions(probs)]
        self.assertAlmostEqual(
            mean_recall_at_k(y_true, probs, 1), macro_recall(y_true, y_pred)
        )

    def test_mr_k_is_monotonic_in_k(self) -> None:
        y_true = [0, 1, 2]
        probs = [[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.34, 0.33, 0.33]]
        values = [mean_recall_at_k(y_true, probs, k) for k in (1, 2, 3)]
        self.assertLessEqual(values[0], values[1])
        self.assertLessEqual(values[1], values[2])
        self.assertAlmostEqual(values[2], 1.0)

    def test_mean_nll_matches_hand_computation(self) -> None:
        probs = [[0.5, 0.5], [0.25, 0.75]]
        expected = (-math.log(0.5) - math.log(0.75)) / 2
        self.assertAlmostEqual(mean_nll([0, 1], probs), expected)

    def test_mean_nll_clamps_zero_probability(self) -> None:
        self.assertTrue(math.isfinite(mean_nll([0], [[0.0, 1.0]])))

    def test_length_mismatch_raises(self) -> None:
        for func in (accuracy, macro_recall):
            with self.assertRaises(ValueError):
                func([0, 1], [0])

    def test_per_class_report_has_support_and_nll(self) -> None:
        rows = per_class_report([0, 0, 1], [0, 1, 1], [[0.6, 0.4]] * 3, ["a", "b"])
        self.assertEqual([r["class_name"] for r in rows], ["a", "b"])
        self.assertEqual(rows[0]["support"], 2)
        self.assertAlmostEqual(rows[0]["accuracy"], 0.5)
        self.assertIsNotNone(rows[0]["mean_nll"])

    def test_hbt_strata_averages_within_groups(self) -> None:
        rows = [
            {"class_name": "a", "accuracy": 1.0},
            {"class_name": "b", "accuracy": 0.0},
            {"class_name": "c", "accuracy": 0.5},
        ]
        strata = hbt_strata(rows, {"a": "head", "b": "head", "c": "tail"})
        self.assertAlmostEqual(strata["head"], 0.5)
        self.assertAlmostEqual(strata["tail"], 0.5)

    def test_summarise_carries_the_micro_metric_warning(self) -> None:
        report = summarise([0, 1], [[0.6, 0.4], [0.3, 0.7]])
        self.assertIn("micro_metric_warning", report)
        self.assertEqual(report["mR@1_is_macro_recall"], True)
        self.assertIn("`on`", micro_metric_warning())


class PairedTestTest(unittest.TestCase):
    def test_identical_predictions_give_zero_difference(self) -> None:
        correct = [1, 1, 0, 0, 1]
        diff = paired_accuracy_diff(correct, correct)
        self.assertAlmostEqual(diff["diff"], 0.0)
        self.assertAlmostEqual(diff["se"], 0.0)
        self.assertEqual(mcnemar_test(correct, correct)["p_value"], 1.0)

    def test_difference_sign_follows_the_first_argument(self) -> None:
        a = [1, 1, 1, 1]
        b = [1, 0, 0, 0]
        self.assertGreater(paired_accuracy_diff(a, b)["diff"], 0.0)
        self.assertLess(paired_accuracy_diff(b, a)["diff"], 0.0)

    def test_ci_covers_zero_for_a_null_difference(self) -> None:
        a = [1, 0] * 500
        b = [0, 1] * 500
        diff = paired_accuracy_diff(a, b)
        self.assertLessEqual(diff["lo"], 0.0)
        self.assertGreaterEqual(diff["hi"], 0.0)
        self.assertEqual(diff["ci_method"], "multinomial_normal")

    def test_mcnemar_matches_hand_computed_exact_p(self) -> None:
        # b = 10, c = 2, n = 12 -> 2 * sum_{k<=2} C(12,k) / 2^12
        a = [1] * 10 + [0] * 2
        b = [0] * 10 + [1] * 2
        expected = 2 * sum(math.comb(12, k) for k in range(3)) / 2**12
        result = mcnemar_test(a, b)
        self.assertAlmostEqual(result["p_value"], min(1.0, expected))
        self.assertEqual(result["method"], "exact")
        self.assertEqual((result["b"], result["c"]), (10, 2))

    def test_mcnemar_uses_fraction_not_float_power(self) -> None:
        """2.0**n overflows a float above ~1024 discordant pairs."""
        a = [1] * 800 + [0] * 400
        b = [0] * 800 + [1] * 400
        result = mcnemar_test(a, b)
        self.assertTrue(math.isfinite(result["p_value"]))
        self.assertGreaterEqual(result["p_value"], 0.0)

    def test_mcnemar_switches_to_approximation_when_large(self) -> None:
        a = [1] * 4000 + [0] * 4000
        b = [0] * 4000 + [1] * 4000
        result = mcnemar_test(a, b)
        self.assertEqual(result["method"], "normal_approximation")
        self.assertTrue(math.isfinite(result["p_value"]))

    def test_mcnemar_p_is_never_above_one(self) -> None:
        a = [1, 0, 1, 0]
        b = [0, 1, 0, 1]
        self.assertLessEqual(mcnemar_test(a, b)["p_value"], 1.0)

    def test_paired_mean_diff_matches_hand_computation(self) -> None:
        result = paired_mean_diff([1.0, 2.0, 3.0], [0.0, 1.0, 1.0])
        self.assertAlmostEqual(result["diff"], 4.0 / 3.0)
        self.assertGreater(result["se"], 0.0)
        self.assertEqual(result["ci_method"], "normal_se")

    def test_norm_ppf_matches_known_quantiles(self) -> None:
        self.assertAlmostEqual(_norm_ppf(0.975), 1.959964, places=4)
        self.assertAlmostEqual(_norm_ppf(0.5), 0.0, places=6)
        self.assertAlmostEqual(_norm_ppf(0.995), 2.575829, places=4)


class PriorBaselineTest(unittest.TestCase):
    def _fixture(self):
        # pair (3,4) carries p1 (5x) and p2 (1x); pair (1,2) carries p3 (2x).
        rows = [(i, 0, 1, 1, 3, 4) for i in range(1, 6)]
        rows += [(6, 0, 1, 2, 3, 4)]
        rows += [(7, 0, 1, 3, 1, 2), (8, 0, 1, 3, 1, 2)]
        return make_table(rows)

    def test_backoff_formula_matches_hand_computation(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        probs = B1LookupPredictor(prior, alpha=1.0).predict_probs(table, [0])[0]
        # P(p1 | (3,4)) = (5 + 1 * (5/8)) / (6 + 1)
        global_p1 = 5 / 8
        self.assertAlmostEqual(probs[0], (5 + 1 * global_p1) / 7)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_unseen_pair_falls_back_to_the_global_distribution(self) -> None:
        """The degeneracy that makes C_prior_conflict empty under pair-OOD."""
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        unseen_row = make_table([(99, 0, 1, 1, 40, 41)])
        lookup = B1LookupPredictor(prior, alpha=1.0).predict_probs(unseen_row, [0])[0]
        self.assertEqual(lookup, prior.global_probs)
        b0 = B0Predictor(prior).predict_probs(unseen_row, [0])[0]
        self.assertEqual(lookup, b0)

    def test_conflict_mask_excludes_rows_without_pair_support(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        # Row 6 is p2 on a pair whose argmax is p1 -> conflict.
        # Row 0 is p1 on its own pair -> no conflict.
        mask = conflict_mask(prior, table, [0, 6], [0, 1])
        self.assertEqual(mask, [False, True])

    def test_conflict_mask_ignores_unseen_pairs_entirely(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        unseen = make_table([(99, 0, 1, 1, 40, 41)])
        # Class index 49 (predicate 50) on a pair with zero support: the prior is
        # a constant, so "the prior is wrong" carries no information.
        mask = conflict_mask(prior, unseen, [0], [49])
        self.assertEqual(mask, [False])

    def test_b0_is_constant_across_all_rows(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        probs = B0Predictor(prior).predict_probs(table, [0, 1, 2])
        self.assertEqual(probs[0], probs[1])
        self.assertEqual(probs[1], probs[2])

    def test_calibration_selects_on_dev_nll(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        labels = [identity_map(PREDICATE_NAMES).remap(table.pred_id[r]) for r in range(len(table))]
        result = calibrate_prior(prior, table, list(range(len(table))), labels)
        self.assertIn(result["alpha"], (0.5, 1.0, 5.0, 20.0))
        self.assertIn(result["temperature"], (0.25, 0.5, 1.0, 2.0, 4.0))
        self.assertTrue(math.isfinite(result["dev_nll"]))
        self.assertEqual(len(result["grid"]), 20)

    def test_pair_ambiguity_counts_multi_predicate_pairs(self) -> None:
        table = self._fixture()
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        ambiguity = pair_ambiguity(prior)
        self.assertEqual(ambiguity["n_pairs_seen"], 2)
        self.assertEqual(ambiguity["n_pairs_multi_predicate"], 1)

    def test_source_split_id_is_required_for_traceability(self) -> None:
        table = self._fixture()
        prior = build_prior_table(
            table, range(len(table)), identity_map(PREDICATE_NAMES), source_split_id="pair_ood"
        )
        self.assertEqual(prior.source_split_id, "pair_ood")
        self.assertEqual(B1LookupPredictor(prior).describe()["source_split_id"], "pair_ood")


class PoolingIdentityTest(unittest.TestCase):
    """The identity claimed in prior_baselines: pooling == rebuilding, exactly."""

    def test_dirichlet_backoff_prior_pools_exactly(self) -> None:
        rows = [(i, 0, 1, 1, 3, 4) for i in range(1, 6)]
        rows += [(6, 0, 1, 2, 3, 4)]
        rows += [(7, 0, 1, 3, 1, 2), (8, 0, 1, 3, 1, 2)]
        rows += [(9, 0, 1, 41, 3, 4), (10, 0, 1, 46, 3, 4)]  # standing on / walking on
        table = make_table(rows)
        vg50 = identity_map(PREDICATE_NAMES)
        l2 = load_canonical_map(level="L2_entail")

        fine_prior = build_prior_table(table, range(len(table)), vg50)
        canon_prior = build_prior_table(table, range(len(table)), l2)

        for alpha in (0.5, 1.0, 5.0, 20.0):
            with self.subTest(alpha=alpha):
                fine_probs = B1LookupPredictor(fine_prior, alpha=alpha).predict_probs(
                    table, [0, 6, 8]
                )
                pooled = [l2.pool_probs(row) for row in fine_probs]
                direct = B1LookupPredictor(canon_prior, alpha=alpha).predict_probs(
                    table, [0, 6, 8]
                )
                for a, b in zip(pooled, direct, strict=True):
                    for x, y in zip(a, b, strict=True):
                        self.assertAlmostEqual(x, y, places=12)

    def test_unseen_pair_pools_to_unseen_pair(self) -> None:
        table = make_table([(i, 0, 1, 1, 3, 4) for i in range(1, 4)])
        vg50 = identity_map(PREDICATE_NAMES)
        l2 = load_canonical_map(level="L2_entail")
        fine_prior = build_prior_table(table, range(len(table)), vg50)
        canon_prior = build_prior_table(table, range(len(table)), l2)
        unseen = make_table([(99, 0, 1, 1, 40, 41)])
        pooled = l2.pool_probs(
            B1LookupPredictor(fine_prior).predict_probs(unseen, [0])[0]
        )
        direct = B1LookupPredictor(canon_prior).predict_probs(unseen, [0])[0]
        for x, y in zip(pooled, direct, strict=True):
            self.assertAlmostEqual(x, y, places=12)


class PairKeyConsistencyTest(unittest.TestCase):
    def test_prior_pair_lookup_matches_undirected_key(self) -> None:
        """The prior must treat (a,b) and (b,a) as the same pair."""
        table = make_table([(1, 0, 1, 1, 3, 4), (2, 0, 1, 1, 4, 3)])
        prior = build_prior_table(table, range(len(table)), identity_map(PREDICATE_NAMES))
        self.assertTrue(prior.is_pair_seen(3, 4))
        self.assertTrue(prior.is_pair_seen(4, 3))
        self.assertEqual(undirected_pair_key(3, 4), undirected_pair_key(4, 3))


if __name__ == "__main__":
    unittest.main()
