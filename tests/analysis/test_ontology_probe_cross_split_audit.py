"""Tests for the cross-split isolation and prior-conflict audit.

Stdlib only, so this runs in the dependency-free CI `validate` job, and it uses
toy relation tables rather than VG150: the CI job has no dataset checkout, and
the properties under test are properties of the audit's logic, not of the data.

The load-bearing case is ``TestSplitSelfAuditIsNotEnough``.  It reproduces the
shape of the real defect -- an eval set that is disjoint from the split's own
train part and yet wholly contained in the fitting set the pipeline actually
uses -- because a checker that only compared a split against itself would report
that eval set as clean, which is exactly how the defect went unnoticed.
"""

from __future__ import annotations

import unittest

from tools.ontology_probe.audit_cross_split import (
    CONFIDENCE_THRESHOLDS,
    CountIndex,
    backoff_probs,
    build_count_index,
    check_pins,
    confidence_bins,
    conflict_flags,
    overlap_report,
)
from tools.ontology_probe.vg_annotations import RelationTable


def make_table(rows: list[tuple[int, int, int, int, int, int]]) -> RelationTable:
    """Build a toy table from ``(image_id, sub_idx, obj_idx, pred_id, c_s, c_o)``."""
    table = RelationTable()
    for image_id, sub_idx, obj_idx, pred_id, c_s, c_o in rows:
        table.image_id.append(image_id)
        table.sub_idx.append(sub_idx)
        table.obj_idx.append(obj_idx)
        table.pred_id.append(pred_id)
        table.c_s.append(c_s)
        table.c_o.append(c_o)
    return table


class TestOverlapReport(unittest.TestCase):
    def test_disjoint_sets_read_zero(self) -> None:
        table = make_table(
            [(1, 0, 1, 1, 10, 20), (1, 0, 1, 2, 10, 20), (2, 0, 1, 1, 10, 30)]
        )
        report = overlap_report(table, [0, 1], [2])
        self.assertEqual(report["shared_rows"], 0)
        self.assertEqual(report["shared_images"], 0)
        self.assertEqual(report["fraction_of_b_rows_shared"], 0.0)

    def test_detects_row_and_image_overlap(self) -> None:
        table = make_table(
            [(1, 0, 1, 1, 10, 20), (1, 0, 1, 2, 10, 20), (3, 0, 1, 1, 10, 30)]
        )
        report = overlap_report(table, [0, 1], [0, 2])
        self.assertEqual(report["shared_rows"], 1)
        self.assertEqual(report["shared_row_sample"], [0])
        self.assertEqual(report["shared_images"], 1)
        self.assertEqual(report["shared_image_sample"], [1])
        # Row 0 is shared and row 2 is not, so one of the two eval rows sits on
        # a fitting image.
        self.assertEqual(report["rows_in_b_on_shared_images"], 1)

    def test_fractions_are_of_the_second_set(self) -> None:
        table = make_table(
            [(1, 0, 1, 1, 10, 20), (2, 0, 1, 1, 10, 20), (3, 0, 1, 1, 10, 20)]
        )
        report = overlap_report(table, [0], [0, 1, 2, 3 - 1])
        self.assertEqual(report["n_rows_a"], 1)
        self.assertEqual(report["n_rows_b"], 3)
        self.assertAlmostEqual(report["fraction_of_b_rows_shared"], 1 / 3)
        self.assertAlmostEqual(report["fraction_of_b_images_shared"], 1 / 3)

    def test_empty_second_set_does_not_divide_by_zero(self) -> None:
        table = make_table([(1, 0, 1, 1, 10, 20)])
        report = overlap_report(table, [0], [])
        self.assertEqual(report["shared_rows"], 0)
        self.assertEqual(report["fraction_of_b_rows_shared"], 0.0)
        self.assertEqual(report["fraction_of_b_images_shared"], 0.0)


class TestSplitSelfAuditIsNotEnough(unittest.TestCase):
    """The failure mode: clean against its own train, dirty against the real one."""

    def setUp(self) -> None:
        # One row per image, so row index and image id are easy to line up.
        # Images 1 and 2 are the actual fitting set; image 3 is the eval split's
        # own train part; image 4 is its private holdout.
        #
        # The eval set is [1, 3]: row 1 is an image-2 relation the pipeline fit
        # on, and row 3 is the genuinely held-out one.  Shared *indices* are what
        # make this the real failure mode -- both splits index one shared relation
        # table, so the same index appearing in two row lists is the same
        # annotated relation, not a coincidence of numbering.
        self.table = make_table(
            [
                (1, 0, 1, 1, 10, 20),  # 0: fitting set, image 1
                (2, 0, 1, 1, 10, 20),  # 1: fitting set, image 2 -- also eval
                (3, 0, 1, 1, 10, 20),  # 2: eval split's own train, image 3
                (4, 0, 1, 1, 10, 20),  # 3: eval row, image 4 -- held out
            ]
        )
        self.fitting_rows = [0, 1]
        self.own_train = [2]
        self.eval_rows = [1, 3]

    def test_split_own_audit_channel_is_clean(self) -> None:
        report = overlap_report(self.table, self.own_train, self.eval_rows)
        self.assertEqual(report["shared_rows"], 0)
        self.assertEqual(report["shared_images"], 0)

    def test_fitting_set_channel_is_not(self) -> None:
        report = overlap_report(self.table, self.fitting_rows, self.eval_rows)
        self.assertEqual(report["shared_rows"], 1)
        self.assertEqual(report["shared_row_sample"], [1])
        self.assertEqual(report["shared_images"], 1)
        self.assertEqual(report["shared_image_sample"], [2])
        self.assertAlmostEqual(report["fraction_of_b_rows_shared"], 0.5)
        self.assertAlmostEqual(report["fraction_of_b_images_shared"], 0.5)

    def test_clean_cohort_is_the_complement(self) -> None:
        fitting_images = {self.table.image_id[row] for row in self.fitting_rows}
        clean = [
            row
            for row in self.eval_rows
            if self.table.image_id[row] not in fitting_images
        ]
        self.assertEqual(clean, [3])


class TestCountIndex(unittest.TestCase):
    def setUp(self) -> None:
        # Two relations on the (10, 20) pair, one in each direction, plus one on
        # (10, 30).  The direction conventions must disagree about the first two.
        self.table = make_table(
            [
                (1, 0, 1, 1, 10, 20),  # c_s=10, c_o=20, label 0
                (1, 1, 0, 2, 20, 10),  # c_s=20, c_o=10, label 1
                (2, 0, 1, 1, 10, 30),  # c_s=10, c_o=30, label 0
            ]
        )
        self.labels = [0, 1, 0]

    def test_unordered_pools_both_directions(self) -> None:
        index = build_count_index(self.table, [0, 1, 2], self.labels, 3, "unordered")
        self.assertEqual(len(index.pair_counts), 2)
        self.assertEqual(index.pair_totals[index.key_for(10, 20)], 2)
        self.assertEqual(index.pair_counts[index.key_for(20, 10)][0], 1.0)
        self.assertEqual(index.pair_counts[index.key_for(20, 10)][1], 1.0)

    def test_ordered_separates_directions(self) -> None:
        index = build_count_index(self.table, [0, 1, 2], self.labels, 3, "ordered")
        self.assertEqual(len(index.pair_counts), 3)
        self.assertEqual(index.pair_totals[index.key_for(10, 20)], 1)
        self.assertEqual(index.pair_totals[index.key_for(20, 10)], 1)

    def test_global_counts_ignore_pair_structure(self) -> None:
        index = build_count_index(self.table, [0, 1, 2], self.labels, 3, "unordered")
        self.assertEqual(index.global_counts, [2.0, 1.0, 0.0])

    def test_rejects_one_based_predicate_ids_as_labels(self) -> None:
        # pred_id is 1..50 and a class index is 0..49.  Passing table.pred_id
        # straight in is the off-by-one this guard exists to catch.
        with self.assertRaises(ValueError):
            build_count_index(
                self.table, [0, 1, 2], [1, 2, 1], 2, "unordered"
            )

    def test_rejects_unknown_direction(self) -> None:
        with self.assertRaises(ValueError):
            build_count_index(self.table, [0], [0], 3, "sideways")

    def test_rejects_misaligned_labels(self) -> None:
        with self.assertRaises(ValueError):
            build_count_index(self.table, [0, 1], [0], 3, "unordered")


class TestBackoffProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.table = make_table([(1, 0, 1, 1, 10, 20), (1, 1, 0, 1, 10, 20)])
        self.labels = [0, 1]
        self.index = build_count_index(self.table, [0, 1], self.labels, 2, "unordered")

    def test_unseen_pair_returns_none_not_the_global_distribution(self) -> None:
        self.assertIsNone(
            backoff_probs(self.index, self.index.key_for(99, 98), alpha=1.0)
        )

    def test_matches_the_closed_form(self) -> None:
        alpha = 1.0
        probs = backoff_probs(self.index, self.index.key_for(10, 20), alpha=alpha)
        # counts [1, 1], total 2, global [0.5, 0.5]
        expected = [(1 + alpha * 0.5) / (2 + alpha)] * 2
        for got, want in zip(probs, expected, strict=True):
            self.assertAlmostEqual(got, want)

    def test_large_alpha_approaches_the_global_distribution(self) -> None:
        probs = backoff_probs(self.index, self.index.key_for(10, 20), alpha=1e9)
        for got, want in zip(probs, self.index.global_probs, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_probs_sum_to_one(self) -> None:
        probs = backoff_probs(self.index, self.index.key_for(10, 20), alpha=5.0)
        self.assertAlmostEqual(sum(probs), 1.0)


class TestConflictFlags(unittest.TestCase):
    def setUp(self) -> None:
        self.table = make_table(
            [
                (1, 0, 1, 1, 10, 20),  # 0: pair seen, label 0
                (1, 1, 0, 1, 10, 20),  # 1: pair seen, label 0
                (1, 2, 3, 1, 10, 20),  # 2: pair seen, label 0
                (1, 4, 5, 2, 10, 99),  # 3: unseen pair
            ]
        )
        self.labels = [0, 0, 1, 0]
        self.index = build_count_index(self.table, [0, 1, 2], self.labels[:3], 2, "unordered")

    def test_unseen_pair_is_not_a_conflict(self) -> None:
        flags = conflict_flags(self.table, [3], [self.labels[3]], self.index, alpha=1.0)
        self.assertFalse(flags[0]["seen"])
        self.assertFalse(flags[0]["wrong"])
        self.assertEqual(flags[0]["support"], 0)

    def test_argmax_against_label_is_wrong(self) -> None:
        # Pair (10,20) has two counts for label 0 and one for label 1, so the
        # argmax is 0; row 2 carries label 1 and is therefore a conflict.
        flags = conflict_flags(self.table, [0, 2], [0, 1], self.index, alpha=1.0)
        self.assertFalse(flags[0]["wrong"])
        self.assertTrue(flags[1]["wrong"])

    def test_support_and_confidence_are_reported(self) -> None:
        flags = conflict_flags(self.table, [0], [0], self.index, alpha=1.0)
        self.assertEqual(flags[0]["support"], 3)
        self.assertGreater(flags[0]["confidence"], 0.0)
        self.assertLessEqual(flags[0]["confidence"], 1.0)

    def test_rejects_misaligned_rows_and_labels(self) -> None:
        with self.assertRaises(ValueError):
            conflict_flags(self.table, [0, 1], [0], self.index, alpha=1.0)


class TestConfidenceBins(unittest.TestCase):
    def test_bins_count_at_or_above_each_floor(self) -> None:
        flags = [
            {"seen": True, "wrong": True, "confidence": 0.95, "support": 3},
            {"seen": True, "wrong": True, "confidence": 0.75, "support": 3},
            {"seen": True, "wrong": True, "confidence": 0.55, "support": 3},
        ]
        bins = confidence_bins(flags)
        self.assertEqual(bins["tau>=0.5"], 3)
        self.assertEqual(bins["tau>=0.7"], 2)
        self.assertEqual(bins["tau>=0.9"], 1)

    def test_correct_rows_are_excluded(self) -> None:
        flags = [
            {"seen": True, "wrong": False, "confidence": 0.99, "support": 3},
            {"seen": True, "wrong": True, "confidence": 0.99, "support": 3},
        ]
        self.assertEqual(confidence_bins(flags)["tau>=0.9"], 1)

    def test_empty_cohort_reports_zero_for_every_floor(self) -> None:
        bins = confidence_bins([])
        self.assertEqual(set(bins), {f"tau>={t}" for t in CONFIDENCE_THRESHOLDS})
        self.assertTrue(all(value == 0 for value in bins.values()))


class TestCheckPins(unittest.TestCase):
    def test_matching_pins_pass(self) -> None:
        check_pins({"rel_json_sha256": "abc"}, {"rel_json_sha256": "abc"})

    def test_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            check_pins({"rel_json_sha256": "abc"}, {"rel_json_sha256": "def"})

    def test_absent_provenance_field_raises(self) -> None:
        with self.assertRaises(ValueError):
            check_pins({}, {"rel_json_sha256": "abc"})

    def test_empty_expectation_passes(self) -> None:
        check_pins({"rel_json_sha256": "abc"}, {})


class TestCountIndexIsReusable(unittest.TestCase):
    def test_index_is_independent_of_eval_rows(self) -> None:
        # The audit builds one index per direction from the fitting set and
        # reuses it for every cohort; scoring a different cohort must not change
        # the prior. Two indices built from the same rows must agree, and
        # scoring must leave both untouched.
        table = make_table([(1, 0, 1, 1, 10, 20), (2, 0, 1, 2, 10, 20)])
        labels = [0, 1]
        first = build_count_index(table, [0, 1], labels, 3, "unordered")
        second = build_count_index(table, [0, 1], labels, 3, "unordered")
        before = backoff_probs(first, first.key_for(10, 20), 1.0)
        conflict_flags(table, [0, 1], labels, second, alpha=1.0)
        after = backoff_probs(first, first.key_for(10, 20), 1.0)
        self.assertEqual(before, after)
        self.assertIsInstance(first, CountIndex)


if __name__ == "__main__":
    unittest.main()
