"""Tests for the canonical predicate label spaces and fine->canonical pooling.

Stdlib only, so this runs in the dependency-free CI `validate` job.
"""

from __future__ import annotations

import copy
import unittest

from tools.ontology_probe.canonical_map import (
    KIND_ENTAILMENT,
    KIND_NOISE,
    KIND_SINGLETON,
    CanonicalMap,
    CanonicalMapError,
    load_canonical_map,
    summarise,
    validate_canonical_map,
)
from tools.ontology_probe.common import hbt_group, read_json
from tools.ontology_probe.validate_canonical_map import (
    DEFAULT_CANONICAL_MAP_PATH,
    EXPECTED_CLASS_COUNTS,
    _check_provenance,
)

LEVELS = ("L1_noise", "L2_entail")


def _predicate_names() -> list[str]:
    return read_json(DEFAULT_CANONICAL_MAP_PATH)["_meta"]["predicate_names"]


def _name_to_id() -> dict[str, int]:
    return {name: i for i, name in enumerate(_predicate_names()) if i > 0}


class CanonicalMapLoadTest(unittest.TestCase):
    def test_both_levels_load_with_expected_class_counts(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)
                self.assertEqual(cmap.n_classes, EXPECTED_CLASS_COUNTS[level])
                self.assertEqual(len(cmap.class_names), cmap.n_classes)

    def test_id_map_covers_every_fine_predicate_exactly_once(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)
                self.assertEqual(cmap.fine_ids, list(range(1, 51)))

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(CanonicalMapError):
            load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L3_nonsense")

    def test_validate_reports_no_errors_for_the_frozen_map(self) -> None:
        names = _predicate_names()
        for level in LEVELS:
            with self.subTest(level=level):
                cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)
                errors, _ = validate_canonical_map(
                    cmap, names, EXPECTED_CLASS_COUNTS[level]
                )
                self.assertEqual(errors, [])

    def test_expected_class_count_mismatch_is_an_error(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L1_noise")
        errors, _ = validate_canonical_map(cmap, _predicate_names(), 999)
        self.assertTrue(any("expected 999" in e for e in errors))


class MergeSemanticsTest(unittest.TestCase):
    def test_l1_merges_only_within_lemma_variants(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L1_noise")
        ids = _name_to_id()
        lying = cmap.class_of_predicate(ids["lying on"])
        self.assertEqual(set(lying), {ids["lying on"], ids["laying on"]})
        self.assertEqual(cmap.kinds[ids["laying on"]], KIND_NOISE)

        wearing = cmap.class_of_predicate(ids["wearing"])
        self.assertEqual(set(wearing), {ids["wearing"], ids["wears"]})
        self.assertEqual(cmap.kinds[ids["wears"]], KIND_NOISE)

    def test_l1_does_not_merge_directional_predicates(self) -> None:
        """Merging behind/above/under into `near` would destroy learnable signal."""
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L1_noise")
        ids = _name_to_id()
        for name in ("behind", "above", "under", "over", "near", "in front of"):
            with self.subTest(predicate=name):
                self.assertFalse(cmap.is_merged(ids[name]))

    def test_l2_absorbs_the_frozen_on_family(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        ids = _name_to_id()
        on_class = set(cmap.class_of_predicate(ids["on"]))
        self.assertEqual(
            on_class,
            {
                ids["on"],
                ids["sitting on"],
                ids["standing on"],
                ids["lying on"],
                ids["laying on"],
                ids["walking on"],
                ids["parked on"],
                ids["mounted on"],
            },
        )
        self.assertEqual(cmap.kinds[ids["sitting on"]], KIND_ENTAILMENT)

    def test_l2_is_strictly_coarser_than_l1(self) -> None:
        l1 = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L1_noise")
        l2 = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        self.assertLess(l2.n_classes, l1.n_classes)
        # Every L1 class must be wholly inside a single L2 class.
        for pred_id in range(1, 51):
            with self.subTest(predicate=pred_id):
                self.assertTrue(
                    set(l1.class_of_predicate(pred_id))
                    <= set(l2.class_of_predicate(pred_id))
                )

    def test_covered_in_folds_into_in_at_l2(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        ids = _name_to_id()
        self.assertEqual(
            set(cmap.class_of_predicate(ids["covered in"])),
            {ids["covered in"], ids["in"]},
        )

    def test_untouched_predicates_are_singletons(self) -> None:
        """The identity the touched/untouched decomposition relies on."""
        for level in LEVELS:
            cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)
            with self.subTest(level=level):
                for pred_id in cmap.untouched_fine_ids():
                    self.assertEqual(cmap.class_of_predicate(pred_id), (pred_id,))
                    self.assertEqual(cmap.kinds[pred_id], KIND_SINGLETON)
                # Merged and untouched must partition the 50 predicates.
                self.assertEqual(
                    cmap.merged_fine_ids() | cmap.untouched_fine_ids(),
                    set(range(1, 51)),
                )


class PoolingTest(unittest.TestCase):
    def test_pooling_sums_mass_and_preserves_total(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        ids = _name_to_id()
        fine = [0.0] * 50
        fine[ids["on"] - 1] = 0.25
        fine[ids["sitting on"] - 1] = 0.10
        fine[ids["has"] - 1] = 0.65
        pooled = cmap.pool_probs(fine)
        self.assertAlmostEqual(sum(pooled), 1.0)
        on_class = cmap.remap(ids["on"])
        self.assertAlmostEqual(pooled[on_class], 0.35)
        self.assertAlmostEqual(pooled[cmap.remap(ids["has"])], 0.65)

    def test_pooling_wrong_width_raises(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L1_noise")
        with self.assertRaises(CanonicalMapError):
            cmap.pool_probs([0.5, 0.5])

    def test_assignment_matrix_is_one_hot_with_correct_column_sums(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        matrix = cmap.assignment_matrix()
        self.assertEqual(len(matrix), 50)
        self.assertEqual(len(matrix[0]), cmap.n_classes)
        for row in matrix:
            self.assertEqual(sum(row), 1.0)
        column_sums = [
            sum(matrix[r][c] for r in range(50)) for c in range(cmap.n_classes)
        ]
        self.assertEqual(column_sums, [len(m) for m in cmap.members])
        self.assertEqual(sum(column_sums), 50)

    def test_pooling_matches_counts_rebuilt_in_canonical_space(self) -> None:
        """Unsmoothed pooling of fine counts must equal canonical counts.

        With Dirichlet smoothing the two deliberately diverge -- a prior must be
        rebuilt from canonical counts, not by summing a smoothed fine prior --
        so this asserts the alpha == 0 case only.
        """
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        fine_counts = [0.0] * 50
        fine_counts[10] = 3.0
        fine_counts[30] = 7.0
        pooled = cmap.pool_probs(fine_counts)
        rebuilt = [0.0] * cmap.n_classes
        for pred_id, class_idx in cmap.id_map.items():
            rebuilt[class_idx] += fine_counts[pred_id - 1]
        self.assertEqual(pooled, rebuilt)

    def test_pooled_argmax_can_differ_from_native_for_untouched_predicate(self) -> None:
        """Documented caveat: pooling can outvote a single untouched predicate.

        `has` is untouched at L2, yet a merged `on` class holding more total mass
        than `has` wins the pooled argmax.  This is why native and pooled
        accuracies are both reported rather than assumed equal on the untouched
        subset.
        """
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        ids = _name_to_id()
        fine = [0.0] * 50
        fine[ids["has"] - 1] = 0.30  # native argmax
        fine[ids["on"] - 1] = 0.20
        fine[ids["sitting on"] - 1] = 0.15  # 0.35 combined -> pooled argmax
        self.assertEqual(max(range(50), key=lambda i: fine[i]), ids["has"] - 1)
        pooled = cmap.pool_probs(fine)
        self.assertEqual(
            max(range(cmap.n_classes), key=lambda i: pooled[i]), cmap.remap(ids["on"])
        )
        self.assertIn(ids["has"], cmap.untouched_fine_ids())

    def test_merged_and_singleton_class_names_are_actual_predicates(self) -> None:
        """Canonicalisation must merge existing predicates, not invent vocabulary."""
        names = _predicate_names()
        for level in LEVELS:
            cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)
            for class_idx, name in enumerate(cmap.class_names):
                with self.subTest(level=level, name=name):
                    member_names = {
                        names[p] for p in cmap.members[class_idx]
                    }
                    self.assertIn(name, member_names)


class ValidationTest(unittest.TestCase):
    def _good_map(self, level: str = "L1_noise") -> CanonicalMap:
        return load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, level)

    def test_tampered_id_map_is_caught(self) -> None:
        cmap = self._good_map()
        broken = CanonicalMap(
            level=cmap.level,
            class_names=cmap.class_names,
            id_map={**cmap.id_map, 1: 999},
            kinds=cmap.kinds,
            members=cmap.members,
        )
        errors, _ = validate_canonical_map(broken, _predicate_names())
        self.assertTrue(any("out-of-range" in e for e in errors))

    def test_duplicate_class_membership_is_caught(self) -> None:
        cmap = self._good_map()
        members = list(cmap.members)
        members[0] = tuple(members[0]) + (members[1][0],)
        broken = CanonicalMap(
            level=cmap.level,
            class_names=cmap.class_names,
            id_map=cmap.id_map,
            kinds=cmap.kinds,
            members=tuple(members),
        )
        errors, _ = validate_canonical_map(broken, _predicate_names())
        self.assertTrue(
            any("more than one class" in e or "does not cover" in e for e in errors)
        )

    def test_missing_coverage_is_caught(self) -> None:
        cmap = self._good_map()
        truncated = {k: v for k, v in cmap.id_map.items() if k != 7}
        broken = CanonicalMap(
            level=cmap.level,
            class_names=cmap.class_names,
            id_map=truncated,
            kinds=cmap.kinds,
            members=cmap.members,
        )
        errors, _ = validate_canonical_map(broken, _predicate_names())
        self.assertTrue(any("must cover" in e for e in errors))

    def test_provenance_mismatch_is_an_error(self) -> None:
        from tools.ontology_probe.common import DEFAULT_DATA_ROOT

        payload = copy.deepcopy(read_json(DEFAULT_CANONICAL_MAP_PATH))
        payload["_meta"]["created_from"]["pob_strong_mappings.json"] = "0" * 64
        errors, _ = _check_provenance(payload, DEFAULT_DATA_ROOT)
        self.assertTrue(any("changed since" in e for e in errors))

    def test_entailment_edge_drift_is_an_error(self) -> None:
        from tools.ontology_probe.common import DEFAULT_DATA_ROOT

        payload = copy.deepcopy(read_json(DEFAULT_CANONICAL_MAP_PATH))
        payload["entailment_edges"][0]["parent_id"] = 999
        errors, _ = _check_provenance(payload, DEFAULT_DATA_ROOT)
        self.assertTrue(any("disagree with pob_strong_mappings" in e for e in errors))


class SummaryTest(unittest.TestCase):
    def test_summary_counts_are_consistent(self) -> None:
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        summary = summarise(cmap)
        self.assertEqual(summary["n_classes"], 41)
        self.assertEqual(
            summary["n_predicates_merged"] + summary["n_predicates_untouched"], 50
        )
        self.assertEqual(
            sum(summary["merged_classes"].values()),
            summary["n_predicates_merged"],
        )

    def test_hbt_membership_is_available_for_every_canonical_class(self) -> None:
        """Canonical classes must stay map-able onto the VG strata for reporting."""
        cmap = load_canonical_map(DEFAULT_CANONICAL_MAP_PATH, "L2_entail")
        for member_ids in cmap.members:
            self.assertTrue(all(hbt_group(p) != "unknown" for p in member_ids))


if __name__ == "__main__":
    unittest.main()
