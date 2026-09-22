"""Tests for the ontology-probe statistics layer.

Stdlib only, so this runs in the dependency-free CI `validate` job.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tools.ontology_probe.common import (
    VG_BODY_IDS,
    VG_HEAD_IDS,
    VG_TAIL_IDS,
    hbt_group,
)
from tools.ontology_probe.ontology_stats import (
    conditional_entropy_bits,
    entropy_bits,
    miller_madow_bits,
    predicate_frequencies_payload,
)
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    build_relation_table,
    directed_pair_key,
    undirected_pair_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_index():
    """Two images, four predicates, one deliberately ambiguous pair."""

    class _Index:
        split = "train"
        data_root = Path(".")
        image_ids = [1, 2]
        object_names = {1: "person", 2: "bike", 3: "dog"}
        rel_categories = ["__background__"] + [
            f"p{i}" for i in range(1, 51)
        ]
        ann_labels_by_image = {1: [1, 2, 1], 2: [1, 2]}
        rels_by_image = {
            1: [(0, 1, 1), (0, 2, 1), (2, 1, 2)],
            2: [(0, 1, 2), (0, 1, 3)],
        }

    return _Index()


class EntropyTest(unittest.TestCase):
    def test_entropy_matches_hand_computed_values(self) -> None:
        self.assertEqual(entropy_bits([]), 0.0)
        self.assertEqual(entropy_bits([5]), 0.0)
        self.assertAlmostEqual(entropy_bits([1, 1]), 1.0)
        self.assertAlmostEqual(entropy_bits([1, 1, 1, 1]), 2.0)
        # 3:1 split -> -(0.75*log2 0.75 + 0.25*log2 0.25)
        self.assertAlmostEqual(entropy_bits([3, 1]), 0.8112781244591328)

    def test_entropy_ignores_zero_and_negative_counts(self) -> None:
        self.assertAlmostEqual(entropy_bits([1, 0, 1]), 1.0)
        self.assertAlmostEqual(entropy_bits([3, 0, 1]), 0.8112781244591328)

    def test_miller_madow_is_never_below_plugin(self) -> None:
        for counts in ([1, 1], [3, 1], [5, 4, 3, 2, 1], [10]):
            with self.subTest(counts=counts):
                self.assertGreaterEqual(
                    miller_madow_bits(counts), entropy_bits(counts) - 1e-12
                )

    def test_miller_madow_correction_shrinks_with_sample_size(self) -> None:
        small = miller_madow_bits([1, 1, 1, 1]) - entropy_bits([1, 1, 1, 1])
        large = miller_madow_bits([100, 100, 100, 100]) - entropy_bits(
            [100, 100, 100, 100]
        )
        self.assertGreater(small, large)
        self.assertGreater(large, 0.0)

    def test_conditional_entropy_of_two_certain_groups_is_zero(self) -> None:
        groups = {0: {10: 5}, 1: {11: 5}}
        plugin, corrected = conditional_entropy_bits(groups)
        self.assertAlmostEqual(plugin, 0.0)
        self.assertGreaterEqual(corrected, 0.0)

    def test_conditional_entropy_weights_groups_by_size(self) -> None:
        # Group A: 8 samples, certain (H=0). Group B: 2 samples, 50/50 (H=1).
        groups = {0: {10: 8}, 1: {11: 1, 12: 1}}
        plugin, _ = conditional_entropy_bits(groups)
        self.assertAlmostEqual(plugin, 0.2 * 1.0)

    def test_conditional_entropy_equals_entropy_when_single_group(self) -> None:
        groups = {0: {10: 3, 11: 1}}
        plugin, _ = conditional_entropy_bits(groups)
        self.assertAlmostEqual(plugin, entropy_bits([3, 1]))

    def test_empty_groups_is_zero(self) -> None:
        self.assertEqual(conditional_entropy_bits({}), (0.0, 0.0))


class PairKeyTest(unittest.TestCase):
    def test_directed_pair_key_distinguishes_order(self) -> None:
        self.assertNotEqual(directed_pair_key(1, 2), directed_pair_key(2, 1))

    def test_undirected_pair_key_collapses_order(self) -> None:
        self.assertEqual(undirected_pair_key(1, 2), undirected_pair_key(2, 1))

    def test_pair_keys_are_injective(self) -> None:
        seen = set()
        for a in range(1, 20):
            for b in range(1, 20):
                key = directed_pair_key(a, b)
                self.assertNotIn(key, seen)
                seen.add(key)


class RelationTableTest(unittest.TestCase):
    def test_build_relation_table_uses_positional_indices(self) -> None:
        table = build_relation_table(_tiny_index())
        self.assertEqual(len(table), 5)
        # Image 1 has labels [1, 2, 1] and relations [(0,1,1), (0,2,1), (2,1,2)].
        # Row 0 (0,1,1): c_s = labels[0] = 1, c_o = labels[1] = 2.
        self.assertEqual((table.c_s[0], table.c_o[0], table.pred_id[0]), (1, 2, 1))
        # Row 1 (0,2,1): c_s = labels[0] = 1, c_o = labels[2] = 1 -- note the
        # positional index 2 selects label 1, not label 2.
        self.assertEqual((table.c_s[1], table.c_o[1], table.pred_id[1]), (1, 1, 1))
        self.assertEqual(table.pred_id[2], 2)

    def test_pair_views_are_memoised_and_consistent(self) -> None:
        table = build_relation_table(_tiny_index())
        self.assertIs(table.directed_pairs(), table.directed_pairs())
        self.assertIs(table.undirected_pairs(), table.undirected_pairs())
        self.assertEqual(len(table.directed_pairs()), len(table))

    def test_pair_to_predicates_finds_ambiguous_pairs(self) -> None:
        table = build_relation_table(_tiny_index())
        mapping = table.pair_to_predicates()
        # (person, bike) carries predicates 1, 2 and 3 across the fixture.
        self.assertEqual(mapping[undirected_pair_key(1, 2)], {1, 2, 3})

    def test_relation_key_round_trips(self) -> None:
        table = RelationTable(
            image_id=[7], sub_idx=[0], obj_idx=[1], pred_id=[4], c_s=[1], c_o=[2]
        )
        self.assertEqual(table.relation_key(0), (7, 0, 1, 4))

    def test_subset_preserves_order_and_values(self) -> None:
        table = build_relation_table(_tiny_index())
        sub = table.subset([4, 0])
        self.assertEqual(sub.pred_id, [table.pred_id[4], table.pred_id[0]])
        self.assertEqual(sub.image_id, [table.image_id[4], table.image_id[0]])


class PredicateFrequenciesSchemaTest(unittest.TestCase):
    """The payload must satisfy the five existing consumers of this file."""

    def test_schema_matches_consumer_expectations(self) -> None:
        payload = predicate_frequencies_payload(
            build_relation_table(_tiny_index()), _tiny_index()
        )
        self.assertIn("predicate_frequencies", payload)
        self.assertIn("predicate_names", payload)
        # rel_categories has 51 entries (background + 50) and index == id.
        self.assertEqual(len(payload["predicate_names"]), 51)
        self.assertEqual(payload["predicate_names"][0], "__background__")
        # Ids 1..50 exactly, and background omitted so it cannot consume a
        # head/body/tail slot in compute_head_body_tail_mr.
        self.assertEqual(
            sorted(int(k) for k in payload["predicate_frequencies"]),
            list(range(1, 51)),
        )
        self.assertNotIn("0", payload["predicate_frequencies"])
        for key, value in payload["predicate_frequencies"].items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, int)


class FrequencyStratumTest(unittest.TestCase):
    """Guard against drift between the local H/B/T constants and ra_sgg.py."""

    def test_hbt_constants_match_ra_sgg(self) -> None:
        source = (REPO_ROOT / "src" / "models" / "ra_sgg.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        found: dict[str, list[int]] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "VG_HEAD_IDS",
                "VG_BODY_IDS",
                "VG_TAIL_IDS",
            }:
                found[target.id] = ast.literal_eval(node.value)
        self.assertEqual(set(found), {"VG_HEAD_IDS", "VG_BODY_IDS", "VG_TAIL_IDS"})
        self.assertEqual(tuple(found["VG_HEAD_IDS"]), VG_HEAD_IDS)
        self.assertEqual(tuple(found["VG_BODY_IDS"]), VG_BODY_IDS)
        self.assertEqual(tuple(found["VG_TAIL_IDS"]), VG_TAIL_IDS)

    def test_hbt_partitions_all_fifty_predicates(self) -> None:
        combined = set(VG_HEAD_IDS) | set(VG_BODY_IDS) | set(VG_TAIL_IDS)
        self.assertEqual(len(VG_HEAD_IDS) + len(VG_BODY_IDS) + len(VG_TAIL_IDS), 50)
        self.assertEqual(combined, set(range(1, 51)))
        self.assertEqual(hbt_group(31), "head")   # on
        self.assertEqual(hbt_group(2), "tail")    # across
        self.assertEqual(hbt_group(0), "unknown")


class IndexValidationTest(unittest.TestCase):
    def test_validation_passes_on_the_tracked_sample_fixture(self) -> None:
        from tools.ontology_probe.common import SAMPLE_ROOT
        from tools.ontology_probe.vg_annotations import (
            load_vg_index,
            validate_index_semantics,
        )

        if not (SAMPLE_ROOT / "rel.json").exists():
            self.skipTest("data/VisualGenome_sample fixture not present")
        index = load_vg_index(SAMPLE_ROOT, "train")
        report = validate_index_semantics(index, n_images=50, seed=0)
        self.assertTrue(report["passed"], report["problems_sample"])
        self.assertGreater(report["relations_checked"], 0)

    def test_validation_detects_out_of_range_index(self) -> None:
        from tools.ontology_probe.vg_annotations import validate_index_semantics

        index = _tiny_index()
        index.rels_by_image[1] = [(0, 99, 1)]  # 99 is outside image 1's 3 boxes
        report = validate_index_semantics(index, n_images=10, seed=0)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("outside annotation list" in p["reason"] for p in report["problems_sample"])
        )


if __name__ == "__main__":
    unittest.main()
