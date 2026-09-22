"""Tests for pair-OOD split construction, its audit, and its reproducibility.

Stdlib only, so this runs in the dependency-free CI `validate` job.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from tools.ontology_probe.build_splits import (
    class_support,
    excluded_classes,
    identify_degenerate_predicates,
    relation_order_sha256,
)
from tools.ontology_probe.canonical_map import identity_map, load_canonical_map
from tools.ontology_probe.ood_split import (
    OODSplitConfig,
    SplitBundle,
    audit_split,
    build_pair_known_split,
    build_pair_ood_split,
    compute_unseen_subset,
    greedy_holdout_pairs,
    holdout_signature,
)
from tools.ontology_probe.vg_annotations import RelationTable, build_relation_table

REPO_ROOT = Path(__file__).resolve().parents[2]

PREDICATE_NAMES = ["__background__"] + [f"p{i}" for i in range(1, 51)]


class _Index:
    """Duck-typed stand-in for VGIndex; the split code only reads these fields."""

    split = "train"
    data_root = Path(".")
    object_names: dict[int, str] = {}
    rel_categories = PREDICATE_NAMES
    image_ids: list[int] = []
    image_file_names: dict[int, str] = {}
    image_sizes: dict[int, tuple[int, int]] = {}


def make_table(rows: list[tuple[int, int, int, int, int, int]]) -> RelationTable:
    """rows are (image_id, sub_idx, obj_idx, pred_id, c_s, c_o)."""
    table = RelationTable()
    for image_id, sub_idx, obj_idx, pred_id, c_s, c_o in rows:
        table.image_id.append(image_id)
        table.sub_idx.append(sub_idx)
        table.obj_idx.append(obj_idx)
        table.pred_id.append(pred_id)
        table.c_s.append(c_s)
        table.c_o.append(c_o)
    return table


def synthetic_table() -> RelationTable:
    """10 images; the (1,2) pair is rare and spread across images so it is a
    viable holdout candidate under a 20% image budget."""
    rows = []
    for image in range(1, 11):
        # A common pair (3,4) on every image under predicate 1=above.
        rows.append((image, 0, 1, 1, 3, 4))
        # Rare pair (1,2) under predicate 2=across on images 8 and 9 only.
        if image in (8, 9):
            rows.append((image, 0, 1, 2, 1, 2))
    return make_table(rows)


class HoldoutSignatureTest(unittest.TestCase):
    def test_signature_is_order_independent(self) -> None:
        self.assertEqual(holdout_signature(42, 1, 2), holdout_signature(42, 2, 1))

    def test_signature_depends_on_seed(self) -> None:
        self.assertNotEqual(holdout_signature(42, 1, 2), holdout_signature(43, 1, 2))

    def test_split_is_independent_of_pythonhashseed(self) -> None:
        """Guards the documented trap: hash() is randomised per process."""
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, r"{root}")
            from tools.ontology_probe.ood_split import holdout_signature
            print(",".join(
                holdout_signature(42, i, i + 1) for i in range(30)
            ))
            """
        ).format(root=str(REPO_ROOT))
        outputs = []
        for hashseed in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hashseed}
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            outputs.append(result.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0])


class GreedyHoldoutTest(unittest.TestCase):
    def test_selects_rare_pair_and_respects_image_budget(self) -> None:
        table = synthetic_table()
        config = OODSplitConfig(max_pair_count=20, min_pair_count=2, image_cap_fraction=0.3)
        pairs, images, stats = greedy_holdout_pairs(
            table, PREDICATE_NAMES, study_predicate_ids=[2], config=config
        )
        # (1,2) is the only pair in [2, 20] with a study-predicate relation.
        self.assertEqual(len(pairs), 1)
        self.assertEqual(images, [8, 9])
        self.assertLessEqual(stats["n_committed_images"], stats["image_budget"])

    def test_pairs_outside_the_count_window_are_not_candidates(self) -> None:
        table = synthetic_table()
        # min_pair_count=3 excludes the 2-instance rare pair, max_pair_count=1
        # excludes nothing but leaves no candidates either.
        pairs, _images, stats = greedy_holdout_pairs(
            table,
            PREDICATE_NAMES,
            study_predicate_ids=[2],
            config=OODSplitConfig(min_pair_count=3, max_pair_count=5, image_cap_fraction=0.5),
        )
        self.assertEqual(pairs, [])
        self.assertEqual(stats["n_selected_pairs"], 0)

    def test_deterministic_for_a_fixed_seed(self) -> None:
        table = synthetic_table()
        config = OODSplitConfig(image_cap_fraction=0.3, seed=7)
        first = greedy_holdout_pairs(table, PREDICATE_NAMES, [2], config)[0]
        second = greedy_holdout_pairs(table, PREDICATE_NAMES, [2], config)[0]
        self.assertEqual(first, second)


class AuditTest(unittest.TestCase):
    """The audit must reject a bad bundle from *any* source.

    These build deliberately-broken bundles by hand rather than corrupting a
    freshly-built one: that is the honest threat model, since the audit exists
    to catch a wrong split, not to catch mutation of its own input.
    """

    def _leaky_table(self) -> RelationTable:
        """Rows 4 and 5 share the undirected pair (1,2) but sit in train/eval."""
        return make_table(
            [
                (1, 0, 1, 1, 3, 4),   # 0 train
                (2, 0, 1, 1, 3, 4),   # 1 train
                (3, 0, 1, 1, 3, 4),   # 2 train
                (4, 0, 1, 2, 1, 2),   # 3 train -- pair (1,2)
                (5, 0, 1, 2, 1, 2),   # 4 eval  -- pair (1,2), same predicate
            ]
        )

    def test_clean_split_passes_and_reports_counts(self) -> None:
        table = synthetic_table()
        config = OODSplitConfig(image_cap_fraction=0.3, dev_fraction=0.0, seed=42)
        bundle = build_pair_ood_split(table, _Index(), PREDICATE_NAMES, [2], config)
        audit = audit_split(table, bundle, PREDICATE_NAMES)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["pair_overlap"], 0)
        self.assertEqual(audit["per_predicate_pair_overlap"], {})

    def test_pair_leak_in_reverse_direction_is_detected(self) -> None:
        """A directed-only holdout would leave (c_o, c_s) in train."""
        table = make_table(
            [
                (1, 0, 1, 2, 1, 2),   # 0 train -- (1,2)
                (2, 0, 1, 2, 2, 1),   # 1 eval  -- (2,1), same undirected pair
            ]
        )
        bundle = SplitBundle(split_id="pair_ood", train=[0], dev=[], eval=[1])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("pair leakage", str(ctx.exception))

    def test_per_predicate_pair_leak_is_detected(self) -> None:
        """A leak confined to one predicate must still fail the audit."""
        table = self._leaky_table()
        bundle = SplitBundle(split_id="pair_ood", train=[0, 1, 2, 3], dev=[], eval=[4])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("per-predicate pair leakage", str(ctx.exception))

    def test_image_level_leak_is_detected(self) -> None:
        table = make_table(
            [
                (1, 0, 1, 1, 3, 4),   # 0 train, image 1
                (1, 0, 1, 2, 1, 2),   # 1 eval,  image 1 -- same image
            ]
        )
        bundle = SplitBundle(split_id="pair_ood", train=[0], dev=[], eval=[1])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("image-level leakage", str(ctx.exception))

    def test_row_overlap_is_detected(self) -> None:
        table = synthetic_table()
        bundle = SplitBundle(split_id="pair_ood", train=[0, 1], dev=[], eval=[0])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("both train and eval", str(ctx.exception))

    def test_dev_train_image_leak_is_detected(self) -> None:
        table = make_table(
            [
                (1, 0, 1, 1, 3, 4),   # 0 train, image 1
                (1, 0, 1, 1, 3, 4),   # 1 dev,   image 1
            ]
        )
        bundle = SplitBundle(split_id="pair_ood", train=[0], dev=[1], eval=[])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("images in both train and dev", str(ctx.exception))

    def test_empty_eval_is_detected(self) -> None:
        table = synthetic_table()
        bundle = SplitBundle(split_id="pair_ood", train=[0], dev=[], eval=[])
        with self.assertRaises(AssertionError) as ctx:
            audit_split(table, bundle, PREDICATE_NAMES)
        self.assertIn("eval split is empty", str(ctx.exception))

    def test_pair_known_split_is_exempt_from_pair_disjointness(self) -> None:
        """The relaxed split deliberately keeps pairs visible in train."""
        table = self._leaky_table()
        bundle = SplitBundle(split_id="pair_known", train=[0, 1, 2, 3], dev=[], eval=[4])
        audit = audit_split(table, bundle, PREDICATE_NAMES)
        self.assertTrue(audit["passed"])


class SplitBundleTest(unittest.TestCase):
    def test_eval_pairs_are_globally_unseen_and_c_unseen_covers_eval(self) -> None:
        table = synthetic_table()
        bundle = build_pair_ood_split(
            table,
            _Index(),
            PREDICATE_NAMES,
            [2],
            OODSplitConfig(image_cap_fraction=0.3, dev_fraction=0.0, seed=42),
        )
        train_pairs = {table.undirected_pairs()[r] for r in bundle.train}
        eval_pairs = {table.undirected_pairs()[r] for r in bundle.eval}
        self.assertEqual(train_pairs & eval_pairs, set())
        # Strict pair-OOD: by construction every eval pair is unseen, so
        # C_unseen is the whole eval set. Computed rather than assumed.
        self.assertEqual(sorted(compute_unseen_subset(table, bundle)), sorted(bundle.eval))

    def test_dev_and_train_are_image_disjoint(self) -> None:
        table = synthetic_table()
        bundle = build_pair_ood_split(
            table,
            _Index(),
            PREDICATE_NAMES,
            [2],
            OODSplitConfig(image_cap_fraction=0.3, dev_fraction=0.5, seed=42),
        )
        train_images = {table.image_id[r] for r in bundle.train}
        dev_images = {table.image_id[r] for r in bundle.dev}
        self.assertEqual(train_images & dev_images, set())
        self.assertTrue(dev_images)

    def test_pair_known_split_keeps_pairs_visible_in_train(self) -> None:
        table = synthetic_table()
        bundle = build_pair_known_split(
            table,
            _Index(),
            PREDICATE_NAMES,
            OODSplitConfig(image_cap_fraction=0.3, dev_fraction=0.2, seed=42),
        )
        train_pairs = {table.undirected_pairs()[r] for r in bundle.train}
        for row in bundle.eval:
            self.assertIn(table.undirected_pairs()[row], train_pairs)


class DegeneratePredicateTest(unittest.TestCase):
    """Threshold logic is tested on explicit bundles.

    A 10-row fixture cannot exercise the real thresholds meaningfully -- every
    predicate in it falls below both -- so the split machinery and the
    thresholding are tested separately rather than conflated.
    """

    def test_rare_predicate_is_excluded_with_both_reasons(self) -> None:
        table = synthetic_table()
        bundle = SplitBundle(split_id="pair_ood", train=[0, 1], dev=[], eval=[2])
        excluded = identify_degenerate_predicates(
            table, bundle, PREDICATE_NAMES, min_train=20, min_eval=50
        )
        self.assertIn("p1", excluded)
        self.assertIn("n_train=2 < 20", excluded["p1"])
        self.assertIn("n_eval=1 < 50", excluded["p1"])

    def test_well_supported_predicate_is_not_excluded(self) -> None:
        table = synthetic_table()
        rows = list(range(len(table)))
        bundle = SplitBundle(split_id="pair_ood", train=rows, dev=[], eval=rows)
        excluded = identify_degenerate_predicates(
            table, bundle, PREDICATE_NAMES, min_train=1, min_eval=1
        )
        self.assertEqual(excluded, {})

    def test_train_and_eval_thresholds_are_independent(self) -> None:
        table = synthetic_table()
        bundle = SplitBundle(split_id="pair_ood", train=[0, 1], dev=[], eval=[2])
        # Low eval threshold, high train threshold -> only the train reason fires.
        excluded = identify_degenerate_predicates(
            table, bundle, PREDICATE_NAMES, min_train=50, min_eval=1
        )
        self.assertIn("n_train=2 < 50", excluded["p1"])
        self.assertNotIn("n_eval", excluded["p1"])

    def test_real_split_excludes_the_known_degenerate_predicates(self) -> None:
        """If the splits are built, `flying in` must be excluded as too rare."""
        import json

        from tools.ontology_probe.common import DEFAULT_OUTPUT_ROOT

        artifact = DEFAULT_OUTPUT_ROOT / "splits" / "split_pair_ood.json"
        if not artifact.exists():
            self.skipTest("splits not built in this checkout")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        excluded = payload["pair_policy"]["excluded_predicates"]
        self.assertIn("flying in", excluded)
        self.assertIn("n_train=4 < 20", excluded["flying in"])
        self.assertEqual(payload["_meta"]["status"], "diagnostic_experiment")


class LabelSpaceSupportTest(unittest.TestCase):
    def test_class_support_sums_to_row_count(self) -> None:
        table = synthetic_table()
        rows = list(range(len(table)))
        for cmap in (identity_map(PREDICATE_NAMES), load_canonical_map(level="L2_entail")):
            with self.subTest(space=cmap.level):
                support = class_support(table, rows, cmap)
                self.assertEqual(sum(support.values()), len(rows))

    def test_identity_map_has_all_singleton_classes(self) -> None:
        cmap = identity_map(PREDICATE_NAMES)
        self.assertEqual(cmap.n_classes, 50)
        self.assertEqual(cmap.merged_fine_ids(), set())
        self.assertEqual(len(cmap.untouched_fine_ids()), 50)

    def test_on_family_folds_into_one_class_at_l2(self) -> None:
        """The concrete reason support must be measured per label space."""
        table = make_table(
            [
                (1, 0, 1, 40, 1, 2),  # sitting on
                (2, 0, 1, 41, 1, 2),  # standing on
                (3, 0, 1, 46, 1, 2),  # walking on
            ]
        )
        vg50 = class_support(table, [0, 1, 2], identity_map(PREDICATE_NAMES))
        l2 = class_support(table, [0, 1, 2], load_canonical_map(level="L2_entail"))
        self.assertEqual(sum(vg50.values()), 3)
        self.assertNotIn("on", [k for k, v in vg50.items() if v])
        self.assertEqual(l2["on"], 3)

    def test_excluded_classes_reports_both_thresholds(self) -> None:
        excluded = excluded_classes({"a": 1, "b": 100}, {"a": 1, "b": 100})
        self.assertIn("a", excluded)
        self.assertNotIn("b", excluded)
        self.assertIn("n_train=1 < 20", excluded["a"])
        self.assertIn("n_eval=1 < 50", excluded["a"])


class AlignmentGuardTest(unittest.TestCase):
    def test_relation_order_hash_is_stable_and_order_sensitive(self) -> None:
        table = synthetic_table()
        digest = relation_order_sha256(table)
        self.assertEqual(digest, relation_order_sha256(synthetic_table()))

        shuffled = make_table(
            [
                (
                    table.image_id[i],
                    table.sub_idx[i],
                    table.obj_idx[i],
                    table.pred_id[i],
                    table.c_s[i],
                    table.c_o[i],
                )
                for i in reversed(range(len(table)))
            ]
        )
        self.assertNotEqual(digest, relation_order_sha256(shuffled))

    def test_verify_relation_order_rejects_mismatch(self) -> None:
        from tools.ontology_probe.splits import (
            SplitAlignmentError,
            verify_relation_order,
        )

        payload = {"_meta": {"relation_order_sha256": "0" * 64}}
        with self.assertRaises(SplitAlignmentError):
            verify_relation_order(synthetic_table(), payload)

    def test_real_split_artifact_aligns_with_the_signature(self) -> None:
        """If the checked-in splits exist, they must match build_relation_table."""
        from tools.ontology_probe.common import DEFAULT_OUTPUT_ROOT, SAMPLE_ROOT
        from tools.ontology_probe.vg_annotations import load_vg_index
        from tools.ontology_probe.splits import load_split

        splits_dir = DEFAULT_OUTPUT_ROOT / "splits"
        if not (splits_dir / "split_pair_ood.json").exists():
            self.skipTest("splits not built in this checkout")
        if not (SAMPLE_ROOT / "rel.json").exists():
            self.skipTest("sample fixture not present")
        # Only meaningful when the artifacts were built from the same data root;
        # the loader is what enforces that, so assert it raises on a mismatch.
        sample_table = build_relation_table(load_vg_index(SAMPLE_ROOT, "train"))
        with self.assertRaises(AssertionError):
            load_split("pair_ood", splits_dir, table=sample_table, verify=True)


if __name__ == "__main__":
    unittest.main()
