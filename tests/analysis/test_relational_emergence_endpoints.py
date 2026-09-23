"""Tests for the Phase IB endpoints and the partitions they are measured across.

The endpoints are the phase's numbers, so they are tested the way a measurement
device is: against cases whose answer is known, in both directions. An evaluator
that has never been shown something it must reject cannot certify anything, and
the plan's own M2 milestone says the same thing in its own words.
"""

from __future__ import annotations

import math
import unittest

from tools.relational_emergence.audits.leakage import apply_standard, standardize
from tools.relational_emergence.eval import design
from tools.relational_emergence.eval.endpoints import ccgp, role_equivariance, synthetic_latents
from tools.relational_emergence.eval.linalg import matmul, orthogonal_procrustes, svd, transpose
from tools.relational_emergence.eval.transplant import (
    cosine,
    flatten,
    paired_transplant,
    sensitivity,
    trajectory_error,
    trajectory_scale,
)


class TestLinearAlgebra(unittest.TestCase):
    def test_orthogonal_procrustes_recovers_a_known_rotation(self) -> None:
        source = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (-1.0, 3.0)]
        angle = 0.7
        rotation = [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
        target = matmul(source, rotation)
        fitted = orthogonal_procrustes(source, target)
        for row_fitted, row_true in zip(fitted, rotation, strict=True):
            for a, b in zip(row_fitted, row_true, strict=True):
                self.assertAlmostEqual(a, b, places=6)

    def test_fitted_transform_is_orthogonal(self) -> None:
        source = [(1.0, 2.0, 0.5), (0.0, 1.0, -1.0), (3.0, -2.0, 1.0), (1.0, 1.0, 1.0)]
        target = [(0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (-1.0, 2.0, 0.0), (2.0, 1.0, -1.0)]
        fitted = orthogonal_procrustes(source, target)
        product = matmul(transpose(fitted), fitted)
        for i, row in enumerate(product):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, places=6)

    def test_svd_reconstructs_its_input(self) -> None:
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        u, singular, vt = svd(matrix)
        rebuilt = matmul(u, [[singular[i] if i == j else 0.0 for j in range(len(singular))] for i in range(len(singular))])
        rebuilt = matmul(rebuilt, vt)
        for row_rebuilt, row_true in zip(rebuilt, matrix, strict=True):
            for a, b in zip(row_rebuilt, row_true, strict=True):
                self.assertAlmostEqual(a, b, places=6)


class TestStandardisation(unittest.TestCase):
    def test_test_set_is_scaled_by_train_statistics(self) -> None:
        """The bug this guards against: scaling each side by its own moments."""
        train = [(0.0, 0.0), (2.0, 4.0), (4.0, 8.0)]
        test = [(100.0, 100.0), (200.0, 300.0)]
        _, means, scales = standardize(train)
        applied = apply_standard(test, means, scales)
        own, _, _ = standardize(test)
        self.assertNotAlmostEqual(applied[0][0], own[0][0], places=3)

    def test_apply_standard_matches_standardize_on_the_same_rows(self) -> None:
        rows = [(1.0, 5.0), (3.0, 7.0), (9.0, 2.0)]
        scaled, means, scales = standardize(rows)
        again = apply_standard(rows, means, scales)
        for left, right in zip(scaled, again, strict=True):
            for a, b in zip(left, right, strict=True):
                self.assertAlmostEqual(a, b, places=9)


class TestCCGP(unittest.TestCase):
    def setUp(self) -> None:
        self.mechanisms = ["containment", "support", "attachment", "free"] * 12
        # Deliberately *not* `index % 4`: that would tie each mechanism to one
        # family, and then no latent could generalize -- the readout would be
        # learning a condition-specific rule and the control would pass while the
        # real case failed. The two axes have to be crossed, as they are in the
        # dataset itself.
        self.conditions = [
            f"family{(index // 4) % 4}|rich" for index in range(len(self.mechanisms))
        ]
        self.chance = {condition: 0.25 for condition in set(self.conditions)}

    def _partition(self) -> design.ConditionPartition:
        return design.ConditionPartition(
            train=frozenset({"family0|rich", "family1|rich"}),
            test=frozenset({"family2|rich", "family3|rich"}),
        )

    def test_mechanism_carrying_latents_generalize(self) -> None:
        partition = self._partition()
        latents = synthetic_latents(
            self.mechanisms, self.conditions, 16, 0, mechanism_weight=3.0, condition_weight=0.0
        )
        result = ccgp(
            latents,
            self.mechanisms,
            self.conditions,
            set(partition.train),
            set(partition.test),
            self.chance,
        )
        self.assertGreater(result.above_baseline, 0.5)

    def test_condition_only_latents_do_not(self) -> None:
        """The control: a code that carries only the condition must not generalize."""
        partition = self._partition()
        latents = synthetic_latents(
            self.mechanisms, self.conditions, 16, 0, mechanism_weight=0.0, condition_weight=3.0
        )
        result = ccgp(
            latents,
            self.mechanisms,
            self.conditions,
            set(partition.train),
            set(partition.test),
            self.chance,
        )
        self.assertLess(result.above_baseline, 0.15)

    def test_overlapping_conditions_are_refused(self) -> None:
        latents = synthetic_latents(self.mechanisms, self.conditions, 4, 0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            ccgp(
                latents,
                self.mechanisms,
                self.conditions,
                {"family0|rich"},
                {"family0|rich"},
                self.chance,
            )


class TestRoleEquivariance(unittest.TestCase):
    def test_recovers_a_known_swap(self) -> None:
        forward = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        reverse = [(2.0, 0.0), (0.0, 3.0), (1.0, 2.0), (4.0, 1.0)]
        families = ["a", "a", "b", "b"]
        result = role_equivariance(forward, reverse, families, {"a"}, {"b"})
        self.assertLess(result.e_role, result.e_role_shuffled)
        self.assertGreater(result.e_role_shuffled - result.e_role, 0.0)

    def test_unrelated_pairs_give_no_gain(self) -> None:
        forward = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        reverse = [(5.0, -3.0), (-2.0, 4.0), (3.0, 3.0), (-1.0, -1.0)]
        families = ["a", "a", "b", "b"]
        result = role_equivariance(forward, reverse, families, {"a"}, {"b"})
        self.assertLess(abs(result.e_role_shuffled - result.e_role), 1.0)

    def test_overlapping_families_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            role_equivariance(
                [(1.0, 0.0)] * 4, [(1.0, 0.0)] * 4, ["a"] * 4, {"a"}, {"a"}
            )


class TestTrajectoryScale(unittest.TestCase):
    def test_a_dead_dimension_is_not_weighted_by_dust(self) -> None:
        """The bug this guards against: a constant coordinate dominating the error."""
        vectors = [
            (float(index), 7.0, 0.5 * index)
            for index in range(20)
        ]
        scale = trajectory_scale(vectors)
        # Dimension 1 never moves. With a floor taken against each dimension's own
        # spread its weight would be the reciprocal of a floating-point residual
        # -- around 1e9 -- and it would dominate every distance. The property that
        # rules that out is a *bounded* ratio between the heaviest and lightest
        # dimension, which the relative floor makes about 1 / relative_floor.
        self.assertGreater(scale[1], 0.0)
        self.assertLess(max(scale) / min(scale), 1.0 / 1e-2 + 1.0)

    def test_scale_tracks_spread(self) -> None:
        wide = [(float(index), 0.001 * index) for index in range(50)]
        scale = trajectory_scale(wide)
        self.assertGreater(scale[0], scale[1])


class TestTrajectoryError(unittest.TestCase):
    def test_identical_trajectories_score_zero(self) -> None:
        trajectory = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(trajectory_error(trajectory, trajectory, [1.0, 1.0]), 0.0)

    def test_the_cap_bounds_a_diverged_rollout(self) -> None:
        truth = [[0.0, 0.0], [0.0, 0.0]]
        wild = [[1000.0, -1000.0], [1000.0, -1000.0]]
        capped = trajectory_error(wild, truth, [1.0, 1.0], cap=5.0)
        uncapped = trajectory_error(wild, truth, [1.0, 1.0], cap=None)
        self.assertAlmostEqual(capped, 5.0, places=9)
        self.assertGreater(uncapped, 100.0)

    def test_length_mismatch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            trajectory_error([[1.0]], [[1.0], [2.0]], [1.0])


class TestTransplantAggregation(unittest.TestCase):
    def test_a_planted_effect_is_detected(self) -> None:
        """A code that helps must show a positive gain with an interval above zero."""
        arm_errors = {"self": [], "correct": [], "wrong": []}
        clusters = []
        for cluster in range(30):
            for _ in range(3):
                correction = 0.4 + 0.05 * ((cluster % 5) - 2)
                arm_errors["correct"].append(1.0)
                arm_errors["wrong"].append(1.0 + correction)
                arm_errors["self"].append(0.9)
                clusters.append(f"c{cluster}")
        result = paired_transplant(arm_errors, clusters, seed=0, resamples=400)
        self.assertGreater(result.gain, 0.0)
        self.assertTrue(result.positive)
        self.assertLess(result.mde, result.gain)

    def test_an_absent_effect_is_not_detected(self) -> None:
        arm_errors = {"self": [], "correct": [], "wrong": []}
        clusters = []
        for cluster in range(30):
            for offset in range(3):
                value = 1.0 + 0.02 * ((cluster + offset) % 7)
                for arm in ("self", "correct", "wrong"):
                    arm_errors[arm].append(value)
                clusters.append(f"c{cluster}")
        result = paired_transplant(arm_errors, clusters, seed=0, resamples=400)
        self.assertFalse(result.positive)

    def test_unaligned_arms_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            paired_transplant(
                {"self": [1.0], "correct": [1.0], "wrong": [1.0, 2.0]}, ["a", "b"]
            )


class TestSensitivity(unittest.TestCase):
    def test_alignment_is_scale_free(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [100.0, 0.0]), 1.0, places=9)
        self.assertAlmostEqual(cosine([1.0, 0.0], [-3.0, 0.0]), -1.0, places=9)

    def test_a_zero_vector_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            cosine([0.0, 0.0], [1.0, 1.0])

    def test_flatten_preserves_order(self) -> None:
        self.assertEqual(flatten([[1.0, 2.0], [3.0, 4.0]]), [1.0, 2.0, 3.0, 4.0])

    def test_an_aligned_code_is_detected_against_zero(self) -> None:
        alignment = [0.6 + 0.02 * (index % 4) for index in range(60)]
        control = [0.05 - 0.01 * (index % 3) for index in range(60)]
        clusters = [f"c{index // 2}" for index in range(60)]
        result = sensitivity(alignment, control, clusters, seed=0, resamples=400)
        self.assertTrue(result.positive)
        self.assertFalse(result.confounded)

    def test_no_alignment_is_not_detected(self) -> None:
        """A code unrelated to the law: the angle concentrates at zero."""
        alignment = [0.01 * ((index % 7) - 3) for index in range(60)]
        control = [0.01 * ((index % 5) - 2) for index in range(60)]
        clusters = [f"c{index // 2}" for index in range(60)]
        result = sensitivity(alignment, control, clusters, seed=0, resamples=400)
        self.assertFalse(result.positive)

    def test_a_control_inside_the_interval_is_flagged(self) -> None:
        """The confound check: a control as aligned as the pairing says nothing.

        Both differences in this endpoint share the dominant direction of the
        trajectory, so an arbitrary pairing can be as aligned as the matched one.
        The statistic must say so rather than report a positive result it cannot
        support.
        """
        values = [0.6 + 0.02 * (index % 4) for index in range(60)]
        clusters = [f"c{index // 2}" for index in range(60)]
        result = sensitivity(values, values, clusters, seed=0, resamples=400)
        self.assertTrue(result.confounded)


class TestDesign(unittest.TestCase):
    def test_context_partition_is_disjoint(self) -> None:
        partition = design.partition_contexts(
            tuple(f"k{index}" for index in range(20)),
            ("t1", "t2"),
            seed=0,
        )
        self.assertFalse(set(partition.fit) & set(partition.validation))
        self.assertFalse(set(partition.seen) & set(partition.test))
        self.assertEqual(len(partition.fit) + len(partition.validation), 20)

    def test_validation_slice_comes_out_of_training(self) -> None:
        with self.assertRaises(ValueError):
            design.partition_contexts(("only",), ("t",), seed=0)

    def test_condition_partition_holds_out_families(self) -> None:
        rows = [
            {"appearance_family": f"family{index % 4}", "regime": "rich", "mechanism": "free"}
            for index in range(40)
        ]
        partition = design.family_condition_partition(rows, seed=0)
        self.assertFalse(partition.train & partition.test)
        train_families = {design.family_of_condition(c) for c in partition.train}
        test_families = {design.family_of_condition(c) for c in partition.test}
        self.assertFalse(train_families & test_families)

    def test_chance_level_is_the_mean_of_reciprocals(self) -> None:
        """A mixed condition must not be rounded to the reciprocal of its mean.

        Three-mechanism tuples mixed with four-mechanism ones give
        ``(1/3 + 1/4) / 2 = 0.2917``; rounding the mean count to 4 would report
        0.25 and understate chance on every such condition.
        """
        rows = [
            {"appearance_family": "f", "regime": "rich", "mechanism": "free", "compatible": ["free", "support", "attachment"]},
            {"appearance_family": "f", "regime": "rich", "mechanism": "free", "compatible": ["free", "support", "attachment", "containment"]},
        ]
        level = design.chance_level_by_condition(rows)
        self.assertAlmostEqual(level["f|rich"], (1 / 3 + 1 / 4) / 2, places=9)

    def test_subsample_is_per_class(self) -> None:
        indices = list(range(20))
        labels = [index % 2 for index in indices]
        chosen = design.subsample_labels(indices, labels, 3, seed=0)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(sum(1 for i in chosen if labels[i] == 0), 3)
        self.assertEqual(len(set(chosen)), len(chosen))

    def test_transplant_plan_needs_a_different_filler(self) -> None:
        trials = design.transplant_plan(
            targets=["g1"],
            sources=["g1", "g2", "g3"],
            regimes_by_key={"g1": ["rich"], "g2": ["rich"], "g3": ["rich"]},
            mechanisms_by_key={
                "g1": ["free", "support"],
                "g2": ["free", "support"],
                "g3": ["free", "support"],
            },
            tuple_by_key={"g1": "T", "g2": "T", "g3": "T"},
            # g2 shares g1's filler; only g3 is a legitimate source.
            filler_by_key={"g1": 7, "g2": 7, "g3": 9},
        )
        self.assertTrue(trials)
        self.assertTrue(all(trial.source_key == "g3" for trial in trials))

    def test_transplant_plan_lists_every_alternative_law(self) -> None:
        trials = design.transplant_plan(
            targets=["g1"],
            sources=["g2"],
            regimes_by_key={"g1": ["rich"], "g2": ["rich"]},
            mechanisms_by_key={"g1": ["free", "support", "attachment"], "g2": ["free", "support", "attachment"]},
            tuple_by_key={"g1": "T", "g2": "T"},
            filler_by_key={"g1": 1, "g2": 2},
        )
        self.assertEqual(len(trials), 3)
        for trial in trials:
            self.assertNotIn(trial.target_mechanism, trial.wrong_mechanisms)
            self.assertEqual(len(trial.wrong_mechanisms), 2)

    def test_transplant_plan_refuses_contexts_without_episodes(self) -> None:
        with self.assertRaises(ValueError):
            design.transplant_plan(
                targets=["g1"],
                sources=["g1"],
                regimes_by_key={"g1": ["rich"]},
                mechanisms_by_key={"g1": ["free", "support"]},
                tuple_by_key={},
                filler_by_key={},
            )


if __name__ == "__main__":
    unittest.main()
