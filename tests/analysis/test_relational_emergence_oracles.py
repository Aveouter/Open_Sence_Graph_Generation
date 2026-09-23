"""Tests for the Level-2 oracle machinery, restricted to its stdlib-only parts.

The training itself needs torch and is exercised by the torch job; what is tested
here is everything that decides *what* gets trained and *whether the comparison
is fair*, none of which needs a tensor. That split is deliberate: the fairness
properties below are the ones a reviewer wants checked on every commit, and they
would otherwise only ever run where torch is installed.

Stdlib only, so this runs in the dependency-free CI `validate` job. The module
under test imports torch lazily for the same reason.

    PYTHONIOENCODING=utf-8 python -m unittest tests.analysis.test_relational_emergence_oracles -v
"""

from __future__ import annotations

import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path

from tools.relational_emergence.models import oracles, rollout_oracle

FOUR = ("containment", "support", "attachment", "free")
THREE = ("support", "attachment", "free")


def _row(
    group: str,
    mechanism: str,
    compatible: tuple[str, ...],
    regime: str = "passive",
    offset: float = 0.0,
) -> dict:
    """One dataset row, shaped like the frozen artifact's records."""
    return {
        "group": group,
        "tuple": "inside|touching|other",
        "regime": regime,
        "mechanism": mechanism,
        "compatible": list(compatible),
        "frame_angle_rad": 0.0,
        "x0": [offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "filler_structural": [0.3, 0.3, 1.0, 1.2, 0.7, 0.6, 1.0, 16.0],
        "filler_nuisance": [0.1, 0.2, 0.3, 0.0, 3.0, 7.0, 1.0, 42.0],
        "impulses": [
            {"target": "i", "direction": "up", "magnitude": 1.0, "start": 1, "steps": 2}
        ],
        "trajectory": [[float(step) for step in range(8)] for _ in range(4)],
    }


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _rows_for(n_groups: int, compatible: tuple[str, ...] = FOUR) -> list[dict]:
    return [
        _row(f"g{index}", mechanism, compatible, offset=float(index))
        for index in range(n_groups)
        for mechanism in compatible
    ]


def _dataset(tmp: Path, n_groups: int = 6, compatible: tuple[str, ...] = FOUR):
    return oracles.load_oracle_dataset(
        _write(tmp / "dataset.jsonl", _rows_for(n_groups, compatible))
    )


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class DatasetContractTest(TempDirTest):
    """The loader is the boundary between the stdlib layer and the torch layer."""

    def test_shapes_come_from_the_data_not_from_constants(self) -> None:
        # Hard-coded widths would silently accept a dataset whose structural
        # block changed shape, and every downstream index would be wrong.
        dataset = _dataset(self.tmp)
        self.assertEqual(dataset.input_width, 8 + 8 + 8 + (4 * 4) + 4)
        self.assertEqual(dataset.output_width, 4 * 8)
        self.assertEqual(dataset.groups(), tuple(f"g{i}" for i in range(6)))

    def test_a_ragged_dataset_is_rejected(self) -> None:
        rows = _rows_for(1)
        rows[1]["filler_nuisance"] = rows[1]["filler_nuisance"][:-1]
        with self.assertRaises(ValueError):
            oracles.load_oracle_dataset(_write(self.tmp / "ragged.jsonl", rows))

    def test_hash_is_stable_and_content_sensitive(self) -> None:
        path = _write(self.tmp / "dataset.jsonl", _rows_for(1))
        first = oracles.oracle_dataset_sha256(path)
        self.assertEqual(first, oracles.oracle_dataset_sha256(path))
        _write(path, _rows_for(1))
        self.assertEqual(first, oracles.oracle_dataset_sha256(path))
        _write(self.tmp / "dataset.jsonl", _rows_for(2))
        self.assertNotEqual(first, oracles.oracle_dataset_sha256(path))

    def test_every_row_keeps_its_own_compatible_set(self) -> None:
        dataset = _dataset(self.tmp, n_groups=2, compatible=THREE)
        for mechanism, legal in zip(dataset.mechanism, dataset.compatible, strict=True):
            self.assertIn(mechanism, legal)


class DerangementTest(unittest.TestCase):
    """The shuffled control is the null; a bad one turns the gate into a no-op."""

    def test_no_fixed_points_and_in_support(self) -> None:
        # A control label equal to the true one is not a control, and a label
        # outside the group's compatible set is detectable as impossible rather
        # than as wrong -- the model would then be tested on outlier detection.
        rng = random.Random(0)
        for labels in (FOUR, THREE, ("support", "free")):
            mapping = oracles.derangement(list(labels), rng)
            self.assertEqual(set(mapping), set(labels))
            for original, shuffled in mapping.items():
                self.assertNotEqual(original, shuffled)
                self.assertIn(shuffled, labels)

    def test_a_singleton_set_is_refused_rather_than_silently_identity(self) -> None:
        # A one-element set admits no derangement. Returning the identity would
        # hand the model the true label under the name of a control, which is
        # worse than refusing: it would look like a measurement.
        with self.assertRaises(ValueError):
            oracles.derangement(["free"], random.Random(0))

    def test_shuffled_labels_preserve_the_marginal_distribution(self) -> None:
        # If shuffling changed the label marginals, delta_M would partly measure
        # a class-balance shift rather than the information in the M slot.
        with tempfile.TemporaryDirectory() as name:
            dataset = _dataset(Path(name), n_groups=8)
            shuffled = oracles.shuffled_mechanisms(dataset, seed=0)
            self.assertEqual(len(shuffled), dataset.n_rows)
            true_counts = {m: 0 for m in FOUR}
            shuffled_counts = {m: 0 for m in FOUR}
            for mechanism, value in zip(dataset.mechanism, shuffled, strict=True):
                true_counts[mechanism] += 1
                shuffled_counts[value] += 1
            self.assertEqual(true_counts, shuffled_counts)

    def test_the_shuffle_is_one_permutation_per_group(self) -> None:
        # A fresh permutation per row would let the shuffle correlate with the
        # regime; one per group keeps the arms differing only in the label's
        # truth.
        with tempfile.TemporaryDirectory() as name:
            rows = [
                _row(f"g{index}", mechanism, FOUR, regime=regime, offset=float(index))
                for index in range(4)
                for regime in ("passive", "rich")
                for mechanism in FOUR
            ]
            dataset = oracles.load_oracle_dataset(_write(Path(name) / "d.jsonl", rows))
            shuffled = oracles.shuffled_mechanisms(dataset, seed=0)
            seen: dict[tuple[str, str], str] = {}
            for group, mechanism, value in zip(
                dataset.group, dataset.mechanism, shuffled, strict=True
            ):
                key = (group, mechanism)
                self.assertEqual(seen.setdefault(key, value), value)


class SplitTest(TempDirTest):
    def test_groups_never_straddle_the_split(self) -> None:
        # A group's twins differ only in the mechanism, so splitting by row
        # would put near-duplicates on both sides and inflate every number.
        dataset = _dataset(self.tmp, n_groups=10)
        split = oracles.group_split(dataset, val_fraction=0.3, seed=0)
        self.assertEqual(split.overlaps(), ())
        train_groups = {dataset.group[i] for i in split.train}
        val_groups = {dataset.group[i] for i in split.val}
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(len(train_groups) + len(val_groups), 10)

    def test_every_row_lands_somewhere(self) -> None:
        dataset = _dataset(self.tmp, n_groups=7)
        split = oracles.group_split(dataset, val_fraction=0.3, seed=3)
        self.assertEqual(sorted(split.train + split.val), list(range(dataset.n_rows)))


class StandardiserTest(TempDirTest):
    def test_a_constant_column_does_not_divide_by_zero(self) -> None:
        # Real datasets contain them: a passive-only impulse channel is zero at
        # every step, and an unfloored standard deviation would produce inf.
        dataset = _dataset(self.tmp, n_groups=4)
        stats = oracles.standardiser(dataset, list(range(dataset.n_rows)))
        self.assertTrue(all(scale > 0.0 for scale in stats.input_std))
        self.assertTrue(all(scale > 0.0 for scale in stats.target_std))
        self.assertTrue(all(scale == scale for scale in stats.input_std))  # not NaN

    def test_signature_is_stable_for_the_same_training_rows(self) -> None:
        dataset = _dataset(self.tmp, n_groups=4)
        rows = list(range(dataset.n_rows))
        self.assertEqual(
            oracles.standardiser(dataset, rows).signature(),
            oracles.standardiser(dataset, rows).signature(),
        )

    def test_signature_changes_when_the_training_rows_change(self) -> None:
        dataset = _dataset(self.tmp, n_groups=6)
        full = oracles.standardiser(dataset, list(range(dataset.n_rows)))
        part = oracles.standardiser(dataset, list(range(0, dataset.n_rows // 2)))
        self.assertNotEqual(full.signature(), part.signature())


class EstimateTest(unittest.TestCase):
    """The interval decides the verdict, so its width has to be honest."""

    def test_interval_brackets_the_mean(self) -> None:
        result = oracles.estimate([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(result.mean, 3.0)
        self.assertLess(result.lo, result.mean)
        self.assertGreater(result.hi, result.mean)

    def test_interval_widens_as_the_sample_shrinks(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        wide = oracles.estimate(values[:3])
        narrow = oracles.estimate(values)
        self.assertGreater(wide.hi - wide.lo, narrow.hi - narrow.lo)

    def test_interval_uses_the_t_quantile_not_the_normal_one(self) -> None:
        # At five seeds the correct two-sided 95% factor is 2.776, not 1.96.
        # The normal quantile would narrow the interval by about 30%, and the
        # error would point toward passing the gate -- the wrong direction to be
        # wrong in.
        result = oracles.estimate([1.0, -1.0, 1.0, -1.0, 1.0])
        half_width = 0.5 * (result.hi - result.lo)
        self.assertAlmostEqual(half_width / result.se, 2.776, places=2)

    def test_a_single_seed_is_refused(self) -> None:
        # An interval from one seed is not an interval, and reporting one would
        # let a one-seed run masquerade as a measurement.
        with self.assertRaises(ValueError):
            oracles.estimate([1.0])


class RolloutWidthTest(TempDirTest):
    """The rollout mode indexes a flat action block; getting the width wrong
    silently reshapes the dataset rather than raising."""

    def test_per_step_width_is_the_block_over_the_horizon(self) -> None:
        dataset = _dataset(self.tmp, n_groups=2)
        self.assertEqual(
            rollout_oracle.per_step_action_width(dataset), 4
        )

    def test_a_block_that_does_not_divide_is_refused(self) -> None:
        # The alternative is a silent mis-reshape, which is the failure this
        # whole repository treats as its worst class of bug.
        dataset = _dataset(self.tmp, n_groups=2)
        broken = dataclasses.replace(dataset, horizon=dataset.horizon + 1)
        with self.assertRaises(ValueError):
            rollout_oracle.per_step_action_width(broken)


class DetectableEffectTest(unittest.TestCase):
    """A null result has to be readable as *absent* or as *undetectable*."""

    def test_the_floor_grows_with_the_standard_error(self) -> None:
        tight = oracles.estimate([1.0, 1.1, 0.9, 1.05, 0.95])
        loose = oracles.estimate([1.0, 3.0, -1.0, 2.0, 0.0])
        self.assertLess(
            oracles.minimum_detectable_effect(tight),
            oracles.minimum_detectable_effect(loose),
        )

    def test_the_floor_shrinks_as_seeds_are_added(self) -> None:
        few = oracles.estimate([0.0, 1.0, 2.0])
        many = oracles.estimate([0.0, 1.0, 2.0] * 5)
        self.assertLess(
            oracles.minimum_detectable_effect(many),
            oracles.minimum_detectable_effect(few),
        )

    def test_the_floor_uses_the_t_quantile(self) -> None:
        # With five seeds that is 2.776, not 1.96; the normal quantile would
        # understate the floor by about 9% and make an underpowered run look
        # adequate, which is the direction that matters.
        values = [1.0, -1.0, 1.0, -1.0, 1.0]
        result = oracles.estimate(values)
        self.assertAlmostEqual(
            oracles.minimum_detectable_effect(result) / result.se, 2.776 + 0.8416, places=3
        )

    def test_a_single_seed_is_refused(self) -> None:
        single = oracles.estimate([1.0, 2.0])
        with self.assertRaises(ValueError):
            oracles.minimum_detectable_effect(
                oracles.Estimate(mean=1.0, lo=0.0, hi=2.0, se=1.0, n=1)
            )
        del single


class Level2ContractTest(unittest.TestCase):
    """What the gate reads, and the conditions under which it may pass."""

    HEALTHY_LEAKAGE = {"__controls_fired__": True, "x0_only": {"significantly_above_baseline": False}}

    def _payload(self, **overrides) -> dict:
        base = {
            "passed": True,
            "delta_M": 0.5,
            "delta_M_ci": [0.2, 0.8],
            "shuffled_vs_base": 0.0,
            "trueM_loss": 1.0,
            "shuffledM_loss": 1.5,
            "base_loss": 1.5,
            "seeds": [0, 1, 2],
            "n_train": 100,
            "n_val": 50,
        }
        base.update(overrides)
        return base

    def test_the_gate_reads_the_keys_the_oracle_writes(self) -> None:
        from tools.relational_emergence.audits import protocol

        decision = protocol.gate_decision(
            self.HEALTHY_LEAKAGE, {}, {"passed": True}, level2=self._payload()
        )
        self.assertEqual(decision["verdict"], "GO_PHASE_IA")
        self.assertTrue(decision["level2_measured"])

    def test_a_ci_containing_zero_fails_the_gate_with_stop(self) -> None:
        # STOP, not STOP_DATA: the data is fine, the learned check is what
        # failed. That distinction is the whole point of the taxonomy.
        from tools.relational_emergence.audits import protocol

        decision = protocol.gate_decision(
            self.HEALTHY_LEAKAGE,
            {},
            {"passed": True},
            level2=self._payload(passed=False, delta_M_ci=[-0.1, 0.4]),
        )
        self.assertEqual(decision["verdict"], "STOP")
        self.assertTrue(any("Level-2" in reason for reason in decision["reasons"]))

    def test_the_reason_quotes_the_numbers_it_was_given(self) -> None:
        from tools.relational_emergence.audits import protocol

        decision = protocol.gate_decision(
            self.HEALTHY_LEAKAGE,
            {},
            {"passed": True},
            level2=self._payload(passed=False, delta_M=-0.25),
        )
        self.assertTrue(any("-0.25" in reason for reason in decision["reasons"]))


if __name__ == "__main__":
    unittest.main()
