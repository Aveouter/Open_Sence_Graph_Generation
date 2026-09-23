"""Tests for the Phase IA relational-emergence simulator and its M1 audits.

Stdlib only -- no torch, no numpy -- so this runs in the dependency-free CI
`validate` job. That is the point rather than a convenience: the simulator is the
instrument every Phase I number is measured with, and an instrument that can only
be checked on a machine carrying the model stack is checked rarely.

Two conventions are load-bearing here.

* The filler pool seed, the group indices and the horizon are fixed, so every
  number below is exactly reproducible. Tolerances are quoted next to the value
  they were measured from; a tolerance recalled from memory instead of measured is
  how a test quietly stops testing anything.
* A failure means either the simulator changed behaviour or the tolerance was
  chosen against a different build. Each test says which property it protects, so
  the reader can tell those two cases apart without re-deriving the measurement.
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import random
import struct
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools.relational_emergence.audits import identifiability as ident
from tools.relational_emergence.audits import leakage as leak
from tools.relational_emergence.audits import protocol
from tools.relational_emergence.simulator import actions as actions_module
from tools.relational_emergence.simulator import roles
from tools.relational_emergence.simulator import splits as SPLITS
from tools.relational_emergence.simulator.actions import (
    REGIMES,
    REGIME_NAMES,
    sample_schedule,
    total_impulse_magnitude,
)
from tools.relational_emergence.simulator.compatibility import compatible_mechanisms
from tools.relational_emergence.simulator.counterfactuals import build_group
from tools.relational_emergence.simulator.factors import S0, enumerate_realizable
from tools.relational_emergence.simulator.fillers import build_filler_pool
from tools.relational_emergence.simulator.geometry import (
    GravityFrame,
    frame_coords,
    half_extents_in_frame,
)
from tools.relational_emergence.simulator.labeling import classify_s0
from tools.relational_emergence.simulator.sampling import sample_for_tuple
from tools.relational_emergence.simulator.state import (
    DEFAULT_CONSTANTS,
    Body,
    WorldState,
)
from tools.relational_emergence.simulator.world import rollout

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIONS_PATH = Path(actions_module.__file__)

HORIZON = 60
# The seed `run_phase1a` builds its filler pool from. Reusing it means a failure
# here describes the run that will actually be cited, not a parallel fixture.
POOL_SEED = 20260923
FILLER_POOL_SIZE = 40

# The pre-registered load-bearing cell (ADR 0007): all three main laws hold `i` on
# the notch floor there, so it is the cell where they are supposed to be
# passively confusable and only separable under excitation.
CELL = S0("inside", "touching", "other")

# The three mechanisms that are supposed to be indistinguishable at `CELL` under
# passive observation, as sorted pair keys.
PASSIVELY_CONFUSABLE_PAIRS = (
    "attachment/containment",
    "attachment/support",
    "containment/support",
)

FILLERS = build_filler_pool(FILLER_POOL_SIZE, random.Random(POOL_SEED))

# Identifiers that would mean `actions.py` had learned about the mechanism. A
# policy that can name the law can condition on it, and `P(A | M) != P(A)`
# invalidates every leakage probe built on top. Docstrings naming these concepts
# are fine -- the check is on the code (see `_module_identifiers`).
FORBIDDEN_IDENTIFIER_TOKENS = (
    "mechanism",
    "compatibility",
    "containment",
    "support",
    "attachment",
    "free",
)


def _module_identifiers(path: Path) -> set[str]:
    """Every name the module's *code* uses: defs, args, imports, attribute access.

    Docstrings are ``ast.Constant`` and so contribute nothing, which is what lets
    a comment explain the constraint without tripping the guard.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
            names.update(alias.name for alias in node.names)
    return {name.lower() for name in names}


def groups_for(s0: S0, count: int, horizon: int = HORIZON) -> list:
    """`count` base contexts for one tuple, fillers round-robined from the pool."""
    return [
        build_group(s0, index, FILLERS[index % len(FILLERS)], horizon)
        for index in range(count)
    ]


def trajectories(group, regime: str, horizon: int = HORIZON) -> dict:
    """The four twin trajectories of one group under one regime."""
    return ident._trajectories(group, regime, horizon, DEFAULT_CONSTANTS)


def trajectory_bytes(trajectory) -> tuple[bytes, ...]:
    """A trajectory as raw IEEE-754 bytes, one blob per step.

    Values are compared, not bits, by ``==``: ``-0.0 == 0.0`` is true in Python,
    so a value comparison would call two runs that differ in the sign of a zero
    identical. A re-run of the data generator must not introduce even that.
    """
    blobs = []
    for state in trajectory:
        vector = state.outcome_vector()
        blobs.append(struct.pack(f"<{len(vector)}d", *vector))
    return tuple(blobs)


def probe_state(index: int) -> WorldState:
    """A distinguishable body pair, for the distance arithmetic."""
    return WorldState(
        i=Body(
            pos=(0.5 * index, -0.25 * index),
            vel=(0.125 * index, 0.0),
            half_extent=(0.3, 0.3),
            mass=1.0,
        ),
        j=Body(
            pos=(0.0, 1.0 + 0.1 * index),
            vel=(0.0, -0.05 * index),
            half_extent=(1.1, 1.2),
            mass=16.0,
        ),
    )


class SamplerLabelRoundTripTest(unittest.TestCase):
    """`sample_for_tuple` and `classify_s0` must agree, in every frame."""

    def test_every_tuple_round_trips_in_every_quarter_turn_frame(self) -> None:
        """Guards a mislabelled row: every number downstream keys on the tuple.

        The sampler places `i` exactly on a predicate boundary -- on the notch
        floor, on a wall's top face, one `GAP` clear of one -- and then classifies
        its own output. If the placement and the predicate disagree, the row is
        labelled `non_touching` while sitting in contact (or the reverse) and the
        oracles are trained on a label the geometry does not support.
        """
        frames = [GravityFrame(math.radians(degrees)) for degrees in (0.0, 90.0, 180.0, 270.0)]
        checked = 0
        for s0 in enumerate_realizable():
            for frame in frames:
                for filler in FILLERS[:3]:
                    with self.subTest(
                        tuple=s0.key(),
                        degrees=round(math.degrees(frame.angle_rad)),
                        filler=filler.index,
                    ):
                        state = sample_for_tuple(
                            s0,
                            filler.structural,
                            frame,
                            random.Random(POOL_SEED + filler.index),
                        )
                        self.assertEqual(
                            classify_s0(state, filler.structural, frame),
                            s0,
                            "sampled X_0 classifies as something other than its own tuple",
                        )
                        checked += 1
        # 6 realizable tuples x 4 frames x 3 fillers. Asserted so that a filter
        # that stops producing samples cannot pass this test by not testing.
        self.assertEqual(checked, len(enumerate_realizable()) * len(frames) * 3)

    def test_a_tuple_that_is_not_placed_is_not_where_the_sampler_puts_this_one(self) -> None:
        """Must-fire control for the round trip: the predicate is not constant.

        A `classify_s0` that always returned `inside|touching|other` would satisfy
        the round trip above at that cell while proving nothing. Lifting the same
        object clear of the floor has to change the observed tuple.
        """
        frame = GravityFrame(0.0)
        filler = FILLERS[0]
        lowered = sample_for_tuple(
            S0("inside", "touching", "other"),
            filler.structural,
            frame,
            random.Random(POOL_SEED),
        )
        lifted = sample_for_tuple(
            S0("inside", "non_touching", "other"),
            filler.structural,
            frame,
            random.Random(POOL_SEED),
        )
        self.assertEqual(classify_s0(lowered, filler.structural, frame).contact, "touching")
        self.assertEqual(classify_s0(lifted, filler.structural, frame).contact, "non_touching")
        self.assertNotEqual(
            classify_s0(lowered, filler.structural, frame),
            classify_s0(lifted, filler.structural, frame),
        )


class CounterfactualTwinTest(unittest.TestCase):
    """A twin set may differ in the mechanism and in nothing else."""

    def setUp(self) -> None:
        self.group = build_group(CELL, 0, FILLERS[0], HORIZON)

    def test_twins_share_one_x0_one_frame_and_one_schedule_per_regime(self) -> None:
        """The identity, not just the equality, of the shared context is checked.

        Equality would survive an audit that rebuilt an equal-but-separate state
        per mechanism; what makes the counterfactual valid is that every twin is
        rolled out from the *same* object, so drift between two copies is
        impossible by construction rather than by luck.
        """
        captured: list[tuple] = []
        real_rollout = ident.rollout

        def spy(*args, **kwargs):
            captured.append(args)
            return real_rollout(*args, **kwargs)

        per_regime: dict[str, list[tuple]] = {}
        with mock.patch.object(ident, "rollout", spy):
            for regime in sorted(self.group.schedules):
                del captured[:]
                trajectories(self.group, regime)
                per_regime[regime] = list(captured)

        self.assertEqual(sorted(per_regime), sorted(REGIME_NAMES))
        for regime, calls in per_regime.items():
            self.assertEqual(len(calls), len(self.group.mechanisms), regime)
            self.assertEqual(
                sorted(call[3] for call in calls), sorted(self.group.mechanisms)
            )
            for call in calls:
                # initial state, schedule, frame -- positional in `rollout`.
                self.assertIs(call[0], self.group.x0, regime)
                self.assertIs(call[1], self.group.schedules[regime], regime)
                self.assertIs(call[4], self.group.frame, regime)

        # Three regimes must be three schedule objects. One schedule reused under
        # three names would make the regime comparison a comparison of nothing.
        self.assertEqual(
            len({id(self.group.schedules[name]) for name in self.group.schedules}),
            len(REGIME_NAMES),
        )

    def test_link_offset_is_the_frozen_x0_offset_and_only_for_attachment(self) -> None:
        """Attachment's link is recorded once at X_0, not recomputed per step.

        A link recomputed from the current state would be a no-op law that can
        never be violated, and `attachment` would then be indistinguishable from
        `free` by construction rather than by measurement.
        """
        frozen = (
            self.group.x0.i.pos[0] - self.group.x0.j.pos[0],
            self.group.x0.i.pos[1] - self.group.x0.j.pos[1],
        )
        self.assertEqual(self.group.link_offset("attachment"), frozen)
        # The object sits in the notch, so the recorded offset is not the trivial
        # zero that would make the check above vacuous. Measured: (0.0, -0.2209).
        self.assertNotEqual(frozen, (0.0, 0.0))
        for mechanism in ("support", "containment", "free"):
            self.assertIsNone(self.group.link_offset(mechanism), mechanism)

    def test_the_shared_x0_is_not_mutated_by_a_rollout(self) -> None:
        """Guards order dependence: twin 2 must not start from twin 1's ending.

        `rollout` returns new states, so the group's X_0 should be untouched.
        Were it mutated, the mechanism rolled out first would be measured against
        a different X_0 than the mechanism rolled out second, and the whole
        counterfactual comparison would collapse.
        """
        before = tuple(self.group.x0.outcome_vector())
        for regime in sorted(self.group.schedules):
            for mechanism in self.group.mechanisms:
                rollout(
                    self.group.x0,
                    self.group.schedules[regime],
                    self.group.filler.structural,
                    mechanism,
                    self.group.frame,
                    self.group.link_offset(mechanism),
                    HORIZON,
                )
        self.assertEqual(tuple(self.group.x0.outcome_vector()), before)
        self.assertEqual(self.group.link_offset("attachment"), (
            before[0] - before[4],
            before[1] - before[5],
        ))

    def test_twins_are_exactly_the_compatible_mechanisms(self) -> None:
        """A group that dropped or added a law would measure a different question."""
        self.assertEqual(
            set(self.group.mechanisms), set(compatible_mechanisms(self.group.s0))
        )


class GroupDeterminismTest(unittest.TestCase):
    """Same inputs must give the same run, to the last bit."""

    def test_rebuilding_a_group_reproduces_every_trajectory_bitwise(self) -> None:
        """Guards a silently unseeded draw anywhere in the group construction.

        A group rebuilt from the same spec has to be the same group, bit for bit:
        otherwise `run_phase1a` is not reproducible from its frozen config, and two
        runs of the "same" experiment could disagree about a verdict.
        """
        first = build_group(CELL, 0, FILLERS[0], HORIZON)
        second = build_group(CELL, 0, FILLERS[0], HORIZON)
        self.assertEqual(first.x0, second.x0)
        self.assertEqual(first.frame, second.frame)
        self.assertEqual(first.filler, second.filler)
        self.assertEqual(first.schedules, second.schedules)

        for regime in sorted(first.schedules):
            for mechanism in first.mechanisms:
                self.assertEqual(
                    trajectory_bytes(trajectories(first, regime)[mechanism]),
                    trajectory_bytes(trajectories(second, regime)[mechanism]),
                    f"{regime}/{mechanism}",
                )

    def test_group_construction_is_independent_of_pythonhashseed(self) -> None:
        """Guards the documented `hash()` trap: it is salted per interpreter.

        `hash_stable` exists so a group's seed does not depend on the process it
        was built in. Two interpreters with different hash salts must therefore
        build the same X_0; this is what the rule is for, and it cannot be checked
        from inside one process.
        """
        script = textwrap.dedent(
            """
            import random, sys
            sys.path.insert(0, r"{root}")
            from tools.relational_emergence.simulator.counterfactuals import (
                build_group,
                hash_stable,
            )
            from tools.relational_emergence.simulator.factors import S0
            from tools.relational_emergence.simulator.fillers import build_filler_pool

            filler = build_filler_pool(1, random.Random(0))[0]
            group = build_group(S0("inside", "touching", "other"), 0, filler, 4)
            print(hash_stable("probe"), repr(group.x0.outcome_vector()))
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


class NoiseTwinTest(unittest.TestCase):
    """The same-mechanism null twin must move only the initial velocity."""

    def test_noise_twin_moves_the_initial_velocity_and_nothing_else(self) -> None:
        """Guards the null floor's meaning: it must measure noise, not a new world.

        The null floor is read as "the same mechanism, independent noise". If the
        re-draw also changed the filler, the frame, the positions or the actions,
        the floor would be an upper bound on everything the sampler varies, and a
        small between-mechanism signal would disappear inside it.
        """
        base = build_group(CELL, 0, FILLERS[0], HORIZON)
        twin = build_group(CELL, 0, FILLERS[0], HORIZON, noise_offset=0x5EED)

        self.assertEqual(twin.filler, base.filler)
        self.assertEqual(twin.frame, base.frame)
        self.assertEqual(twin.schedules, base.schedules)
        self.assertEqual(twin.x0.i.pos, base.x0.i.pos)
        self.assertEqual(twin.x0.j.pos, base.x0.j.pos)
        self.assertEqual(twin.x0.i.half_extent, base.x0.i.half_extent)
        self.assertEqual(twin.x0.j.half_extent, base.x0.j.half_extent)
        self.assertEqual(twin.x0.i.mass, base.x0.i.mass)

        base_velocity = base.x0.i.vel + base.x0.j.vel
        twin_velocity = twin.x0.i.vel + twin.x0.j.vel
        # All four components differ for this construction; one is enough to make
        # the twin a different run, zero would make the null floor identically 0.
        self.assertEqual(sum(1 for a, b in zip(base_velocity, twin_velocity) if a != b), 4)
        self.assertNotEqual(base_velocity, twin_velocity)

    def test_zero_offset_reproduces_the_base_group(self) -> None:
        """The twin must be a re-draw, not a different group: offset 0 is the base."""
        base = build_group(CELL, 0, FILLERS[0], HORIZON)
        same = build_group(CELL, 0, FILLERS[0], HORIZON, noise_offset=0)
        self.assertEqual(same.x0, base.x0)
        self.assertEqual(same.schedules, base.schedules)

    def test_the_twin_actually_diverges(self) -> None:
        """A twin that never diverges would report a null floor of zero.

        The floor is the denominator of every precondition test in the gate, so a
        twin that is silently a duplicate would make every pair look separable.
        """
        base = build_group(CELL, 0, FILLERS[0], HORIZON)
        twin = build_group(CELL, 0, FILLERS[0], HORIZON, noise_offset=0x5EED)
        for regime in sorted(base.schedules):
            base_trajectories = trajectories(base, regime)
            twin_trajectories = trajectories(twin, regime)
            for mechanism in base.mechanisms:
                self.assertNotEqual(
                    trajectory_bytes(base_trajectories[mechanism]),
                    trajectory_bytes(twin_trajectories[mechanism]),
                    f"{regime}/{mechanism}",
                )


class ActionPolicyIndependenceTest(unittest.TestCase):
    """The action policy must know the regime and nothing about the law."""

    def test_sampling_a_schedule_consumes_only_the_policy_and_the_rng(self) -> None:
        """Same seed, same schedule -- for every regime.

        Determinism is what makes "the twins share one schedule" checkable at all;
        a policy that drew from the interpreter's global random state would make
        `P(A | M) = P(A)` unverifiable rather than false.
        """
        for name in REGIME_NAMES:
            first = sample_schedule(REGIMES[name], HORIZON, random.Random(POOL_SEED))
            second = sample_schedule(REGIMES[name], HORIZON, random.Random(POOL_SEED))
            self.assertEqual(first, second, name)
            self.assertEqual(len(first.impulses), REGIMES[name].n_impulses, name)

    def test_the_sampler_signature_admits_no_mechanism(self) -> None:
        """A structural guard: the policy cannot read what it cannot name."""
        self.assertEqual(
            tuple(inspect.signature(sample_schedule).parameters),
            ("policy", "horizon", "rng"),
        )

    def test_the_actions_module_does_not_reference_any_mechanism(self) -> None:
        """Gating an action on the mechanism would invalidate every leakage probe.

        If this fails, `P(A | M) != P(A)` somewhere in the policy, and the whole
        "the leak is in the dynamics, not the actions" argument is gone.
        """
        identifiers = _module_identifiers(ACTIONS_PATH)
        offending = sorted(
            identifier
            for identifier in identifiers
            for token in FORBIDDEN_IDENTIFIER_TOKENS
            if token in identifier
        )
        self.assertEqual(offending, [])
        self.assertNotIn("compatibility", identifiers)


class RegimeOrderingTest(unittest.TestCase):
    """Passive < weak < rich, and passive is exactly nothing."""

    def test_passive_is_zero_impulse_and_the_regimes_are_strictly_ordered(self) -> None:
        """Guards a mislabelled regime: the intervention claim rests on this order.

        If `weak` and `rich` were not ordered, the trend across regimes that
        `gate_decision` uses to choose between "increase excitation" and "merge the
        ontology" would be reading a noise difference. Measured for this seed:
        passive 0.0, weak 0.759, rich 16.4753.
        """
        totals = {
            name: total_impulse_magnitude(
                sample_schedule(REGIMES[name], HORIZON, random.Random(POOL_SEED))
            )
            for name in REGIME_NAMES
        }
        self.assertEqual(totals["passive"], 0.0)
        self.assertLess(totals["passive"], totals["weak"])
        self.assertLess(totals["weak"], totals["rich"])

    def test_passive_carries_no_impulse_at_all(self) -> None:
        """Not "a small impulse": passive observation means the world is left alone."""
        schedule = sample_schedule(REGIMES["passive"], HORIZON, random.Random(POOL_SEED))
        self.assertEqual(schedule.impulses, ())
        for step in range(HORIZON):
            self.assertEqual(
                schedule.impulse_vector("i", step, GravityFrame(0.0)), (0.0, 0.0)
            )


class LoadBearingCellTest(unittest.TestCase):
    """The pre-registered witness of ADR 0007, measured where it is supposed to hold.

    All numbers below are `J_ab` at the end of a 60-step horizon, passive regime,
    tuple `inside|touching|other`, 16 base contexts, computed with the audit's own
    `pair_curves` and its frozen robust scale (recomputed from the same 16
    groups). Measured for this configuration:

    * the three main pairs: containment/support 0.2053, attachment/support
      0.2309, attachment/containment 0.4069
    * the three pairs involving `free`: containment/free 0.4197, free/support
      0.6506, attachment/free 0.8123
    """

    GROUPS = 16

    @classmethod
    def setUpClass(cls) -> None:
        cls.groups = groups_for(CELL, cls.GROUPS)
        cls.scale = ident.robust_scale(
            ident.collect_calibration_vectors(cls.groups, HORIZON, DEFAULT_CONSTANTS)
        )
        cls.curves = ident.pair_curves(cls.groups, cls.scale, HORIZON, DEFAULT_CONSTANTS)[
            CELL.key()
        ]

    def _end_distances(self, regime: str) -> dict[str, float]:
        return {
            entry.pair: entry.mean_curve()[-1]
            for entry in self.curves
            if entry.regime == regime
        }

    def test_the_three_main_laws_are_passively_confusable(self) -> None:
        """The witness is empty unless passive observation cannot tell them apart.

        If the three laws already separated at this cell without any excitation,
        the intervention claim would have nothing to explain: there would be no
        pair that is indistinguishable passively and separable actively.
        """
        distances = self._end_distances("passive")
        self.assertEqual(sorted(distances), sorted(PASSIVELY_CONFUSABLE_PAIRS + (
            "attachment/free",
            "containment/free",
            "free/support",
        )))
        main = sorted(distances[pair] for pair in PASSIVELY_CONFUSABLE_PAIRS)
        free = sorted(
            distances[pair] for pair in distances if pair not in PASSIVELY_CONFUSABLE_PAIRS
        )
        mean_free = sum(free) / len(free)

        # Each main pair is within a small absolute distance of the others.
        # Measured max 0.4069, so 0.5 keeps ~20% headroom.
        self.assertLess(max(main), 0.5)
        # The spread among the three main laws is smaller than their distance to
        # `free`. Measured 0.2016 against a mean free distance of 0.6275.
        self.assertLess(max(main) - min(main), mean_free)
        # No main pair is further apart than the free law's average distance.
        # Measured 0.4069 < 0.6275, the widest margin the cell offers.
        self.assertLess(max(main), mean_free)
        # Even the closest free pair is further apart than the closest main pair.
        # Measured 0.4197 > 0.2053.
        self.assertGreater(min(free), min(main))

    def test_excitation_creates_the_increment_passive_observation_lacks(self) -> None:
        """The other half of the witness: the impulse has to separate them.

        ADR 0007 says the intervention claim is reported as unresolved -- not as
        negative -- if no pair shows a non-empty Passive-to-Rich increment. This
        asserts the increment exists for all three main pairs, which is the
        condition under which the measured matrix can testify at all. Measured:
        containment/support 0.2309 -> 5.6078, attachment/support 0.2053 -> 5.6908,
        attachment/containment 0.4069 -> 1.2086.
        """
        passive = self._end_distances("passive")
        rich = self._end_distances("rich")
        for pair in PASSIVELY_CONFUSABLE_PAIRS:
            with self.subTest(pair=pair):
                self.assertGreater(rich[pair], passive[pair])
                self.assertGreater(rich[pair] - passive[pair], 0.0)


class ArenaBoundTest(unittest.TestCase):
    """The arena bounds are compliances, not projections, and this is the envelope.

    `_resolve_bounds` corrects a fraction (`GAIN_POS = 0.6`) of a violation per
    pass rather than projecting to the wall, which is what keeps the map
    continuous. A body arriving at speed therefore overshoots and is relaxed back
    over the next steps. Measured over 8 groups x 4 mechanisms x 60 rich steps:

    * lateral: worst outer face 4.8343 against `arena_half_t = 4.0` (excess 0.8343)
    * vertical: worst top 6.6166 against `arena_top_u = 6.0` (excess 0.6166)
    * under the passive regime the same measurement is negative at every step
      (excess -2.6931 laterally, -3.8634 vertically): without an impulse the
      nearest approach still leaves 2.69 units of clearance

    The tolerances below are the measured envelope with headroom. A strict bound
    (excess <= 0) does *not* hold in the rich regime, so it is not asserted; what
    is asserted is that the overshoot stays bounded by a small multiple of what
    was measured, which is what fails if the relaxation is weakened.
    """

    GROUPS = 8
    # Floating point only. The projection is exact, so anything above this is a
    # body genuinely outside the world rather than accumulated rounding.
    TOLERANCE = 1e-9

    def _excursions(self, regime: str) -> tuple[float, float]:
        """Worst `(top - arena_top_u, |lateral face| - arena_half_t)` over the rollout."""
        worst_top = -math.inf
        worst_lateral = -math.inf
        constants = DEFAULT_CONSTANTS
        for group in groups_for(CELL, self.GROUPS):
            for mechanism in group.mechanisms:
                trajectory = rollout(
                    group.x0,
                    group.schedules[regime],
                    group.filler.structural,
                    mechanism,
                    group.frame,
                    group.link_offset(mechanism),
                    HORIZON,
                )
                for state in trajectory:
                    for body in (state.i, state.j):
                        along_up, along_lateral = frame_coords(
                            body.pos, (0.0, 0.0), group.frame
                        )
                        half_up, half_lateral = half_extents_in_frame(
                            body.half_extent, group.frame
                        )
                        worst_top = max(worst_top, along_up + half_up - constants.arena_top_u)
                        worst_lateral = max(
                            worst_lateral,
                            abs(along_lateral) + half_lateral - constants.arena_half_t,
                        )
        return worst_top, worst_lateral

    def test_no_body_reaches_a_wall_under_passive_observation(self) -> None:
        """This is the strict bound, and it holds where it should: no excitation.

        Under passive observation the bodies never touch the arena at all. If this
        ever fails, the sampler is placing objects outside the world, or a regime
        labelled passive is applying an impulse.
        """
        worst_top, worst_lateral = self._excursions("passive")
        self.assertLessEqual(worst_top, 0.0)
        self.assertLessEqual(worst_lateral, 0.0)

    def test_rich_excursions_stay_inside_the_walls(self) -> None:
        """The bound must hold under the regime most likely to break it.

        The walls exist so that distance is decided by the law under test rather
        than by which way a body wandered -- unbounded, one impulse drifts a body
        about 150 units. This is the strict invariant, not an envelope: an
        earlier revision relaxed positions back by a fraction of the violation,
        and under rich impulses 26.9% of body-steps then sat outside a wall by up
        to 0.83. A tolerance here would let that return unnoticed, so there is
        none beyond floating point.
        """
        worst_top, worst_lateral = self._excursions("rich")
        self.assertLessEqual(worst_top, self.TOLERANCE)
        self.assertLessEqual(worst_lateral, self.TOLERANCE)


class NumericalFloorTest(unittest.TestCase):
    """`floor_num` is arithmetic noise, not physics."""

    GROUPS = 4

    def test_the_numerical_floor_is_tiny(self) -> None:
        """Guards a discontinuous constraint: a switch makes the floor macroscopic.

        `contact_softness` exists so that a 1e-17 difference cannot flip a contact
        branch and diverge macroscopically. Measured on 4 groups (2880 samples):
        max 2.486e-13, mean 1.843e-15. A switch-like constraint put the max near
        1e-2 or higher, and the floor would then be a large part of any `J_ab`.
        """
        groups = groups_for(CELL, self.GROUPS)
        scale = ident.robust_scale(
            ident.collect_calibration_vectors(groups, HORIZON, DEFAULT_CONSTANTS)
        )
        floor = ident.numerical_floor(groups, scale, HORIZON, DEFAULT_CONSTANTS)
        self.assertEqual(floor["n_samples"], self.GROUPS * 3 * 4 * HORIZON)
        # The pre-registered meaning of "tiny": smaller than any effect the gate
        # would act on, with headroom over the measured 2.486e-13.
        self.assertLess(floor["max"], 0.5)
        self.assertLess(floor["max"], 1e-6)
        self.assertLess(floor["mean"], floor["max"])


class NormalizedDistanceTest(unittest.TestCase):
    """`d_norm` must be scale-invariant, exactly zero on a duplicate, and defined."""

    def test_identical_states_are_exactly_zero_and_scaling_halves_the_distance(self) -> None:
        """Guards the primary scalar: every `J_ab` is this function.

        A distance that were not exactly zero on a shared state (an epsilon floor,
        a missing dimension) would add a constant to every curve and to the null
        floor alike, and the precondition test compares one against the other.
        Doubling the scale must halve, not merely shrink, or the frozen-scale
        convention means something other than a unit change.
        """
        vectors = [probe_state(index).outcome_vector() for index in range(1, 6)]
        scale = ident.robust_scale(vectors)
        first, second = probe_state(1), probe_state(3)

        self.assertEqual(ident.normalized_distance(first, first, scale), 0.0)
        doubled = tuple(2.0 * value for value in scale)
        self.assertAlmostEqual(
            ident.normalized_distance(first, second, doubled),
            ident.normalized_distance(first, second, scale) / 2.0,
            places=12,
        )
        self.assertGreater(ident.normalized_distance(first, second, scale), 0.0)

    def test_robust_scale_is_strictly_positive_even_on_a_constant_dimension(self) -> None:
        """Guards the degenerate-velocity case: a bare MAD is exactly zero there.

        Both bodies at rest for the whole window is the ordinary case, not a
        corner: the velocity dimensions are then constant, their MAD is zero, and
        dividing by it makes every distance infinite. The floors in `robust_scale`
        are what keep the scale defined; this asserts none of them is ever zero.
        """
        resting = [
            (float(index), 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0) for index in range(5)
        ]
        scale = ident.robust_scale(resting)
        self.assertEqual(len(scale), len(resting[0]))
        for index, value in enumerate(scale):
            self.assertGreater(value, 0.0, f"dimension {index} has a non-positive scale")
        # Dimension 3 is constant zero, so only the absolute floor can save it.
        self.assertEqual(scale[3], ident.ABSOLUTE_SCALE_FLOOR)
        # And a distance on that degenerate pair is finite, not a division by zero.
        self.assertEqual(
            ident.normalized_distance(probe_state(1), probe_state(1), scale), 0.0
        )


class LeakageProbeTest(unittest.TestCase):
    """The M0 probe must be able to see a leak before its clean verdict counts."""

    GROUPS = 24
    PERMUTATIONS = 5

    def test_the_control_fires_and_the_real_sampler_does_not(self) -> None:
        """Guards against certifying the sampler with a probe that cannot detect.

        A probe too weak to find a deliberately-planted `X_0 -> M` shift would
        report the real sampler clean and mean nothing by it. Measured on 24
        contexts of the load-bearing tuple: control 0.5625 against a 0.25
        baseline, real sampler 0.0833. Five permutations is deliberately few (the
        smallest attainable p-value is 1/6), so the must-fire condition checked
        here is the accuracy margin, not the permutation p-value that the frozen
        config's 40 permutations supply.
        """
        groups = groups_for(CELL, self.GROUPS)
        records = leak.build_records(groups)
        real = leak.evaluate_feature_set(
            records, "x0_only", n_permutations=self.PERMUTATIONS, seed=7
        )
        control = leak.evaluate_feature_set(
            leak.leaky_sampler_control(groups),
            "x0_only",
            n_permutations=self.PERMUTATIONS,
            seed=7,
        )

        # Same row set, same tuples, same mechanisms: the two accuracies are on
        # one scale only if the baselines agree.
        self.assertEqual(
            real["compatibility_aware_baseline"], control["compatibility_aware_baseline"]
        )
        self.assertEqual(real["compatibility_aware_baseline"], 1.0 / 4.0)
        self.assertGreater(control["accuracy"], control["compatibility_aware_baseline"])
        self.assertLessEqual(real["accuracy"], real["compatibility_aware_baseline"])

    def test_the_control_shifts_x0_and_nothing_else(self) -> None:
        """Must-fire control for the control: the leak has to be in the features.

        If the leaky sampler also changed the labels, the probe's accuracy would
        rise for a reason that has nothing to do with reading a leak out of the
        context, and the must-fire demonstration would be invalid.
        """
        groups = groups_for(CELL, 4)
        records = leak.build_records(groups)
        control = leak.leaky_sampler_control(groups)
        self.assertEqual(
            [(r.tuple_key, r.mechanism, r.filler_vector) for r in records],
            [(r.tuple_key, r.mechanism, r.filler_vector) for r in control],
        )
        shifted = sum(1 for a, b in zip(records, control) if a.x0_vector != b.x0_vector)
        unshifted = sum(
            1 for a, b in zip(records, control) if a.x0_vector[1:] != b.x0_vector[1:]
        )
        self.assertEqual(unshifted, 0)
        self.assertEqual(shifted, len(records) - len(groups))


class HorizonSelectionTest(unittest.TestCase):
    """`H` is read off the measured curve, and never below the floor."""

    @staticmethod
    def _curves(regime: str, curve: tuple[float, ...]) -> dict:
        return {
            CELL.key(): [
                ident.PairCurves(
                    tuple_key=CELL.key(),
                    regime=regime,
                    pair="containment/support",
                    per_group=(curve,),
                )
            ]
        }

    def test_a_saturating_curve_selects_at_least_the_minimum_horizon(self) -> None:
        """Guards a horizon so short that every pair looks separable.

        The raw saturation point of this constructed curve is step 2. Measured
        over 2 steps, an impulse that has not yet acted would look like a law that
        does nothing, so the floor has to win: `selected_horizon` must be at least
        `MIN_HORIZON`, and for this curve exactly it.
        """
        curve = (0.0, 0.0) + (5.0,) * 10
        result = protocol.select_horizon(self._curves("rich", curve), "rich")
        self.assertTrue(result["saturated"])
        self.assertEqual(result["n_curves"], 1)
        # This curve reaches 90% of its final value at step 2, so the measured
        # saturation point is 2 while the reported horizon is the floor.
        self.assertEqual(result["per_pair_saturation_quantile_90"], 2)
        self.assertEqual(result["mean_curve_saturation_horizon"], 2)
        self.assertGreaterEqual(result["selected_horizon"], protocol.MIN_HORIZON)
        self.assertEqual(result["selected_horizon"], protocol.MIN_HORIZON)

    def test_a_flat_zero_curve_is_not_reported_as_saturated(self) -> None:
        """Guards a saturation claim over a curve that never rose.

        `SATURATION_FRACTION * final` is zero when the curve is zero, so every
        step would clear a target of zero and step 0 would be reported as
        saturated. That is the code pretending to have found a divergence point in
        a curve with nothing in it -- the sort of number a report could then quote.
        """
        result = protocol.select_horizon(self._curves("rich", (0.0,) * 12), "rich")
        self.assertFalse(result["saturated"])
        self.assertNotIn("saturation_target", result)
        self.assertEqual(result["final_mean"], 0.0)
        self.assertGreaterEqual(result["selected_horizon"], protocol.MIN_HORIZON)

    def test_a_regime_with_no_curves_is_rejected(self) -> None:
        """Silently selecting over an empty sample would be a chart of nothing."""
        curves = self._curves("passive", (1.0,) * 12)
        with self.assertRaises(ValueError):
            protocol.select_horizon(curves, "rich")


class GateDecisionTest(unittest.TestCase):
    """The gate must be able to say four different things, from hand-built inputs.

    Every input here is constructed rather than measured: the verdict logic is
    what is under test, not the pipeline that feeds it. A gate that can only ever
    return one verdict is not a gate, and the whole point of pre-registering the
    criteria is that it can fail.
    """

    @staticmethod
    def _leakage(**flags: bool) -> dict:
        result = {
            name: {"significantly_above_baseline": flags.get(name, False)}
            for name in leak.FEATURE_SETS
        }
        result["__controls_fired__"] = flags.get("__controls_fired__", True)
        return result

    def test_a_missing_level2_is_incomplete_not_go(self) -> None:
        """`GO_PHASE_IA` requires three checks and level 2 has not run yet.

        Returning GO here would let the stdlib gate certify a delivery whose
        utility check was never measured -- the one comparison the plan requires
        and the one a stdlib-only run cannot make.
        """
        decision = protocol.gate_decision(
            self._leakage(), {}, {"passed": True}, level2=None
        )
        self.assertEqual(decision["verdict"], "INCOMPLETE")
        self.assertFalse(decision["level2_measured"])
        self.assertTrue(decision["reasons"])

    def test_everything_passing_is_go(self) -> None:
        """The other end: all four checks clean must be able to reach GO."""
        decision = protocol.gate_decision(
            self._leakage(),
            {},
            {"passed": True},
            level2={"passed": True, "delta_M": 0.4, "shuffled_vs_base": 0.01},
        )
        self.assertEqual(decision["verdict"], "GO_PHASE_IA")
        self.assertEqual(decision["reasons"], [])
        self.assertTrue(decision["level2_measured"])

    def test_a_probe_above_baseline_blocks_the_gate(self) -> None:
        """A leaking sampler must not reach GO, whatever level 2 says.

        The verdict is `STOP_DATA`, because a leaking sampler is a failure of the
        *data*: no amount of oracle training repairs an initial state that
        reveals the law it is about to obey. `STOP` is reserved for a failure of
        the learned check -- see the taxonomy comment above the verdict cascade
        in `audits/protocol.py`.
        """
        decision = protocol.gate_decision(
            self._leakage(x0_only=True),
            {},
            {"passed": True},
            level2={"passed": True},
        )
        self.assertNotEqual(decision["verdict"], "GO_PHASE_IA")
        self.assertNotEqual(decision["verdict"], "INCOMPLETE")
        self.assertIn("x0_only", decision["reasons"][0])
        self.assertEqual(decision["verdict"], "STOP_DATA")

    def test_the_precondition_failure_is_stop_data(self) -> None:
        """`STOP_DATA` is what ADR 0007 assigns to the null-floor precondition.

        When the same-mechanism floor is comparable to the between-mechanism
        signal, no `J_ab` means anything and no amount of oracle training fixes
        it: the data itself cannot support the question.
        """
        decision = protocol.gate_decision(
            self._leakage(),
            {},
            {"passed": False, "floor_null_mean": 1.0},
            level2={"passed": True},
        )
        self.assertEqual(decision["verdict"], "STOP_DATA")
        self.assertTrue(any("premise" in reason for reason in decision["reasons"]))

    def test_a_control_that_did_not_fire_blocks_the_gate(self) -> None:
        """A clean M0 result is only admissible with a demonstrated detector."""
        decision = protocol.gate_decision(
            self._leakage(__controls_fired__=False),
            {},
            {"passed": True},
            level2={"passed": True},
        )
        self.assertEqual(decision["verdict"], "STOP_DATA")
        self.assertTrue(
            any("control" in reason for reason in decision["reasons"]),
            decision["reasons"],
        )


class RoleAssignmentTest(unittest.TestCase):
    """Roles belong to the law, not to the argument order (ADR 0005)."""

    def test_roles_follow_the_law(self) -> None:
        self.assertEqual(roles.roles_for("containment").probe, "inner")
        self.assertEqual(roles.roles_for("support").probe, "supported")
        self.assertEqual(roles.roles_for("attachment").probe, "endpoint_1")
        self.assertEqual(roles.roles_for("free").probe, "none")

    def test_unknown_mechanism_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roles.roles_for("holding")

    def test_every_law_covers_every_mechanism(self) -> None:
        # A law added without a role entry would otherwise raise only when a
        # dataset happened to mention it, which is far from where it is noticed.
        from tools.relational_emergence.simulator.compatibility import MECHANISMS

        self.assertEqual(set(roles.ROLES_BY_MECHANISM), set(MECHANISMS))

    def test_attachment_is_exactly_symmetric(self) -> None:
        # The generative process must not prefer either endpoint, or a later
        # role-equivariance check would be measuring the encoder against a
        # ground truth that does not exist.
        self.assertTrue(roles.roles_for("attachment").symmetric)
        self.assertIn("attachment", roles.SYMMETRIC_MECHANISMS)
        # The two endpoint names differ, but neither is privileged: the roles
        # swap cleanly and reversing twice returns the original.
        self.assertEqual(
            roles.roles_for("attachment").reversed().reversed(),
            roles.roles_for("attachment"),
        )

    def test_containment_and_support_are_not_symmetric(self) -> None:
        for mechanism in ("containment", "support"):
            self.assertFalse(roles.roles_for(mechanism).symmetric)
            self.assertNotEqual(
                roles.roles_for(mechanism).probe, roles.roles_for(mechanism).supporter
            )

    def test_swap_is_an_involution_for_every_law(self) -> None:
        # An evaluator fitting T_swap expects T_swap^2 = I. That has to hold for
        # every law, not only the symmetric ones, or the expectation would be
        # broken by the ontology rather than by the encoder.
        for mechanism in roles.ROLES_BY_MECHANISM:
            self.assertTrue(roles.swap_is_involution(mechanism), mechanism)

    def test_reversing_twice_is_the_identity(self) -> None:
        for mechanism in roles.ROLES_BY_MECHANISM:
            original = roles.roles_for(mechanism)
            self.assertEqual(original.reversed().reversed(), original)

    def test_both_directions_share_one_episode_id_and_swap_roles(self) -> None:
        forward, reverse = roles.ordered_pairs("g0#3", "containment")
        self.assertEqual(forward.episode_id, reverse.episode_id)
        self.assertEqual(forward.subject, reverse.object)
        self.assertEqual(forward.object, reverse.subject)
        self.assertEqual(forward.subject_role, reverse.object_role)
        self.assertEqual(forward.object_role, reverse.subject_role)

    def test_the_relation_name_is_the_law_in_both_directions(self) -> None:
        # Naming the reverse direction R^-1 would double the label space while
        # carrying no extra information: containment seen from the container is
        # still containment.
        forward, reverse = roles.ordered_pairs("g0#3", "support")
        self.assertEqual(forward.relation, "support")
        self.assertEqual(reverse.relation, "support")


class DatasetSplitTest(unittest.TestCase):
    """A split that leaks reports generalization where it measured memorization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.groups = groups_for(CELL, 24)

    def test_every_kind_partitions_without_overlap(self) -> None:
        for kind in SPLITS.SPLIT_KINDS:
            split = SPLITS.build_split(self.groups, kind, seed=0)
            self.assertEqual(set(split.train) & set(split.test), set(), kind)
            self.assertEqual(len(split.train) + len(split.test), len(self.groups), kind)

    def test_unseen_filler_holds_out_whole_identities(self) -> None:
        # Splitting by row instead of by filler would leave the same object on
        # both sides, which is the leak this split exists to prevent.
        split = SPLITS.build_split(self.groups, "unseen_filler", seed=0)
        by_key = {group.spec.key(): group.filler.index for group in self.groups}
        train_fillers = {by_key[key] for key in split.train}
        test_fillers = {by_key[key] for key in split.test}
        self.assertFalse(train_fillers & test_fillers)
        self.assertTrue(split.held_out)

    def test_unseen_family_holds_out_whole_families(self) -> None:
        split = SPLITS.build_split(self.groups, "unseen_family", seed=0)
        by_key = {group.spec.key(): group.filler.nuisance.appearance_family for group in self.groups}
        train_families = {by_key[key] for key in split.train}
        test_families = {by_key[key] for key in split.test}
        self.assertFalse(train_families & test_families)
        self.assertEqual(test_families, set(split.held_out))

    def test_pair_recombination_holds_out_pairs_but_not_identities(self) -> None:
        """Both halves of the claim, and the second is the one that is easy to lose.

        Holding out every pair containing an identity would remove that identity
        from training entirely, turning this into an unseen-identity split that
        measures something easier than it says. The construction drops such
        holdouts; this asserts it did.
        """
        split = SPLITS.build_split(self.groups, "pair_recombination", seed=0)
        by_key = {group.spec.key(): SPLITS.identity_pair(group) for group in self.groups}
        train_pairs = {by_key[key] for key in split.train}
        test_pairs = {by_key[key] for key in split.test}
        self.assertFalse(train_pairs & test_pairs)
        self.assertTrue(test_pairs)
        train_probes = {pair[0] for pair in train_pairs}
        train_supporters = {pair[1] for pair in train_pairs}
        for probe, supporter in test_pairs:
            self.assertIn(probe, train_probes)
            self.assertIn(supporter, train_supporters)

    def test_identity_buckets_use_the_declared_range_not_the_sample(self) -> None:
        # Edges fixed by the sample would move when the pool is resampled, so a
        # held-out pair would not stay held out across runs.
        low = SPLITS.probe_identity(type("S", (), {"i_half_u": 0.2501})())
        high = SPLITS.probe_identity(type("S", (), {"i_half_u": 0.3499})())
        self.assertNotEqual(low, high)
        self.assertEqual(
            low, SPLITS.probe_identity(type("S", (), {"i_half_u": 0.2501})())
        )

    def test_the_split_is_deterministic_given_the_seed(self) -> None:
        for kind in SPLITS.SPLIT_KINDS:
            first = SPLITS.build_split(self.groups, kind, seed=3)
            second = SPLITS.build_split(self.groups, kind, seed=3)
            self.assertEqual(first, second, kind)

    def test_an_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SPLITS.build_split(self.groups, "unseen_morphology", seed=0)

    def test_the_four_planned_kinds_are_all_present(self) -> None:
        # A kind that silently disappears from the tuple would make a whole
        # generalization axis unreachable while every test still passed.
        self.assertEqual(
            set(SPLITS.SPLIT_KINDS),
            {"iid", "unseen_filler", "unseen_family", "pair_recombination"},
        )

    def test_an_impossible_test_fraction_is_rejected(self) -> None:
        for fraction in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                SPLITS.build_split(self.groups, "iid", seed=0, test_fraction=fraction)

    def test_disjointness_is_asserted_not_warned(self) -> None:
        # Constructing a split object by hand with an overlap must raise, so a
        # future caller cannot build one and simply not check.
        with self.assertRaises(ValueError):
            SPLITS.GroupSplit(
                kind="iid",
                seed=0,
                train=("a", "b"),
                test=("b", "c"),
                held_out=(),
            ).assert_disjoint()


if __name__ == "__main__":
    unittest.main()
