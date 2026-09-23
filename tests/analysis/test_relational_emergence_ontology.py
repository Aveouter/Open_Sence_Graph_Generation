"""Tests for the factorized ``S_0`` ontology, its compatibility table and the
purity of the ``pose`` predicate.

Stdlib only, so this runs in the dependency-free CI `validate` job -- which is
the point: the ontology has to be frozen before any data exists, and the freeze
must be enforced by the same CI that guards everything else.
"""

from __future__ import annotations

import ast
import inspect
import math
import unittest
from pathlib import Path
from unittest import mock

from tools.relational_emergence.config import (
    DEVELOPMENT_SEEDS,
    WORLD_AUDIT,
    WORLD_DEVELOPMENT,
    Phase1AConfig,
)
from tools.relational_emergence.simulator import compatibility as compat
from tools.relational_emergence.simulator import geometry
from tools.relational_emergence.simulator.factors import (
    CONTACT_VALUES,
    CONTAINMENT_VALUES,
    FACTOR_DOMAINS,
    POSE_VALUES,
    S0,
    derived_readouts,
    enumerate_realizable,
    jointly_consistent,
)

GEOMETRY_PATH = Path(geometry.__file__)

# Identifiers that would mean `geometry` had grown a dependency on mechanism
# state. Docstrings naming these concepts are fine -- the check is on the code.
FORBIDDEN_IDENTIFIER_TOKENS = ("mechanism", "compatibility", "contactforce", "active", "factor")


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


class S0FactorizationTest(unittest.TestCase):
    def test_factor_domains_are_the_frozen_ones(self) -> None:
        self.assertEqual(CONTAINMENT_VALUES, ("inside", "outside"))
        self.assertEqual(CONTACT_VALUES, ("touching", "non_touching"))
        self.assertEqual(POSE_VALUES, ("on_top", "other"))

    def test_realizable_set_is_derived_not_hard_coded(self) -> None:
        expected = tuple(
            S0(c, k, p)
            for c in CONTAINMENT_VALUES
            for k in CONTACT_VALUES
            for p in POSE_VALUES
            if jointly_consistent(c, k, p)
        )
        self.assertEqual(enumerate_realizable(), expected)

    def test_inside_with_on_top_is_excluded(self) -> None:
        for contact in CONTACT_VALUES:
            self.assertFalse(jointly_consistent("inside", contact, "on_top"))

    def test_every_other_combination_is_realizable(self) -> None:
        # 3 factors x 2 values = 8 raw combinations, minus the 2 excluded ones.
        self.assertEqual(len(enumerate_realizable()), 6)

    def test_contact_does_not_constrain_the_other_factors(self) -> None:
        # An object inside a cavity may rest on its floor or float clear of it.
        for containment in CONTAINMENT_VALUES:
            for pose in POSE_VALUES:
                if containment == "inside" and pose == "on_top":
                    continue
                consistent = {jointly_consistent(containment, k, pose) for k in CONTACT_VALUES}
                self.assertEqual(consistent, {True})

    def test_unknown_factor_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            S0("sideways", "touching", "other")
        with self.assertRaises(ValueError):
            jointly_consistent("inside", "almost_touching", "other")

    def test_derived_readouts_are_readouts_not_categories(self) -> None:
        s0 = S0("inside", "non_touching", "other")
        self.assertEqual(
            derived_readouts(s0),
            {"inside": True, "on_top": False, "separated": True},
        )


class CompatibilityTableTest(unittest.TestCase):
    def test_audit_passes_on_the_frozen_table(self) -> None:
        self.assertEqual(compat.audit_compatibility(), [])

    def test_every_realizable_tuple_has_at_least_three_mechanisms(self) -> None:
        for s0 in enumerate_realizable():
            mechanisms = compat.compatible_mechanisms(s0)
            self.assertGreaterEqual(len(mechanisms), compat.MIN_MECHANISMS_PER_TUPLE, s0.key())
            self.assertIn(compat.FREE_MECHANISM, mechanisms, s0.key())

    def test_every_active_mechanism_spans_at_least_two_tuples(self) -> None:
        tuples = enumerate_realizable()
        for mechanism in compat.ACTIVE_MECHANISMS:
            span = [s0.key() for s0 in tuples if compat.compatible(s0, mechanism)]
            self.assertGreaterEqual(
                len(span), compat.MIN_TUPLES_PER_ACTIVE_MECHANISM, mechanism
            )

    def test_bipartite_graph_is_connected(self) -> None:
        self.assertTrue(compat._bipartite_connected(compat.compatibility_table()))

    def test_baseline_is_one_over_the_compatible_count(self) -> None:
        baseline = compat.analytic_baseline_by_tuple()
        for s0 in enumerate_realizable():
            self.assertAlmostEqual(
                baseline[s0.key()], 1.0 / len(compat.compatible_mechanisms(s0))
            )

    def test_baseline_is_not_a_fixed_chance_level(self) -> None:
        # The compatible count varies by tuple, so a hard-coded 25% or 33%
        # figure would be wrong for some rows.
        baseline = compat.analytic_baseline_by_tuple()
        self.assertGreater(len(set(baseline.values())), 1)

    def test_containment_needs_containing_geometry(self) -> None:
        outside_other = S0("outside", "touching", "other")
        self.assertFalse(compat.compatible(outside_other, "containment"))
        self.assertTrue(compat.compatible(S0("outside", "touching", "on_top"), "containment"))
        self.assertTrue(compat.compatible(S0("inside", "touching", "other"), "containment"))

    def test_unknown_mechanism_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compat.compatible(S0("inside", "touching", "other"), "collision")

    def test_audit_fires_when_a_constraint_is_violated(self) -> None:
        """Must-fire control: a broken table has to produce violations.

        Without this, the audit could pass because it checks nothing.
        """
        broken = dict(compat.compatibility_table())
        first = next(iter(broken))
        broken[first] = tuple(m for m in broken[first] if m != compat.FREE_MECHANISM)
        with mock.patch.object(compat, "compatibility_table", return_value=broken):
            violations = compat.audit_compatibility()
        self.assertTrue(any("free" in v for v in violations), violations)

    def test_audit_fires_on_a_disconnected_graph(self) -> None:
        # Two islands: half the tuples reachable only through `free`, the other
        # half only through `attachment`. Note that a table where *every* tuple
        # shares one mechanism is connected, so that would not test anything.
        keys = sorted(compat.compatibility_table())
        half = len(keys) // 2
        broken = {
            key: (("free",) if index < half else ("attachment",))
            for index, key in enumerate(keys)
        }
        self.assertFalse(compat._bipartite_connected(broken))
        with mock.patch.object(compat, "compatibility_table", return_value=broken):
            violations = compat.audit_compatibility()
        self.assertTrue(any("connected" in v for v in violations), violations)


class PosePredicatePurityTest(unittest.TestCase):
    """``pose=on_top`` must stay a pure function of geometry and gravity frame."""

    ALLOWED_PARAMETERS = ("i_pos", "i_half_extent", "j_pos", "j_half_extent", "frame", "tol")

    def test_signature_admits_no_mechanism_state(self) -> None:
        parameters = tuple(inspect.signature(geometry.pose_is_on_top).parameters)
        self.assertEqual(parameters, self.ALLOWED_PARAMETERS)

    def test_geometry_module_does_not_reference_mechanism_state(self) -> None:
        # A structural guard: the predicate cannot read what it cannot name.
        offending = {
            identifier
            for identifier in _module_identifiers(GEOMETRY_PATH)
            for token in FORBIDDEN_IDENTIFIER_TOKENS
            if token in identifier
        }
        self.assertEqual(offending, set())

    def test_geometry_module_does_not_import_the_ontology(self) -> None:
        # `factors` and `compatibility` are the modules that know about `M` and
        # about the observable factors; geometry must depend on neither.
        self.assertNotIn("factors", _module_identifiers(GEOMETRY_PATH))
        self.assertNotIn("compatibility", _module_identifiers(GEOMETRY_PATH))


class PosePredicateBehaviourTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = geometry.GravityFrame(0.0)
        self.j_pos = (0.0, 0.0)
        self.j_half = (1.0, 1.0)

    def test_hovering_directly_above_is_on_top(self) -> None:
        # Deliberately not a contact test: contact is a separate factor.
        self.assertTrue(
            geometry.pose_is_on_top((0.0, 3.0), (0.5, 0.5), self.j_pos, self.j_half, self.frame)
        )

    def test_resting_on_the_face_is_on_top(self) -> None:
        self.assertTrue(
            geometry.pose_is_on_top((0.0, 1.5), (0.5, 0.5), self.j_pos, self.j_half, self.frame)
        )

    def test_below_the_face_is_not_on_top(self) -> None:
        self.assertFalse(
            geometry.pose_is_on_top((0.0, -3.0), (0.5, 0.5), self.j_pos, self.j_half, self.frame)
        )

    def test_laterally_clear_is_not_on_top(self) -> None:
        self.assertFalse(
            geometry.pose_is_on_top((5.0, 3.0), (0.5, 0.5), self.j_pos, self.j_half, self.frame)
        )

    def test_gravity_frame_rotates_what_counts_as_above(self) -> None:
        # pi/2 puts gravity along +x, so "above" becomes -x.
        rotated = geometry.GravityFrame(math.pi / 2)
        self.assertTrue(
            geometry.pose_is_on_top((-3.0, 0.0), (0.5, 0.5), self.j_pos, self.j_half, rotated)
        )
        self.assertFalse(
            geometry.pose_is_on_top((3.0, 0.0), (0.5, 0.5), self.j_pos, self.j_half, rotated)
        )

    def test_gravity_frame_vectors_are_orthonormal(self) -> None:
        for degrees in (0.0, 37.0, 90.0, 180.0, 270.0, 314.0):
            frame = geometry.GravityFrame(math.radians(degrees))
            self.assertAlmostEqual(geometry.dot(frame.up, frame.up), 1.0)
            self.assertAlmostEqual(geometry.dot(frame.lateral, frame.lateral), 1.0)
            self.assertAlmostEqual(geometry.dot(frame.up, frame.lateral), 0.0)
            self.assertAlmostEqual(geometry.dot(frame.up, frame.down), -1.0)


class Phase1AConfigTest(unittest.TestCase):
    def test_default_world_is_development(self) -> None:
        self.assertEqual(Phase1AConfig().world, WORLD_DEVELOPMENT)

    def test_seed_pools_are_disjoint_by_construction(self) -> None:
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(Phase1AConfig().audit_seeds))

    def test_overlapping_seed_pools_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Phase1AConfig(development_seeds=(0, 1), audit_seeds=(1, 2))

    def test_empty_seed_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Phase1AConfig(development_seeds=(), audit_seeds=(10,))

    def test_unknown_world_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Phase1AConfig(world="staging_world")

    def test_seed_pool_follows_the_world(self) -> None:
        config = Phase1AConfig()
        self.assertEqual(config.seed_pool(), config.development_seeds)
        self.assertEqual(
            Phase1AConfig(world=WORLD_AUDIT).seed_pool(),
            Phase1AConfig(world=WORLD_AUDIT).audit_seeds,
        )


class OntologyAuditReportTest(unittest.TestCase):
    def test_report_is_stamped_and_admissible(self) -> None:
        from tools.relational_emergence.audits.ontology_audit import build_report

        report = build_report(Phase1AConfig())
        self.assertEqual(report["status"], "research_experiment")
        self.assertTrue(report["not_a_reproduction"])
        self.assertEqual(report["scope"], "synthetic_relational_emergence_phase1a")
        self.assertTrue(report["admissible"])
        self.assertEqual(report["constraint_violations"], [])
        self.assertEqual(report["n_realizable_tuples"], len(enumerate_realizable()))

    def test_report_lists_every_realizable_tuple_with_its_baseline(self) -> None:
        from tools.relational_emergence.audits.ontology_audit import build_report

        report = build_report(Phase1AConfig())
        reported = {row["containment"] + "|" + row["contact"] + "|" + row["pose"] for row in report["realizable_tuples"]}
        self.assertEqual(reported, {s0.key() for s0 in enumerate_realizable()})
        self.assertEqual(
            set(report["analytic_baseline_by_tuple"]),
            {s0.key() for s0 in enumerate_realizable()},
        )

    def test_report_factor_domains_match_the_frozen_definitions(self) -> None:
        from tools.relational_emergence.audits.ontology_audit import build_report

        report = build_report(Phase1AConfig())
        self.assertEqual(
            {name: tuple(values) for name, values in report["factor_domains"].items()},
            FACTOR_DOMAINS,
        )


if __name__ == "__main__":
    unittest.main()
