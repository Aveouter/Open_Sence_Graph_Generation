"""Dependency-free checks: v1 evidence cannot authorize a v2 claim."""

import unittest
from dataclasses import replace

from tools.relational_emergence.config import AUDIT_SEEDS, WORLD_GENERATOR_SEEDS
from tools.relational_emergence.simulator.fillers import ALL_FAMILIES
from tools.relational_emergence.v2.data import build_scenes
from tools.relational_emergence.v2.protocol import (
    AUDIT_FAMILIES,
    AUDIT_SEED,
    DEV_FAMILIES,
    DEV_SEED,
    Protocol,
    code_digest,
    require_phase_ib,
)


class TestProtocol(unittest.TestCase):
    def test_new_allocations_are_disjoint_from_v1(self):
        self.assertFalse(set(DEV_FAMILIES + AUDIT_FAMILIES) & set(ALL_FAMILIES))
        self.assertFalse(set(DEV_FAMILIES) & set(AUDIT_FAMILIES))
        self.assertNotIn(
            AUDIT_SEED, (*AUDIT_SEEDS, *WORLD_GENERATOR_SEEDS.values(), DEV_SEED)
        )

    def test_configuration_refuses_invalid_windows_and_two_object_claim(self):
        for changes in (
            {"objects": 2},
            {"cutoff": 0},
            {"prediction_steps": 30},
            {"seeds": (1, 1, 2)},
            {"groups_per_tuple": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Protocol(), **changes)

    def test_digest_covers_cutoff_and_object_count(self):
        self.assertNotEqual(Protocol().digest(), replace(Protocol(), cutoff=9).digest())
        self.assertNotEqual(
            Protocol().digest(), replace(Protocol(), objects=3).digest()
        )

    def test_audit_generation_is_sealed(self):
        with self.assertRaisesRegex(ValueError, "sealed"):
            build_scenes(Protocol(), world="audit")

    def test_v1_and_partial_evidence_are_refused(self):
        for evidence in ({}, {"verdict": "GO_PHASE_IA"}, {"verdict": "GO_PHASE_IA_V2"}):
            with self.assertRaises(ValueError):
                require_phase_ib(evidence, Protocol(), "dataset")

    def test_every_gate_and_provenance_field_is_required(self):
        protocol = Protocol()
        evidence = {
            "protocol_sha256": protocol.digest(),
            "code_sha256": code_digest(),
            "dataset_sha256": "dataset",
            "verdict": "GO_PHASE_IA_V2",
            "m0": "PASS_NO_DETECTABLE_LEAKAGE",
            "evaluator_controls": "PASS",
            "transplant_controls": "PASS",
            "level2": "PASS",
            "multi_object": "PASS",
            "chronology_control": "PASS",
            "selected_horizon": 10,
            "training_seeds": list(protocol.seeds),
        }
        require_phase_ib(evidence, protocol, "dataset")
        for key in evidence:
            bad = dict(evidence)
            bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                require_phase_ib(bad, protocol, "dataset")
        with self.assertRaises(ValueError):
            require_phase_ib({**evidence, "selected_horizon": 11}, protocol, "dataset")


if __name__ == "__main__":
    unittest.main()
