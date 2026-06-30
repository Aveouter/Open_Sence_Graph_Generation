from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "run_evidence_gate_checks.py"

spec = importlib.util.spec_from_file_location("run_evidence_gate_checks", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvidenceGateRunnerTest(unittest.TestCase):
    def test_normalize_status_treats_returncode_2_as_blocked(self) -> None:
        status = runner.normalize_status(
            2,
            {
                "status": "BLOCKED",
                "blockers": ["missing_official_checkpoint"],
            },
        )

        self.assertEqual(status, "BLOCKED")

    def test_normalize_status_treats_missing_list_as_blocked(self) -> None:
        status = runner.normalize_status(
            0,
            {
                "status": "BLOCKED_MISSING_SGB_VG_INPUTS",
                "missing": ["VG-SGG-with-attri.h5"],
            },
        )

        self.assertEqual(status, "BLOCKED")

    def test_normalize_status_pass_requires_clean_zero_exit(self) -> None:
        self.assertEqual(runner.normalize_status(0, {"status": "PASS"}), "PASS")

    def test_unexpected_returncode_is_error(self) -> None:
        self.assertEqual(runner.normalize_status(1, {"status": "BLOCKED"}), "ERROR")

    def test_default_order_matches_requested_priority(self) -> None:
        self.assertEqual(
            list(runner.CHECKS),
            ["freq", "tde", "vctree", "penet", "shagcl", "ra_sgg"],
        )

    def test_summary_claim_boundary_constant(self) -> None:
        self.assertEqual(runner.SUMMARY_CLAIM, "not reproduction-ready")
        self.assertIn("not sufficient", runner.SUMMARY_CLAIM_NOTE)


if __name__ == "__main__":
    unittest.main()
