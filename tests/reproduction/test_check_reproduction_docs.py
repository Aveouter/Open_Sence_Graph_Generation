from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "check_reproduction_docs.py"

spec = importlib.util.spec_from_file_location("check_reproduction_docs", SCRIPT)
assert spec is not None and spec.loader is not None
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


class ReproductionDocsGuardrailTest(unittest.TestCase):
    def test_add_file_check_reports_untracked_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            path.write_text("audit\n")
            findings: list[str] = []

            docs.add_file_check(findings, path, set(), require_tracked=True)

        self.assertEqual(len(findings), 1)
        self.assertIn("not tracked", findings[0])

    def test_add_file_check_allows_untracked_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            path.write_text("audit\n")
            findings: list[str] = []

            docs.add_file_check(findings, path, set(), require_tracked=False)

        self.assertEqual(findings, [])

    def test_suite_summary_rejects_reproduction_ready_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "reproduction" / "evidence" / "evidence_gate_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "claim": "reproduction-ready",
                        "status": "BLOCKED",
                        "results": [],
                    }
                )
            )

            findings = docs.check_suite_summary(
                root,
                ["freq"],
                {summary},
                require_tracked=True,
            )

        self.assertTrue(
            any("not reproduction-ready" in finding for finding in findings)
        )
        self.assertTrue(
            any("missing suite result for freq" in finding for finding in findings)
        )

    def test_blocked_suite_summary_requires_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "reproduction" / "evidence" / "evidence_gate_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "claim": "not reproduction-ready",
                        "status": "BLOCKED",
                        "results": [
                            {
                                "key": "freq",
                                "status": "BLOCKED",
                                "output": "reproduction/evidence/freq/sgb_freq_input_check.json",
                                "blockers": [],
                            }
                        ],
                    }
                )
            )

            findings = docs.check_suite_summary(
                root,
                ["freq"],
                {summary},
                require_tracked=True,
            )

        self.assertTrue(
            any("BLOCKED result must list blockers" in finding for finding in findings)
        )

    def test_baseline_requires_input_check_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "reproduction" / "evidence" / "freq"
            baseline.mkdir(parents=True)

            findings = docs.check_baseline(
                root,
                "freq",
                docs.BASELINES["freq"],
                set(),
                require_tracked=False,
            )

        self.assertTrue(any("missing file" in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
