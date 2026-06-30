from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "check_reproduction_claims.py"

spec = importlib.util.spec_from_file_location("check_reproduction_claims", SCRIPT)
assert spec is not None and spec.loader is not None
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)


class ReproductionClaimGuardrailTest(unittest.TestCase):
    def test_adjacent_qualifier_suppresses_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qualified.md"
            path.write_text("This is not a reproduction claim.\nreproduced\n")

            self.assertEqual(claims.scan_file(path), [])

    def test_weak_evidence_claim_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.md"
            path.write_text(
                "Baseline reproduced.\nrandom-init tiny-slice eval, all 0.0\n"
            )

            findings = claims.scan_file(path)

        self.assertEqual(len(findings), 1)
        self.assertIn("weak evidence", findings[0])

    def test_missing_explicit_path_is_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "input path does not exist"):
            claims.iter_paths(["definitely-missing-report.md"])

    def test_changed_from_works_from_subdirectory(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "check_reproduction_claims.py",
                "--changed-from",
                "origin/main",
                "../../AGENTS.md",
                "../../reproduction",
                "../../docs/reproduction",
            ],
            cwd=REPO_ROOT / "tools" / "reproduction",
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_invalid_changed_from_is_argparse_error(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--changed-from",
                "definitely-missing-ref",
                "AGENTS.md",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("failed to resolve changed files", result.stderr)


if __name__ == "__main__":
    unittest.main()
