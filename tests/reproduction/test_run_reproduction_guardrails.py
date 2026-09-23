"""Behavioral coverage for the canonical repository guardrail path."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "run_reproduction_guardrails.py"
spec = importlib.util.spec_from_file_location("run_reproduction_guardrails", SCRIPT)
assert spec is not None and spec.loader is not None
guardrails = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guardrails)


class ProtocolGuardrailIntegrationTest(unittest.TestCase):
    def test_protocol_violation_fails_the_canonical_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analysis_root = Path(directory)
            phase1a = analysis_root / "phase1a"
            phase1a.mkdir()
            (phase1a / "unlabelled.json").write_text(
                json.dumps({"status": "research_experiment"}), encoding="utf-8"
            )

            calls = []
            output = io.StringIO()
            errors = io.StringIO()
            def run_step(name: str, command: list[str], cwd: Path) -> int:
                calls.append(name)
                if name != "Run relational-emergence protocol validation":
                    return 0
                result = subprocess.run(
                    command, cwd=cwd, capture_output=True, text=True, check=False
                )
                output.write(result.stdout)
                errors.write(result.stderr)
                return result.returncode

            with (
                patch.object(sys, "argv", [str(SCRIPT), "--protocol-root", str(analysis_root)]),
                patch.object(guardrails, "git_root", return_value=REPO_ROOT),
                patch.object(guardrails, "run_step", side_effect=run_step),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                try:
                    exit_code = guardrails.main()
                except SystemExit as error:
                    exit_code = int(error.code)

        self.assertEqual(exit_code, 2, output.getvalue() + errors.getvalue())
        self.assertIn("Run relational-emergence protocol validation", calls)
        self.assertIn("unlabelled.json: missing stamp fields", output.getvalue())

    def test_protocol_check_is_skipped_when_research_surface_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def run_step(name: str, command: list[str], cwd: Path) -> int:
                calls.append(name)
                return 0

            with (
                patch.object(sys, "argv", [str(SCRIPT), "--root", str(root)]),
                patch.object(guardrails, "git_root", return_value=root),
                patch.object(guardrails, "run_step", side_effect=run_step),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = guardrails.main()

        self.assertEqual(exit_code, 0)
        self.assertNotIn("Run relational-emergence protocol validation", calls)


if __name__ == "__main__":
    unittest.main()
