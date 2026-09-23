"""Characterization contracts for the stable SGG system.

These tests pin the behaviour the main repository exists to provide: the training
entry point, the registered Visual Genome configs, the core evaluator, the smoke
test's model-to-method resolver, and the reproduction guardrails.  They are the
"green before, green after" baseline for the research-surface migration in issue
#113 -- the migration is only safe while every one of them keeps passing.

Two of them are boundary assertions rather than surface assertions.  A
repository whose supported runtime or whose documented pre-PR workflow reached
into research-only code could not survive removing that code, so the absence of
that reach is itself a contract worth pinning.

Most of this runs in the dependency-free ``validate`` CI job, so anything
needing torch is guarded through ``tests._optional``.
"""

from __future__ import annotations

import importlib.util
import posixpath
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType

from tests._optional import skip_unless

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str) -> ModuleType:
    """Import a loose script by path.

    ``tools/`` is not an importable package for these scripts, so the repository
    loads them through an explicit spec.  The spec name is what shows up in
    tracebacks; it does not have to match the filename.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingEntrypointTest(unittest.TestCase):
    def test_main_training_entrypoint_imports(self) -> None:
        """``train.py`` must import without torch and without the framework.

        The entry point does its framework imports under the ``__main__`` guard,
        which is what lets tooling read or lint the file without paying for
        lightning.  Binding ``BaseExperiment`` at module scope would mean an
        import of ``train`` silently became a full framework import.
        """
        module = _load_script("opensgg_train_entry", "train.py")

        self.assertFalse(
            hasattr(module, "BaseExperiment"),
            "train.py bound BaseExperiment at module scope; the framework "
            "imports belong under the __main__ guard",
        )


class RegisteredConfigTest(unittest.TestCase):
    def test_registered_vg_configs_load(self) -> None:
        """Every configs/VisualGenome config must parse and name a real method.

        This reuses the AST checks the pre-PR validator already performs rather
        than reimplementing them, so the two cannot drift apart.
        """
        ci_validate = _load_script("opensgg_ci_validate", "tools/ci_validate.py")

        try:
            method_maps = ci_validate.extract_method_maps(
                REPO_ROOT / "src" / "methods" / "__init__.py"
            )
            self.assertTrue(method_maps, "src/methods/__init__.py declared no methods")
            errors = ci_validate.validate_configs(
                REPO_ROOT / "configs" / "VisualGenome", method_maps
            )
        except SystemExit as exc:  # parse_py_file exits hard on a syntax error
            self.fail(f"config validation aborted: {exc}")

        self.assertEqual(errors, [], "registered Visual Genome configs are broken")


class CoreEvaluatorTest(unittest.TestCase):
    @skip_unless("torch", "numpy")
    def test_core_evaluator_imports(self) -> None:
        """The core triplet evaluator must import and expose its public surface."""
        module = _load_script("opensgg_sg_eval", "lib/evaluation/sg_eval.py")

        for name in ("SceneGraphEvaluator", "evaluate_recall", "aggregate_mean_recall"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(module, name), f"sg_eval lost {name}")


class SmokeContractTest(unittest.TestCase):
    def test_main_smoke_contract(self) -> None:
        """The smoke test must still resolve changed files to registered methods.

        ``detect_changed_methods`` is what decides which methods the slow smoke
        job exercises, so a regression here silently narrows CI coverage rather
        than failing it.
        """
        smoke = _load_script("opensgg_ci_smoke_test", "tools/ci_smoke_test.py")

        self.assertEqual(
            smoke.detect_changed_methods(
                "src/methods/usg_method.py configs/VisualGenome/SHA_GCL.py"
            ),
            ["shagcl", "usg"],
        )
        self.assertEqual(smoke.detect_changed_methods(""), [])

        # The historical filename-to-key mismatches, which a naive lowercasing
        # would get wrong.
        for stem, expected in (("GPS_Net", "gpsnet"), ("PE_NET", "penet"), ("SHA_GCL", "shagcl")):
            with self.subTest(stem=stem):
                self.assertEqual(smoke._resolve_config_key(stem), expected)

    @skip_unless("torch")
    def test_smoke_resolves_model_stems(self) -> None:
        """Model stems resolve to method keys (needs src.methods, hence torch)."""
        smoke = _load_script("opensgg_ci_smoke_test", "tools/ci_smoke_test.py")

        self.assertEqual(smoke._find_method_for_model("usg"), "usg")
        self.assertEqual(smoke._find_method_for_model("not_a_model"), None)


class ReproductionWorkflowIndependenceTest(unittest.TestCase):
    """The generic reproduction workflow must outlive the Phase I decision records.

    Issue #113 moves ADRs 0005-0010 and the relational-emergence glossary out of
    the main repository.  That is only safe while they are referenced exclusively
    from the research surface, so these tests assert the absence of any reference
    from the generic workflow -- and include a positive control, because a
    scanner that matches nothing would otherwise pass for the wrong reason.
    """

    _PHASE1_REFERENCE_PATTERNS = (
        re.compile(r"reproduction/adr/000[5-9]"),
        re.compile(r"reproduction/adr/0010"),
        re.compile(r"\bCONTEXT\.md\b"),
    )

    _MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")

    _EXCLUDED_ADR_PREFIXES = tuple(
        f"reproduction/adr/{number:04d}" for number in range(5, 11)
    )

    # The boundary guardrail enumerates the retired ADR band and the glossary as
    # policy targets -- it has to name them in order to freeze them.  That is the
    # opposite of depending on their content, so it is exempt by name, and the
    # exemption is narrow on purpose: any other file that starts naming these
    # paths is a genuine finding.
    _BOUNDARY_POLICY_FILES = frozenset(
        {"tools/reproduction/check_repository_boundary.py"}
    )

    @classmethod
    def _dependency_hits(cls, posix_path: str, text: str) -> list[str]:
        """Citations that would break if the Phase I records were removed.

        The distinction this draws is the whole point of the test.  A markdown
        *link*, or a path a script consumes, is a dependency.  A bare mention in
        prose is not, and neither is a CI change-trigger: the workflow lists
        ``CONTEXT.md`` so that a policy edit *runs the boundary check*, which
        still works if the file is gone -- the trigger simply stops firing.

        Without that distinction the check would be unusable, because the
        migration record has to name these files in order to describe the
        migration that moves them.
        """
        hits: list[str] = []

        if posix_path.endswith(".py"):
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(
                    pattern.search(line) for pattern in cls._PHASE1_REFERENCE_PATTERNS
                ):
                    hits.append(f"{posix_path}:{lineno}: {line.strip()}")
            return hits

        base = posix_path.rsplit("/", 1)[0] if "/" in posix_path else ""
        for lineno, line in enumerate(text.splitlines(), start=1):
            for target in cls._MARKDOWN_LINK.findall(line):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                resolved = posixpath.normpath(posixpath.join(base, target))
                if any(
                    pattern.search(resolved) for pattern in cls._PHASE1_REFERENCE_PATTERNS
                ):
                    hits.append(f"{posix_path}:{lineno}: links to {target}")
        return hits

    @classmethod
    def _generic_workflow_texts(cls) -> list[Path]:
        """Text files that belong to the supported workflow, not the research surface."""
        candidates: list[Path] = []
        candidates.extend(sorted((REPO_ROOT / "tools" / "reproduction").glob("*.py")))
        candidates.extend(sorted((REPO_ROOT / "reproduction").rglob("*.md")))
        candidates.extend(sorted((REPO_ROOT / ".github").rglob("*.md")))
        candidates.extend(sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
        candidates.extend(
            REPO_ROOT / name for name in ("AGENTS.md", "CONTRIBUTING.md", "README.md")
        )

        kept: list[Path] = []
        for path in candidates:
            if not path.is_file():
                continue
            posix = path.relative_to(REPO_ROOT).as_posix()
            if posix.startswith(cls._EXCLUDED_ADR_PREFIXES):
                continue
            if posix in cls._BOUNDARY_POLICY_FILES:
                continue
            kept.append(path)
        return kept

    def test_scanner_separates_links_from_mentions(self) -> None:
        """Positive control, both directions.

        A scanner that matched nothing would pass the real assertion for the
        wrong reason, and one that matched every mention would flag the
        migration record for describing itself.
        """
        linked = "See [ADR 0005](reproduction/adr/0005-phase1-data-generating-process.md).\n"
        mentioned = (
            "ADR 0005 and `reproduction/adr/0005-phase1-data-generating-process.md`\n"
            "are discussed, and so is CONTEXT.md.\n"
        )
        scripted = "RECORD = 'reproduction/adr/0010-phase1-v2-development-boundary.md'\n"

        self.assertEqual(len(self._dependency_hits("reports/README.md", linked)), 1)
        self.assertEqual(self._dependency_hits("reports/README.md", mentioned), [])
        self.assertEqual(len(self._dependency_hits("tools/reproduction/tool.py", scripted)), 1)

    def test_boundary_policy_exemption_is_load_bearing(self) -> None:
        """The exemption must cover a file that really does name them.

        An exemption that exempts nothing is how a scan quietly stops covering
        what it claims to cover.
        """
        for relative in sorted(self._BOUNDARY_POLICY_FILES):
            with self.subTest(path=relative):
                text = (REPO_ROOT / relative).read_text(
                    encoding="utf-8", errors="replace"
                )
                self.assertTrue(
                    self._dependency_hits(relative, text),
                    f"{relative} no longer names these paths; drop the exemption",
                )

    def test_reproduction_workflow_does_not_depend_on_phase1_decision_records(self) -> None:
        """No supported workflow file may consume the Phase I decision records."""
        offenders: list[str] = []
        for path in self._generic_workflow_texts():
            posix = path.relative_to(REPO_ROOT).as_posix()
            hits = self._dependency_hits(
                posix, path.read_text(encoding="utf-8", errors="replace")
            )
            offenders.extend(hits)

        self.assertEqual(
            offenders,
            [],
            "the generic reproduction workflow depends on Phase I research records; "
            "extracting them would break these:\n" + "\n".join(offenders),
        )

    def test_reproduction_checkers_import_without_research_packages(self) -> None:
        """The guardrail scripts must not reach research-only code.

        ``sys.modules`` is poisoned rather than the packages being absent, so the
        assertion does not depend on which optional packages happen to be
        installed.  A pseudo-module of ``None`` makes any import of that name
        raise ImportError.
        """
        blocked = ("tools.relational_emergence", "tools.ontology_probe")
        saved = {name: sys.modules.get(name) for name in blocked}
        for name in blocked:
            sys.modules[name] = None  # type: ignore[assignment]
        try:
            for name, relative in (
                ("opensgg_claims", "tools/reproduction/check_reproduction_claims.py"),
                ("opensgg_docs", "tools/reproduction/check_reproduction_docs.py"),
                ("opensgg_guardrails", "tools/reproduction/run_reproduction_guardrails.py"),
            ):
                with self.subTest(module=relative):
                    _load_script(name, relative)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
