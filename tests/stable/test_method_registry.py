"""The method registry is the single source of truth for method identity.

A method's name is currently written down in five places that have already
drifted.  The registry replaces them, but it can only do so if the checker can
read it: ``tools/ci_validate.py`` runs on the dependency-free CI job, and
``src/methods/__init__.py`` cannot be imported there because it pulls in torch.
That is why the registry stores ``module`` and ``class`` as strings and resolves
them lazily, and why this file guards the property rather than the text.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ci_validate():
    """Load the pre-PR checker by path, as the stable characterisation test does."""
    script = REPO_ROOT / "tools" / "ci_validate.py"
    spec = importlib.util.spec_from_file_location("opensgg_ci_validate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistryImportTest(unittest.TestCase):
    def test_importable_without_third_party_packages(self) -> None:
        """``-S`` drops site-packages, which is the CI ``validate`` job's env.

        A registry that imported a model module to name a class would pass every
        other test here and fail exactly where it is needed, so this is the
        first thing pinned.
        """
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import src.method_registry"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class RegistryCompletenessTest(unittest.TestCase):
    """The table must name every method module the system ships, and no others.

    Checked against the filesystem rather than against ``method_maps``: once the
    runtime dict is built *from* this table, comparing the two would assert
    nothing.  ``base_method`` is the shared base, not a method.
    """

    def _method_modules_on_disk(self) -> set[str]:
        directory = REPO_ROOT / "src" / "methods"
        return {
            path.stem
            for path in directory.glob("*_method.py")
            if path.stem != "base_method"
        }

    def test_every_spec_names_a_module_that_exists(self) -> None:
        from src.method_registry import METHODS

        for spec in METHODS:
            with self.subTest(method=spec.key):
                self.assertTrue(
                    (REPO_ROOT / "src" / "methods" / f"{spec.method_module}.py").is_file(),
                    f"{spec.key} names a missing module: {spec.method_module}",
                )

    def test_cli_and_config_names_resolve_to_the_same_spec(self) -> None:
        from src.method_registry import BY_CLI_NAME, BY_KEY, METHODS, key_for_config_stem

        self.assertEqual(len(BY_CLI_NAME), len(METHODS))
        self.assertEqual(len(BY_KEY), len(METHODS))
        for spec in METHODS:
            with self.subTest(method=spec.key):
                self.assertIs(BY_CLI_NAME[spec.cli_name], spec)
                self.assertIs(BY_KEY[spec.key], spec)
                for stem in spec.config_stems:
                    self.assertEqual(key_for_config_stem(stem), spec.key)

    def test_every_method_module_is_registered(self) -> None:
        from src.method_registry import METHODS

        registered = {spec.method_module for spec in METHODS}
        self.assertEqual(self._method_modules_on_disk() - registered, set())

    def test_every_cls_name_is_defined_in_its_module(self) -> None:
        """``resolve`` imports the module and getattrs the class.

        Checked by parsing rather than importing: importing a method module
        pulls in torch, which the CI ``validate`` job does not have.  Without
        this, a typo in ``cls_name`` would pass every test here and fail only
        when a run tried to build that method.
        """
        import ast

        from src.method_registry import METHODS

        for spec in METHODS:
            with self.subTest(method=spec.key):
                source = (
                    REPO_ROOT / "src" / "methods" / f"{spec.method_module}.py"
                ).read_text(encoding="utf-8")
                defined = {
                    node.name
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.ClassDef)
                }
                self.assertIn(
                    spec.cls_name,
                    defined,
                    f"{spec.method_module}.py does not define {spec.cls_name}",
                )


class CheckerRegistryTest(unittest.TestCase):
    """The checker runs in the dependency-free CI job, so it must read the
    registry rather than the runtime dict, which imports torch."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_registered_method_without_a_config_file_is_reported(self) -> None:
        from src.method_registry import METHODS

        ci_validate = _load_ci_validate()
        wanted = METHODS[0].config_stem
        for spec in METHODS:
            for stem in spec.config_stems:
                if stem != wanted:
                    (self.tmp / f"{stem}.py").write_text("method = 'x'\n")

        errors = ci_validate.validate_registry_configs(self.tmp)

        self.assertTrue(any(wanted in e for e in errors), errors)

    def test_config_declaring_an_unregistered_method_is_reported(self) -> None:
        """The direction ``validate_configs`` covered; it must survive the move."""
        from src.method_registry import METHODS

        ci_validate = _load_ci_validate()
        for spec in METHODS:
            for stem in spec.config_stems:
                (self.tmp / f"{stem}.py").write_text(f"method = '{spec.cli_name}'\n")
        (self.tmp / "Nonesuch.py").write_text("method = 'Nonesuch'\n")

        errors = ci_validate.validate_registry_configs(self.tmp)

        self.assertTrue(any("Nonesuch" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
