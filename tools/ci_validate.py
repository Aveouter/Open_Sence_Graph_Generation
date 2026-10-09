#!/usr/bin/env python3
"""
CI Validation Script for OpenSGG.

Validates the integrity of model/method/config registration without importing
torch or lightning.  The one project module it imports is
``src.method_registry``, which is stdlib-only by design -- the tables it
replaced could not be read here at all, so this checker used to re-derive them
by AST-parsing ``method_maps`` out of ``src/methods/__init__.py``.

Scope:
  - Steps 2-4 (global consistency checks) only ERROR when the PR touches
    the files those checks inspect.  Otherwise pre-existing inconsistencies
    are reported as warnings so they don't block unrelated PRs.
  - Step 5 (new-file registration) always errors — it is the core PR check.

Checks:
  1. Read src/method_registry.py, the single source of truth for method identity
  2. Every registered method has its config, and every config names a registered
     method (error if touching configs/)
  3. Model ↔ method registration consistency (error if touching src/models/ or src/methods/)
  4. utils/parser.py derives --method choices from the registry instead of listing
     them (error if touching parser.py or methods/__init__)
  5. New files in the PR diff are properly registered
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent

# The registry is the source of truth for method identity and is stdlib-only by
# design, so this checker reads it directly instead of re-deriving the tables by
# AST.  Run as a script, sys.path[0] is tools/, so the repository root has to be
# added for `src.method_registry` to resolve.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This script is part of the required pre-PR workflow on whatever platform a
# contributor uses, and its progress output contains a few non-ASCII glyphs
# (arrows, em dashes). On a cp1252 console those raise UnicodeEncodeError and the
# whole check dies before it can report anything. Pinning the stream encoding
# fixes every glyph at once, including ones added later.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def parse_py_file(path: Path) -> ast.Module:
    """Parse a Python file and return its AST."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print(f"[ERROR] Syntax error in {path}: {e}")
        sys.exit(1)


def extract_imports_from_init(path: Path) -> Set[str]:
    """Extract the set of imported names (local modules) from an __init__.py."""
    tree = parse_py_file(path)
    imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                module = node.module.lstrip(".")
                imports.add(module)
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}")

    return imports


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def declared_method_value(cfg_path: Path) -> str | None:
    """The ``method = '<name>'`` constant a config file assigns, if any."""
    tree = parse_py_file(cfg_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "method"
                    and isinstance(node.value, ast.Constant)
                ):
                    return node.value.value
    return None


def validate_registry_configs(config_dir: Path) -> List[str]:
    """Check ``config_dir`` against the registry, in both directions.

    Replaces the ``method_maps``-based checker this file used to carry.  It asks
    the same two questions -- does every registered method have a config, and
    does every config declare a registered method -- but reads the answers from
    ``src/method_registry``, which is stdlib-only.  ``method_maps`` is not: it
    holds live classes, so importing it here would import torch and this checker
    runs in the dependency-free CI job.
    """
    from src.method_registry import BY_KEY, METHODS

    errors: List[str] = []

    if not config_dir.exists():
        errors.append(f"Config directory not found: {config_dir}")
        return errors

    present = {path.stem for path in config_dir.glob("*.py")}
    if not present:
        errors.append(f"No config files found in {config_dir}")
        return errors

    for spec in METHODS:
        for stem in spec.config_stems:
            if stem not in present:
                errors.append(
                    f"[{spec.key}] the registry names config '{stem}' but "
                    f"{config_dir.name}/{stem}.py does not exist"
                )

    for path in sorted(config_dir.glob("*.py")):
        declared = declared_method_value(path)
        if declared is None:
            errors.append(f"[{path.name}] Missing 'method' field assignment")
        elif declared.lower() not in BY_KEY:
            errors.append(
                f"[{path.name}] method='{declared}' does not match any registered "
                f"method. Known: {sorted(BY_KEY)}"
            )

    return errors


def validate_model_method_mapping(
    models_dir: Path,
    methods_dir: Path,
    models_init: Path,
) -> List[str]:
    """Check that models and methods are consistently registered.

    The method side is checked against the registry rather than against the
    import list in ``src/methods/__init__.py``: that file now resolves its
    classes from the registry, so an import list no longer exists to check.
    Whether every method module is registered is covered by
    ``tests/stable/test_method_registry.py``.
    """
    from src.method_registry import METHODS

    errors: List[str] = []

    for spec in METHODS:
        expected_file = methods_dir / f"{spec.method_module}.py"
        if not expected_file.exists():
            errors.append(
                f"method_registry['{spec.key}'] → {spec.cls_name}, but "
                f"src/methods/{spec.method_module}.py does not exist"
            )

    models_init_imports = extract_imports_from_init(models_init)
    for p in sorted(models_dir.glob("*.py")):
        if p.name in ("__init__.py", "backbone.py"):
            continue
        stem = p.stem
        if stem not in models_init_imports:
            if stem.lower() not in {i.lower() for i in models_init_imports}:
                errors.append(
                    f"Model file src/models/{p.name} is not imported in "
                    f"src/models/__init__.py"
                )

    return errors


def validate_parser_derives_from_registry(parser_path: Path) -> List[str]:
    """``--method`` must derive its choices from the registry, not list them.

    Once the parser builds ``choices`` from ``METHODS`` the two cannot disagree,
    so there is nothing left to cross-check -- what is worth guarding is that
    the list does not come *back*.  A literal here is the drift this change
    removes: the previous version of this checker existed only to notice when
    that literal and ``method_maps`` fell out of step.
    """
    tree = parse_py_file(parser_path)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        method_arg = any(
            isinstance(a, ast.Constant) and a.value in ("--method", "-m")
            for a in node.args
        )
        if not method_arg:
            continue
        for keyword in node.keywords:
            if keyword.arg != "choices":
                continue
            if isinstance(keyword.value, (ast.List, ast.Tuple)):
                return [
                    f"{parser_path.name}: --method choices are a literal; derive "
                    "them from src.method_registry.METHODS so there is one table"
                ]
        return []

    return [f"{parser_path.name}: no --method argument found"]


def validate_new_files(changed_files: List[str]) -> List[str]:
    """Check that newly added model/method/config files are properly registered.

    The method side checks the registry only.  ``src/methods/__init__.py`` used
    to import every method module, and this check demanded a new method appear
    in that import list; that file now resolves classes from the registry, so
    the list no longer exists and demanding it would fail every new method.
    Registry membership is the registration -- ``resolve()`` imports the module
    named by the spec -- and step 3 checks the reverse direction.
    """
    from src.method_registry import METHODS

    errors: List[str] = []

    new_model_files = [
        f
        for f in changed_files
        if f.startswith("src/models/")
        and f.endswith(".py")
        and "__init__" not in f
        and "backbone" not in f
    ]
    new_method_files = [
        f
        for f in changed_files
        if f.startswith("src/methods/")
        and f.endswith(".py")
        and "__init__" not in f
        and "base_method" not in f
    ]
    new_config_files = [
        f for f in changed_files if f.startswith("configs/") and f.endswith(".py")
    ]

    if new_model_files:
        models_init = ROOT / "src" / "models" / "__init__.py"
        models_imports = extract_imports_from_init(models_init)
        for f in new_model_files:
            stem = Path(f).stem
            if stem not in models_imports:
                if stem.lower() not in {i.lower() for i in models_imports}:
                    errors.append(
                        f"[NEW FILE] {f} is not imported in "
                        f"src/models/__init__.py — add: from .{stem} import ..."
                    )

    if new_method_files:
        known_keys = {spec.key for spec in METHODS}
        for f in new_method_files:
            stem = Path(f).stem.replace(".py", "")
            expected_key = stem.replace("_method", "")
            if expected_key not in known_keys:
                errors.append(
                    f"[NEW FILE] {f}: not registered in src/method_registry.py "
                    f"under key '{expected_key}'"
                )

    if new_config_files:
        for f in new_config_files:
            cfg_path = ROOT / f
            if cfg_path.exists():
                method_value = declared_method_value(cfg_path)

                if method_value is None:
                    errors.append(f"[NEW FILE] {f}: missing 'method' field assignment")
                elif method_value.lower() not in known_keys:
                    errors.append(
                        f"[NEW FILE] {f}: method='{method_value}' not found in "
                        f"src/method_registry.py. Did you register it?"
                    )

    return errors


# ---------------------------------------------------------------------------
# Changed-files helpers
# ---------------------------------------------------------------------------


def get_pr_changed_files() -> List[str]:
    """Get list of files changed in this PR (relative to merge-base)."""
    try:
        base_ref = os.environ.get("GITHUB_BASE_REF", "origin/main")
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRD", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRD", "HEAD~1...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except Exception:
        pass

    return []


def _any_changed(patterns: List[str], changed: List[str]) -> bool:
    """Check if any changed file matches one of the glob-like patterns."""
    for f in changed:
        for pat in patterns:
            if pat.endswith("/**"):
                if f.startswith(pat[:-3]):
                    return True
            elif pat.endswith("/"):
                if f.startswith(pat):
                    return True
            elif f == pat or f.endswith("/" + pat):
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []

    models_init = ROOT / "src" / "models" / "__init__.py"
    models_dir = ROOT / "src" / "models"
    methods_dir = ROOT / "src" / "methods"
    config_dir = ROOT / "configs" / "VisualGenome"
    parser_path = ROOT / "utils" / "parser.py"

    changed_files = get_pr_changed_files()
    is_pr = bool(os.environ.get("GITHUB_BASE_REF")) or len(changed_files) > 0

    print("=" * 60)
    print("OpenSGG CI Validation")
    print("=" * 60)
    if changed_files:
        print(f"\nPR changed files ({len(changed_files)}):")
        for f in changed_files:
            print(f"  {f}")

    # ================================================================
    # 1. Read the registry (fatal if empty)
    # ================================================================
    from src.method_registry import METHODS

    print("\n[1/5] Reading src/method_registry.py ...")
    print(f"  Found {len(METHODS)} registered methods: {sorted(s.key for s in METHODS)}")
    if not METHODS:
        errors.append("No registered methods found in src/method_registry.py!")
    else:
        for spec in sorted(METHODS, key=lambda s: s.key):
            print(f"    {spec.key:20s} → {spec.cls_name}")

    # ================================================================
    # 2. Config validation
    #    → ERROR only when PR touches configs/ or methods/__init__.py
    # ================================================================
    print(f"\n[2/5] Validating configs in {config_dir} ...")
    cfg_issues = validate_registry_configs(config_dir)
    cfg_touched = _any_changed(["configs/", "src/methods/__init__.py"], changed_files)
    if cfg_issues:
        for e in cfg_issues:
            if cfg_touched or not is_pr:
                print(f"  [FAIL] {e}")
                errors.append(e)
            else:
                print(f"  [WARN] (pre-existing) {e}")
                warnings.append(e)
    if not cfg_issues:
        n = len(list(config_dir.glob("*.py")))
        print(f"  All configs OK ({n} files)")

    # ================================================================
    # 3. Model ↔ Method mapping
    #    → ERROR only when PR touches src/models/ or src/methods/
    # ================================================================
    print("\n[3/5] Validating model ↔ method registration ...")
    map_issues = validate_model_method_mapping(models_dir, methods_dir, models_init)
    map_touched = _any_changed(["src/models/", "src/methods/"], changed_files)
    if map_issues:
        for e in map_issues:
            if map_touched or not is_pr:
                print(f"  [FAIL] {e}")
                errors.append(e)
            else:
                print(f"  [WARN] (pre-existing) {e}")
                warnings.append(e)
    if not map_issues:
        print("  All model/method registrations OK")

    # ================================================================
    # 4. Parser choices
    #    → ERROR only when PR touches utils/parser.py or methods/__init__.py
    # ================================================================
    print("\n[4/5] Validating parser --method choices ...")
    parser_issues = validate_parser_derives_from_registry(parser_path)
    parser_touched = _any_changed(
        ["utils/parser.py", "src/methods/__init__.py"], changed_files
    )
    if parser_issues:
        for e in parser_issues:
            if parser_touched or not is_pr:
                print(f"  [FAIL] {e}")
                errors.append(e)
            else:
                print(f"  [WARN] (pre-existing) {e}")
                warnings.append(e)
    if not parser_issues:
        print("  Parser derives --method choices from the registry")

    # ================================================================
    # 5. New-file registration (always errors — PR-specific)
    # ================================================================
    print("\n[5/5] Checking PR diff for new file registration ...")
    new_errors = validate_new_files(changed_files)
    if new_errors:
        for e in new_errors:
            print(f"  [FAIL] {e}")
        errors.extend(new_errors)
    elif changed_files:
        print("  All new/modified files properly registered")
    else:
        print("  No changed files detected")

    # ================================================================
    # Report
    # ================================================================
    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        if warnings:
            print(f"\n({len(warnings)} pre-existing warning(s) — see above)")
        return 1
    else:
        print("ALL CHECKS PASSED")
        if warnings:
            print(f"\n({len(warnings)} pre-existing warning(s) reported above)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
