#!/usr/bin/env python3
"""
CI Validation Script for OpenSGG.

Validates the integrity of model/method/config registration without importing
any project code (uses AST parsing, no PyTorch/lightning deps required).

Scope:
  - Steps 2-4 (global consistency checks) only ERROR when the PR touches
    the files those checks inspect.  Otherwise pre-existing inconsistencies
    are reported as warnings so they don't block unrelated PRs.
  - Step 5 (new-file registration) always errors — it is the core PR check.

Checks:
  1. Extract method_maps from src/methods/__init__.py
  2. Every config file declares a valid 'method' field (error if touching configs/)
  3. Model ↔ method registration consistency (error if touching src/models/ or src/methods/)
  4. method_maps keys match parser --method choices (error if touching parser.py or methods/__init__)
  5. New files in the PR diff are properly registered in __init__.py files
"""

from __future__ import annotations

import ast
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent

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


def extract_method_maps(path: Path) -> Dict[str, str]:
    """Extract the method_maps dict from src/methods/__init__.py."""
    tree = parse_py_file(path)
    method_maps: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "method_maps":
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values, strict=True):
                            if isinstance(key, ast.Constant):
                                name = key.value
                                cls = (
                                    value.id
                                    if isinstance(value, ast.Name)
                                    else ast.unparse(value)
                                )
                                method_maps[name] = cls

    return method_maps


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


def extract_parser_choices(path: Path) -> List[str]:
    """Extract --method choices from utils/parser.py."""
    tree = parse_py_file(path)
    choices: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                args = node.args
                kwargs = {
                    kw.arg: kw.value for kw in node.keywords if kw.arg is not None
                }
                method_arg = any(
                    isinstance(a, ast.Constant) and a.value in ("--method", "-m")
                    for a in args
                )
                if method_arg and "choices" in kwargs:
                    choices_node = kwargs["choices"]
                    if isinstance(choices_node, ast.List):
                        choices = [
                            elt.value
                            for elt in choices_node.elts
                            if isinstance(elt, ast.Constant)
                        ]
    return choices


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def validate_configs(config_dir: Path, method_maps: Dict[str, str]) -> List[str]:
    """Validate all config files in config_dir."""
    errors: List[str] = []

    if not config_dir.exists():
        errors.append(f"Config directory not found: {config_dir}")
        return errors

    config_files = sorted(config_dir.glob("*.py"))
    if not config_files:
        errors.append(f"No config files found in {config_dir}")
        return errors

    config_methods: Set[str] = set()

    for cfg_path in config_files:
        tree = parse_py_file(cfg_path)
        method_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "method"
                        and isinstance(node.value, ast.Constant)
                    ):
                        method_value = node.value.value

        if method_value is None:
            errors.append(f"[{cfg_path.name}] Missing 'method' field assignment")
            continue

        method_lower = method_value.lower()
        config_methods.add(method_lower)

        if method_lower not in method_maps:
            errors.append(
                f"[{cfg_path.name}] method='{method_value}' does not match "
                f"any key in method_maps. Known: {sorted(method_maps.keys())}"
            )

    missing_configs = set(method_maps.keys()) - config_methods
    if missing_configs:
        errors.append(
            f"method_maps keys without config files: {sorted(missing_configs)}"
        )

    return errors


def validate_model_method_mapping(
    models_dir: Path,
    methods_dir: Path,
    models_init: Path,
    methods_init: Path,
    method_maps: Dict[str, str],
) -> List[str]:
    """Check that models and methods are consistently registered."""
    errors: List[str] = []

    for mm_key, mm_cls in method_maps.items():
        expected_file = methods_dir / f"{mm_key}_method.py"
        if not expected_file.exists():
            errors.append(
                f"method_maps['{mm_key}'] → {mm_cls}, but "
                f"src/methods/{mm_key}_method.py does not exist"
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

    methods_init_imports = extract_imports_from_init(methods_init)
    for p in sorted(methods_dir.glob("*_method.py")):
        if p.name == "base_method.py":
            continue
        stem = p.stem.replace(".py", "")
        if stem not in methods_init_imports:
            errors.append(
                f"Method file src/methods/{p.name} is not imported in "
                f"src/methods/__init__.py"
            )

    mm_keys = set(method_maps.keys())
    methods_imported = {i.lower() for i in methods_init_imports}
    for key in mm_keys:
        expected_import = f"{key}_method"
        if expected_import not in methods_imported:
            errors.append(
                f"method_maps['{key}'] is registered but "
                f"{key}_method is not imported in src/methods/__init__.py"
            )

    return errors


def validate_parser_choices(
    parser_path: Path, method_maps: Dict[str, str]
) -> List[str]:
    """Ensure --method choices in parser.py match method_maps."""
    errors: List[str] = []

    parser_choices = extract_parser_choices(parser_path)
    if not parser_choices:
        errors.append("Could not extract --method choices from utils/parser.py")
        return errors

    parser_set = {c.lower() for c in parser_choices}
    mm_set = set(method_maps.keys())

    only_in_parser = parser_set - mm_set
    only_in_maps = mm_set - parser_set

    if only_in_parser:
        errors.append(
            f"--method choices in parser but NOT in method_maps: "
            f"{sorted(only_in_parser)}"
        )
    if only_in_maps:
        errors.append(
            f"method_maps keys NOT in --method parser choices: {sorted(only_in_maps)}"
        )

    return errors


def validate_new_files(changed_files: List[str]) -> List[str]:
    """Check that newly added model/method/config files are properly registered."""
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
        methods_init = ROOT / "src" / "methods" / "__init__.py"
        methods_imports = extract_imports_from_init(methods_init)
        method_maps = extract_method_maps(methods_init)
        for f in new_method_files:
            stem = Path(f).stem.replace(".py", "")
            if stem not in methods_imports:
                errors.append(
                    f"[NEW FILE] {f} is not imported in "
                    f"src/methods/__init__.py — add: from .{stem} import ..."
                )
            expected_key = stem.replace("_method", "")
            if expected_key not in method_maps:
                errors.append(
                    f"[NEW FILE] {f}: method class should be registered in "
                    f"method_maps under key '{expected_key}'"
                )

    if new_config_files:
        for f in new_config_files:
            cfg_path = ROOT / f
            if cfg_path.exists():
                try:
                    tree = parse_py_file(cfg_path)
                except SyntaxError:
                    errors.append(f"[NEW FILE] {f} has syntax errors")
                    continue

                method_value = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "method"
                                and isinstance(node.value, ast.Constant)
                            ):
                                method_value = node.value.value

                if method_value is None:
                    errors.append(f"[NEW FILE] {f}: missing 'method' field assignment")
                else:
                    methods_init = ROOT / "src" / "methods" / "__init__.py"
                    mm = extract_method_maps(methods_init)
                    if method_value.lower() not in mm:
                        errors.append(
                            f"[NEW FILE] {f}: method='{method_value}' not found "
                            f"in method_maps. Did you register it?"
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

    methods_init = ROOT / "src" / "methods" / "__init__.py"
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
    # 1. Extract method_maps (fatal if empty)
    # ================================================================
    print("\n[1/5] Extracting method_maps from src/methods/__init__.py ...")
    method_maps = extract_method_maps(methods_init)
    print(
        f"  Found {len(method_maps)} registered methods: {sorted(method_maps.keys())}"
    )
    if not method_maps:
        errors.append("No method_maps entries found!")
    else:
        for key, cls_name in sorted(method_maps.items()):
            print(f"    {key:20s} → {cls_name}")

    # ================================================================
    # 2. Config validation
    #    → ERROR only when PR touches configs/ or methods/__init__.py
    # ================================================================
    print(f"\n[2/5] Validating configs in {config_dir} ...")
    cfg_issues = validate_configs(config_dir, method_maps)
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
    map_issues = validate_model_method_mapping(
        models_dir, methods_dir, models_init, methods_init, method_maps
    )
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
    parser_issues = validate_parser_choices(parser_path, method_maps)
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
        print("  Parser choices match method_maps exactly")

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
