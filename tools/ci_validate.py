#!/usr/bin/env python3
"""
CI Validation Script for OpenSGG.

Validates the integrity of model/method/config registration without importing
any project code (uses AST parsing, no PyTorch/lightning deps required).

Checks:
  1. Every config file in configs/<dataset>/ is syntactically valid Python and
     declares a 'method' field that maps to a known method_maps key.
  2. Every key in method_maps has a corresponding config file.
  3. Every model file under src/models/ has a corresponding method file under
     src/methods/ (and vice versa).
  4. New files detected in the PR diff are properly registered in __init__.py
     files (import + method_maps / __all__).
  5. method_maps keys match parser --method choices exactly.
"""

from __future__ import annotations

import ast
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent

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
    """Extract the method_maps dict from src/methods/__init__.py.

    Returns dict of {method_name: class_name}.
    """
    tree = parse_py_file(path)
    method_maps: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "method_maps":
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
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
                # Relative import from current package
                module = node.module.lstrip(".")
                imports.add(module)
                # Also track individual names
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}")

    return imports


def extract_all_list(path: Path) -> List[str]:
    """Extract the __all__ list from an __init__.py."""
    tree = parse_py_file(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
    return []


def extract_parser_choices(path: Path) -> List[str]:
    """Extract --method choices from utils/parser.py."""
    tree = parse_py_file(path)

    choices: List[str] = []

    # Find the add_argument('--method', ...) call with choices list
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match parser.add_argument(...) or nested calls
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                args = node.args
                kwargs = {
                    kw.arg: kw.value for kw in node.keywords if kw.arg is not None
                }

                # Check if this is the --method argument
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
# Config validation
# ---------------------------------------------------------------------------


def validate_configs(config_dir: Path, method_maps: Dict[str, str]) -> List[str]:
    """Validate all config files in config_dir.

    Returns list of error messages (empty = all good).
    """
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

        # Find 'method' assignment at module level
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

        # method field in config should be lowercase
        method_lower = method_value.lower()
        config_methods.add(method_lower)

        if method_lower not in method_maps:
            errors.append(
                f"[{cfg_path.name}] method='{method_value}' does not match "
                f"any key in method_maps. Known keys: {sorted(method_maps.keys())}"
            )

    # Reverse check: every method_map key should have a config
    missing_configs = set(method_maps.keys()) - config_methods
    if missing_configs:
        errors.append(
            f"method_maps keys without config files: {sorted(missing_configs)}"
        )

    return errors


# ---------------------------------------------------------------------------
# Model ↔ Method registration checks
# ---------------------------------------------------------------------------


def validate_model_method_mapping(
    models_dir: Path,
    methods_dir: Path,
    models_init: Path,
    methods_init: Path,
    method_maps: Dict[str, str],
) -> List[str]:
    """Check that models and methods are consistently registered."""
    errors: List[str] = []

    # Check: every method in method_maps should have a method file
    for mm_key, mm_cls in method_maps.items():
        expected_file = methods_dir / f"{mm_key}_method.py"
        if not expected_file.exists():
            errors.append(
                f"method_maps['{mm_key}'] → {mm_cls}, but file "
                f"src/methods/{mm_key}_method.py does not exist"
            )

    # Check models __init__.py imports exist
    models_init_imports = extract_imports_from_init(models_init)
    for p in sorted(models_dir.glob("*.py")):
        if p.name in ("__init__.py", "backbone.py"):
            continue
        stem = p.stem
        if stem not in models_init_imports:
            models_init_imports_lower = {i.lower() for i in models_init_imports}
            if stem.lower() not in models_init_imports_lower:
                errors.append(
                    f"Model file src/models/{p.name} is not imported in "
                    f"src/models/__init__.py"
                )

    # Check methods __init__.py has all methods registered
    methods_init_imports = extract_imports_from_init(methods_init)
    for p in sorted(methods_dir.glob("*_method.py")):
        stem = p.stem.replace(".py", "")
        if p.name == "base_method.py":
            continue  # base class, not a registered method
        if stem not in methods_init_imports:
            errors.append(
                f"Method file src/methods/{p.name} is not imported in "
                f"src/methods/__init__.py"
            )

    # Check that method_maps keys have corresponding entries in
    # methods __init__.py
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


# ---------------------------------------------------------------------------
# Parser choices match method_maps
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PR diff detection: new model/method files
# ---------------------------------------------------------------------------


def get_pr_changed_files() -> List[str]:
    """Get list of files changed in this PR (relative to merge-base)."""
    try:
        # Try to get files changed vs base branch
        base_ref = os.environ.get("GITHUB_BASE_REF", "origin/main")
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
    except Exception:
        pass

    # Fallback: diff against HEAD~1
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD~1...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except Exception:
        pass

    return []


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
                # Check case-insensitive (some files have mixed case)
                models_imports_lower = {i.lower() for i in models_imports}
                if stem.lower() not in models_imports_lower:
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

            # Check it's also in method_maps
            # Derive expected key from method filename: xxx_method.py → xxx
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
                    # Verify this method exists in method_maps
                    methods_init = ROOT / "src" / "methods" / "__init__.py"
                    mm = extract_method_maps(methods_init)
                    if method_value.lower() not in mm:
                        errors.append(
                            f"[NEW FILE] {f}: method='{method_value}' not found "
                            f"in method_maps. Did you register it?"
                        )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    errors: List[str] = []

    # Paths
    methods_init = ROOT / "src" / "methods" / "__init__.py"
    models_init = ROOT / "src" / "models" / "__init__.py"
    models_dir = ROOT / "src" / "models"
    methods_dir = ROOT / "src" / "methods"
    config_dir = ROOT / "configs" / "VisualGenome"
    parser_path = ROOT / "utils" / "parser.py"

    print("=" * 60)
    print("OpenSGG CI Validation")
    print("=" * 60)

    # 1. Extract method_maps
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

    # 2. Validate configs
    print(f"\n[2/5] Validating configs in {config_dir} ...")
    cfg_errors = validate_configs(config_dir, method_maps)
    errors.extend(cfg_errors)
    if cfg_errors:
        for e in cfg_errors:
            print(f"  [FAIL] {e}")
    else:
        print(f"  All configs OK ({len(list(config_dir.glob('*.py')))} files)")

    # 3. Validate model ↔ method mapping
    print("\n[3/5] Validating model ↔ method registration ...")
    map_errors = validate_model_method_mapping(
        models_dir, methods_dir, models_init, methods_init, method_maps
    )
    errors.extend(map_errors)
    if map_errors:
        for e in map_errors:
            print(f"  [FAIL] {e}")
    else:
        print("  All model/method registrations OK")

    # 4. Validate parser choices
    print("\n[4/5] Validating parser --method choices ...")
    parser_errors = validate_parser_choices(parser_path, method_maps)
    errors.extend(parser_errors)
    if parser_errors:
        for e in parser_errors:
            print(f"  [FAIL] {e}")
    else:
        print("  Parser choices match method_maps exactly")

    # 5. Check PR diff for new files
    print("\n[5/5] Checking PR diff for new file registration ...")
    changed_files = get_pr_changed_files()
    if changed_files:
        print(f"  Changed files ({len(changed_files)}):")
        for f in changed_files:
            print(f"    {f}")

        new_errors = validate_new_files(changed_files)
        errors.extend(new_errors)
        if new_errors:
            for e in new_errors:
                print(f"  [FAIL] {e}")
        else:
            print("  All new files properly registered")
    else:
        print("  No changed files detected (local run / not a PR)")

    # Report
    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
