#!/usr/bin/env python3
"""Validate predicate_semantic_map_vg150.json against VG150 data sources.

Usage:
    python tools/analysis/validate_predicate_semantic_map.py
    python tools/analysis/validate_predicate_semantic_map.py --dry-run
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SEMANTIC_MAP_PATH = PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"
REL_JSON_PATH = PROJECT_ROOT / "data" / "VisualGenome" / "rel.json"
PRED_FREQ_PATH = PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "semantic_map_validation.json"
PREDICATES_TXT_PATH = OUTPUT_DIR / "vg150_predicates.txt"


def load_vg150_predicates():
    """Load the VG150 predicate list from the canonical data sources.

    Returns:
        predicate_names: list of predicate name strings (index = VG predicate ID).
        predicate_freqs: dict mapping VG predicate ID (str) -> train frequency.
    """
    with open(REL_JSON_PATH, "r") as f:
        rel_data = json.load(f)

    # rel_categories is a list where index = VG predicate ID
    cats = rel_data["rel_categories"]
    predicate_names = list(cats)  # copy

    with open(PRED_FREQ_PATH, "r") as f:
        freq_data = json.load(f)

    predicate_freqs = freq_data["predicate_frequencies"]

    return predicate_names, predicate_freqs


def build_predicate_set(predicate_names):
    """Build a set of predicate names (excluding __background__)."""
    return {name for name in predicate_names if name != "__background__"}


def validate_semantic_map(semantic_map, predicate_set, predicate_names, predicate_freqs):
    """Validate the semantic map against the VG150 predicate list.

    Returns:
        errors: list of error strings.
        warnings: list of warning strings.
        stats: dict of summary statistics.
    """
    errors = []
    warnings = []
    stats = {
        "total_mappings": 0,
        "strong_count": 0,
        "medium_count": 0,
        "weak_count": 0,
        "main_count": 0,
        "predicate_not_found": [],
        "parent_not_found": [],
        "missing_fields": [],
        "main_not_strong": [],
        "main_not_enabled": [],
    }

    required_fields = {"parent", "confidence", "evidence", "use_in_main"}
    valid_confidences = {"strong", "medium", "weak"}

    for pred_name, mapping in semantic_map.items():
        stats["total_mappings"] += 1

        # Check required fields
        missing = required_fields - set(mapping.keys())
        if missing:
            err = f"'{pred_name}': missing required fields: {sorted(missing)}"
            errors.append(err)
            stats["missing_fields"].append(pred_name)
            continue

        # Check predicate exists in VG150
        if pred_name not in predicate_set:
            err = f"'{pred_name}': predicate NOT FOUND in VG150 predicate list"
            errors.append(err)
            stats["predicate_not_found"].append(pred_name)

        # Check parent exists in VG150
        parent = mapping["parent"]
        if parent not in predicate_set:
            err = f"'{pred_name}': parent '{parent}' NOT FOUND in VG150 predicate list"
            errors.append(err)
            stats["parent_not_found"].append(pred_name)

        # Check confidence value
        confidence = mapping["confidence"]
        if confidence not in valid_confidences:
            err = f"'{pred_name}': invalid confidence '{confidence}' (must be: {sorted(valid_confidences)})"
            errors.append(err)

        # Count by confidence
        if confidence == "strong":
            stats["strong_count"] += 1
        elif confidence == "medium":
            stats["medium_count"] += 1
        elif confidence == "weak":
            stats["weak_count"] += 1

        # Check use_in_main
        use_in_main = mapping["use_in_main"]
        if not isinstance(use_in_main, bool):
            err = f"'{pred_name}': use_in_main must be boolean, got {type(use_in_main).__name__}"
            errors.append(err)

        # Main experiment rule: use_in_main=true implies confidence=strong
        if use_in_main and confidence != "strong":
            err = (
                f"'{pred_name}': use_in_main=true but confidence='{confidence}' "
                f"(main experiment requires confidence=strong)"
            )
            errors.append(err)
            stats["main_not_strong"].append(pred_name)

        if use_in_main:
            stats["main_count"] += 1

        # Check evidence is non-empty
        evidence = mapping.get("evidence", "")
        if not evidence or not evidence.strip():
            warnings.append(f"'{pred_name}': evidence field is empty or whitespace-only")

    # Check: every strong mapping with no errors should have use_in_main=true (consistency)
    for pred_name, mapping in semantic_map.items():
        if pred_name in stats["missing_fields"]:
            continue
        if mapping["confidence"] == "strong" and not mapping["use_in_main"]:
            warnings.append(
                f"'{pred_name}': confidence=strong but use_in_main=false "
                f"(strong mappings are expected to be in main experiment)"
            )

    return errors, warnings, stats


def generate_predicate_list_txt(predicate_names, predicate_freqs):
    """Generate vg150_predicates.txt with ID, name, and train frequency."""
    lines = []
    lines.append("# VG150 Predicate List")
    lines.append("# Format: VG_ID  predicate_name  train_frequency")
    lines.append("# Generated by validate_predicate_semantic_map.py")
    lines.append("")
    for vg_id, name in enumerate(predicate_names):
        freq = predicate_freqs.get(str(vg_id), 0)
        if name == "__background__":
            lines.append(f"{vg_id:3d}  {name:20s}  (background, excluded)")
        else:
            lines.append(f"{vg_id:3d}  {name:20s}  {freq}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Validate predicate semantic map against VG150 data sources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Predicate Semantic Map Validator")
    print("=" * 70)
    print()

    # 1. Load VG150 predicate list
    print("[1/6] Loading VG150 predicate list...")
    predicate_names, predicate_freqs = load_vg150_predicates()
    predicate_set = build_predicate_set(predicate_names)
    print(f"      Found {len(predicate_names)} predicate entries "
          f"({len(predicate_set)} actual predicates, + __background__)")
    print(f"      Sources: {REL_JSON_PATH.name}, {PRED_FREQ_PATH.name}")
    print()

    # 2. Load semantic map
    print("[2/6] Loading semantic map...")
    if not SEMANTIC_MAP_PATH.exists():
        print(f"      ERROR: Semantic map not found at {SEMANTIC_MAP_PATH}")
        sys.exit(1)

    with open(SEMANTIC_MAP_PATH, "r") as f:
        semantic_map_raw = json.load(f)

    # Separate _meta from predicate entries
    meta = semantic_map_raw.pop("_meta", {})
    semantic_map = semantic_map_raw

    print(f"      Loaded {len(semantic_map)} predicate mappings from {SEMANTIC_MAP_PATH.name}")
    print()

    # 3. Validate
    print("[3/6] Validating semantic map...")
    errors, warnings, stats = validate_semantic_map(
        semantic_map, predicate_set, predicate_names, predicate_freqs
    )
    print()

    # 4. Print results
    print("[4/6] Validation Results")
    print("-" * 70)

    # Summary counts
    print(f"  Total mappings:          {stats['total_mappings']}")
    print(f"  Strong (main experiment): {stats['strong_count']}")
    print(f"  Medium (exploratory):     {stats['medium_count']}")
    print(f"  Weak (appendix/excluded): {stats['weak_count']}")
    print(f"  use_in_main=true:         {stats['main_count']}")
    print()

    # Main experiment rule check
    if stats["main_not_strong"]:
        print(f"  FAIL: {len(stats['main_not_strong'])} mapping(s) have use_in_main=true "
              f"but confidence != strong:")
        for name in stats["main_not_strong"]:
            m = semantic_map[name]
            print(f"    - '{name}': confidence={m['confidence']}, use_in_main={m['use_in_main']}")
    else:
        print(f"  PASS: All {stats['main_count']} main mappings have confidence=strong.")

    # Predicate existence check
    if stats["predicate_not_found"]:
        print(f"  FAIL: {len(stats['predicate_not_found'])} predicate(s) not found in VG150:")
        for name in stats["predicate_not_found"]:
            print(f"    - '{name}'")
    else:
        print(f"  PASS: All {stats['total_mappings']} predicate names exist in VG150.")

    # Parent existence check
    if stats["parent_not_found"]:
        print(f"  FAIL: {len(stats['parent_not_found'])} parent(s) not found in VG150:")
        for name in stats["parent_not_found"]:
            m = semantic_map[name]
            print(f"    - '{name}' -> parent '{m['parent']}' not in VG150")
    else:
        print(f"  PASS: All parent predicates exist in VG150.")

    # Required fields check
    if stats["missing_fields"]:
        print(f"  FAIL: {len(stats['missing_fields'])} mapping(s) missing required fields:")
        for name in stats["missing_fields"]:
            print(f"    - '{name}'")
    else:
        print(f"  PASS: All mappings have required fields (parent, confidence, evidence, use_in_main).")

    # Any other errors
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")

    # Warnings
    if warnings:
        print()
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    print()

    # 5. Overall pass/fail
    print("[5/6] Overall Result")
    has_failures = bool(
        errors
        or stats["predicate_not_found"]
        or stats["parent_not_found"]
        or stats["missing_fields"]
        or stats["main_not_strong"]
    )
    if has_failures:
        print("  STATUS: FAILED")
        print(f"  {len(errors)} error(s) found.")
    else:
        print("  STATUS: PASSED")
        print("  All acceptance criteria satisfied.")

    print()

    # 6. Write output files
    print("[6/6] Writing output files...")

    validation_output = {
        "status": "PASSED" if not has_failures else "FAILED",
        "summary": {
            "total_mappings": stats["total_mappings"],
            "strong_count": stats["strong_count"],
            "medium_count": stats["medium_count"],
            "weak_count": stats["weak_count"],
            "main_count": stats["main_count"],
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "checks": {
            "all_predicates_in_vg150": len(stats["predicate_not_found"]) == 0,
            "all_parents_in_vg150": len(stats["parent_not_found"]) == 0,
            "all_required_fields_present": len(stats["missing_fields"]) == 0,
            "main_mappings_are_strong": len(stats["main_not_strong"]) == 0,
        },
        "details": {
            "predicates_not_found": stats["predicate_not_found"],
            "parents_not_found": stats["parent_not_found"],
            "missing_fields": stats["missing_fields"],
            "main_not_strong": stats["main_not_strong"],
        },
        "errors": errors,
        "warnings": warnings,
    }

    if args.dry_run:
        print("      --dry-run: skipping file writes")
        print(f"      Would write: {VALIDATION_OUTPUT_PATH}")
        print(f"      Would write: {PREDICATES_TXT_PATH}")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(VALIDATION_OUTPUT_PATH, "w") as f:
            json.dump(validation_output, f, indent=2)
        print(f"      Wrote: {VALIDATION_OUTPUT_PATH}")

        predicates_txt = generate_predicate_list_txt(predicate_names, predicate_freqs)
        with open(PREDICATES_TXT_PATH, "w") as f:
            f.write(predicates_txt)
        print(f"      Wrote: {PREDICATES_TXT_PATH}")

    print()

    # Summary of mapped predicates by confidence
    print("=" * 70)
    print("Mapped Predicates by Confidence:")
    print("=" * 70)
    for confidence in ["strong", "medium", "weak"]:
        entries = [
            (name, m)
            for name, m in semantic_map.items()
            if m.get("confidence") == confidence
        ]
        if entries:
            print(f"\n  [{confidence.upper()}] ({len(entries)} mappings):")
            for name, m in entries:
                in_main = " [MAIN]" if m.get("use_in_main") else ""
                print(f"    {name:20s} -> {m['parent']:15s}{in_main}")

    print()
    print("Done.")

    if has_failures:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
