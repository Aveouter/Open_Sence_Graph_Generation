"""Validate ``configs/predicate_canonical_map_vg150.json``.

Checks the map structurally (classes partition the 50 predicates exactly once,
ids in range, names unique) *and* re-verifies its provenance: the two frozen
inputs it was derived from must still hash to what the config recorded, and the
entailment edges must still agree with ``configs/pob_strong_mappings.json``.
A divergence there is an error, because that file is cited as primary evidence.

Usage::

    python tools/ontology_probe/validate_canonical_map.py
    python tools/ontology_probe/validate_canonical_map.py --data-root data/VisualGenome
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.build_canonical_map import POB_PATH, SEMANTIC_MAP_PATH
from tools.ontology_probe.canonical_map import (
    DEFAULT_CANONICAL_MAP_PATH,
    load_canonical_map,
    summarise,
    validate_canonical_map,
)
from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    read_json,
    sha256_file,
    status_block,
    write_json,
)

EXPECTED_CLASS_COUNTS = {"L1_noise": 48, "L2_entail": 41}


def _check_provenance(payload: dict[str, Any], data_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    recorded = payload.get("_meta", {}).get("created_from", {})

    live = {
        "rel.json": data_root / "rel.json",
        "pob_strong_mappings.json": POB_PATH,
        "predicate_semantic_map_vg150.json": SEMANTIC_MAP_PATH,
    }
    for name, path in live.items():
        if name not in recorded:
            warnings.append(f"config does not record a hash for {name}")
            continue
        if not path.exists():
            warnings.append(f"{name} not present on disk; provenance unverified")
            continue
        actual = sha256_file(path)
        if actual != recorded[name]:
            errors.append(
                f"{name} changed since the canonical map was generated "
                f"(recorded {recorded[name][:12]}..., actual {actual[:12]}...); "
                f"regenerate with build_canonical_map.py"
            )

    # The entailment edges must still match the frozen POB file verbatim.
    pob = read_json(POB_PATH)
    expected_edges = set()
    for group_key in ("primary_on_family", "boundary_in_family"):
        for mapping in (pob.get(group_key) or {}).get("mappings", []):
            expected_edges.add(
                (mapping["child"], mapping["child_id"], mapping["parent"], mapping["parent_id"])
            )
    actual_edges = {
        (e["child"], e["child_id"], e["parent"], e["parent_id"])
        for e in payload.get("entailment_edges", [])
    }
    if expected_edges != actual_edges:
        missing = expected_edges - actual_edges
        extra = actual_edges - expected_edges
        errors.append(
            f"entailment edges disagree with pob_strong_mappings.json: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default=str(DEFAULT_CANONICAL_MAP_PATH))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    map_path = Path(args.map)
    data_root = Path(args.data_root)
    payload = read_json(map_path)
    predicate_names = payload["_meta"]["predicate_names"]

    errors, warnings = _check_provenance(payload, data_root)
    summaries: dict[str, Any] = {}

    for level in payload["levels"]:
        cmap = load_canonical_map(map_path, level)
        level_errors, level_warnings = validate_canonical_map(
            cmap, predicate_names, expected_class_count=EXPECTED_CLASS_COUNTS.get(level)
        )
        errors.extend(f"{level}: {e}" for e in level_errors)
        warnings.extend(f"{level}: {w}" for w in level_warnings)
        summaries[level] = summarise(cmap)

    report = status_block(
        map_path=str(map_path),
        expected_class_counts=EXPECTED_CLASS_COUNTS,
        summaries=summaries,
        errors=errors,
        warnings=warnings,
        passed=not errors,
    )
    if args.output_dir:
        write_json(Path(args.output_dir) / "canonical_map_validation.json", report)

    for level, summary in summaries.items():
        print(
            f"[validate_canonical_map] {level}: {summary['n_classes']} classes, "
            f"{summary['n_predicates_merged']} predicates merged, "
            f"{summary['n_predicates_untouched']} untouched"
        )
    for warning in warnings:
        print(f"[validate_canonical_map] WARN  {warning}")
    for error in errors:
        print(f"[validate_canonical_map] ERROR {error}")
    print(
        "[validate_canonical_map] "
        + ("PASSED" if not errors else f"FAILED ({len(errors)} errors)")
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
