"""Generate ``configs/predicate_canonical_map_vg150.json`` (L1_noise / L2_entail).

The output is a frozen artifact: regenerate only deliberately, and expect the
``source_sha256`` block to change if an input changes.

Usage::

    python tools/ontology_probe/build_canonical_map.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    SAMPLE_ROOT,
    read_json,
    sha256_file,
    status_block,
    write_json,
)

DEFAULT_OUTPUT = _PROJECT_ROOT / "configs" / "predicate_canonical_map_vg150.json"
POB_PATH = _PROJECT_ROOT / "configs" / "pob_strong_mappings.json"
SEMANTIC_MAP_PATH = _PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"

#: L1 merges: within-lemma surface variants only.  These make no entailment
#: claim, so any gain they produce is annotation noise being absorbed.
NOISE_MERGE_RULES: list[dict[str, Any]] = [
    {
        "members": ["laying on", "lying on"],
        "canonical": "lying on",
        "evidence": (
            "spelling variant of the same lemma (lie/lay); the two are used "
            "interchangeably in VG annotations and carry the same visual relation"
        ),
    },
    {
        "members": ["wears", "wearing"],
        "canonical": "wearing",
        "evidence": "tense variant of the same lemma; same visual relation",
    },
]

#: Pairs that look like duplicates but are deliberately NOT merged at L1, with
#: the reason, so the exclusions are auditable rather than implicit.
EXCLUDED_PAIRS: list[dict[str, str]] = [
    {
        "pair": "holding | carrying",
        "reason": "distinct actions (grasp in hand vs transport); not surface variants",
    },
    {
        "pair": "looking at | watching",
        "reason": "distinct (gaze target vs sustained attention over time)",
    },
    {
        "pair": "covering | covered in",
        "reason": "different semantic frames (active agent vs stative coverage)",
    },
    {
        "pair": "on | on back of",
        "reason": "on-back-of adds an anatomical location, so merging loses information",
    },
    {
        "pair": "of | part of | made of",
        "reason": "different meronymy frames (ownership vs composition vs material)",
    },
    {
        "pair": "behind | in front of | under | above | over | near",
        "reason": (
            "directional predicates: merging them destroys the axis information "
            "that a visual model can actually learn"
        ),
    },
]


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _load_pob_entailment_edges(
    pob: dict[str, Any],
    name_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Extract child->parent edges from the frozen POB strong mappings."""
    edges: list[dict[str, Any]] = []
    for group_key in ("primary_on_family", "boundary_in_family"):
        group = pob.get(group_key)
        if not group:
            continue
        for mapping in group["mappings"]:
            child, parent = mapping["child"], mapping["parent"]
            if child not in name_to_id or parent not in name_to_id:
                raise KeyError(f"POB mapping references unknown predicate: {mapping}")
            if name_to_id[child] != int(mapping["child_id"]):
                raise ValueError(f"POB child id mismatch for {child!r}")
            if name_to_id[parent] != int(mapping["parent_id"]):
                raise ValueError(f"POB parent id mismatch for {parent!r}")
            edges.append(
                {
                    "child": child,
                    "child_id": name_to_id[child],
                    "parent": parent,
                    "parent_id": name_to_id[parent],
                    "confidence": mapping["confidence"],
                    "group": group_key,
                    "evidence": mapping.get("evidence", ""),
                }
            )
    return edges


def _build_level(
    level: str,
    predicate_names: list[str],
    noise_rules: list[dict[str, Any]],
    entailment_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Union predicates per level, then emit classes ordered by smallest id."""
    name_to_id = {name: i for i, name in enumerate(predicate_names) if i > 0}
    fine_ids = sorted(name_to_id.values())
    uf = _UnionFind(fine_ids)

    edges: list[dict[str, Any]] = []
    for rule in noise_rules:
        ids = [name_to_id[m] for m in rule["members"]]
        for other in ids[1:]:
            uf.union(ids[0], other)
        edges.append(
            {
                "kind": "noise_only",
                "members": ids,
                "evidence": rule["evidence"],
            }
        )
    for edge in entailment_edges:
        uf.union(edge["child_id"], edge["parent_id"])
        edges.append(
            {
                "kind": "semantic_entailment",
                "members": [edge["child_id"], edge["parent_id"]],
                "group": edge["group"],
                "confidence": edge["confidence"],
                "evidence": edge["evidence"],
            }
        )

    groups: dict[int, list[int]] = {}
    for pred_id in fine_ids:
        groups.setdefault(uf.find(pred_id), []).append(pred_id)
    ordered_groups = sorted(groups.values(), key=lambda g: min(g))

    # Explicit canonical names: linguistic correctness, not frequency.  `laying
    # on` is actually the more frequent variant, but `lying on` is the lemma.
    explicit_names: dict[int, str] = {}
    for rule in noise_rules:
        explicit_names[name_to_id[rule["members"][0]]] = rule["canonical"]
    for edge in entailment_edges:
        explicit_names[edge["child_id"]] = edge["parent"]

    classes: list[dict[str, Any]] = []
    class_names: list[str] = []
    id_map: dict[int, int] = {}
    for class_idx, member_ids in enumerate(ordered_groups):
        member_names = [predicate_names[p] for p in member_ids]
        canonical = next(
            (explicit_names[p] for p in member_ids if p in explicit_names),
            None,
        )
        if canonical is None:
            canonical = predicate_names[member_ids[0]]
        if canonical not in member_names:
            raise ValueError(
                f"{level}: canonical {canonical!r} is not a member of {member_names}"
            )
        group_edges = [
            e for e in edges if any(m in member_ids for m in e["members"])
        ]
        if len(member_ids) == 1:
            kind = "singleton"
        elif any(e["kind"] == "semantic_entailment" for e in group_edges):
            kind = "semantic_entailment"
        else:
            kind = "noise_only"

        class_names.append(canonical)
        for pred_id in member_ids:
            id_map[pred_id] = class_idx
        classes.append(
            {
                "class_id": class_idx,
                "canonical": canonical,
                "kind": kind,
                "vg_ids": member_ids,
                "vg_names": member_names,
                "edges": group_edges,
            }
        )

    return {
        "classes": classes,
        "class_names": class_names,
        "id_maps": {str(k): v for k, v in sorted(id_map.items())},
    }


def build_payload(data_root: Path) -> dict[str, Any]:
    rel = read_json(data_root / "rel.json")
    predicate_names = list(rel["rel_categories"])
    pob = read_json(POB_PATH)
    entailment_edges = _load_pob_entailment_edges(
        pob, {n: i for i, n in enumerate(predicate_names)}
    )

    l1 = _build_level("L1_noise", predicate_names, NOISE_MERGE_RULES, [])
    l2 = _build_level("L2_entail", predicate_names, NOISE_MERGE_RULES, entailment_edges)

    return {
        "_meta": status_block(
            description=(
                "Canonical predicate label spaces for the VG150 cross-pair "
                "generalization diagnostic. L1_noise merges within-lemma surface "
                "variants only; L2_entail adds the frozen strong entailment "
                "mappings. Comparing Δ_ontology between the two separates "
                "annotation noise from semantic entailment."
            ),
            created_from={
                "rel.json": sha256_file(data_root / "rel.json"),
                "pob_strong_mappings.json": sha256_file(POB_PATH),
                "predicate_semantic_map_vg150.json": sha256_file(SEMANTIC_MAP_PATH),
            },
            total_vg_classes=len(predicate_names) - 1,
            predicate_names=predicate_names,
            class_counts={
                "L1_noise": len(l1["class_names"]),
                "L2_entail": len(l2["class_names"]),
            },
            excluded_pairs_note="see excluded_pairs for pairs deliberately NOT merged",
            levels=["L1_noise", "L2_entail"],
        ),
        "levels": ["L1_noise", "L2_entail"],
        "noise_merge_rules": NOISE_MERGE_RULES,
        "excluded_pairs": EXCLUDED_PAIRS,
        "entailment_edges": entailment_edges,
        "classes": {"L1_noise": l1["classes"], "L2_entail": l2["classes"]},
        "class_names": {
            "L1_noise": l1["class_names"],
            "L2_entail": l2["class_names"],
        },
        "id_maps": {"L1_noise": l1["id_maps"], "L2_entail": l2["id_maps"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dry-run", action="store_true", help="build from the sample fixture"
    )
    parser.add_argument(
        "--print-summary", action="store_true", help="do not write, only report"
    )
    args = parser.parse_args(argv)

    data_root = Path(SAMPLE_ROOT if args.dry_run else args.data_root)
    payload = build_payload(data_root)

    if not args.dry_run and not args.print_summary:
        write_json(Path(args.output), payload)
        print(f"[build_canonical_map] wrote {args.output}")

    for level in payload["levels"]:
        names = payload["class_names"][level]
        sizes = [len(c["vg_ids"]) for c in payload["classes"][level]]
        merged = {n: s for n, s in zip(names, sizes) if s > 1}
        print(
            f"[build_canonical_map] {level}: {len(names)} classes "
            f"({len(merged)} merged) {merged}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
