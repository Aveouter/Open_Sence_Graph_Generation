"""Step 3: build the pair-OOD / pair-known splits and audit them for leakage.

Writes index-based split files.  Storing *indices* into a deterministic relation
table (rather than repeating 315k tuples as JSON) keeps the artifacts loadable;
``relation_order_sha256`` guards against the table being rebuilt in a different
order, which would silently misalign every downstream subset.

Usage::

    python tools/ontology_probe/build_splits.py --data-root data/VisualGenome
    python tools/ontology_probe/build_splits.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SAMPLE_ROOT,
    STUDY_PREDICATES,
    sha256_file,
    status_block,
    write_json,
)
from tools.ontology_probe.ood_split import (
    OODSplitConfig,
    SplitBundle,
    audit_split,
    build_pair_known_split,
    build_pair_ood_split,
    compute_unseen_subset,
)
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    build_relation_table,
    load_vg_index,
)

DEFAULT_CONFIG = OODSplitConfig()
MIN_PREDICATE_SUPPORT = 20
MIN_EVAL_SUPPORT = 50

__all__ = ["relation_order_sha256", "bundle_to_payload", "identify_degenerate_predicates"]


def relation_order_sha256(table: RelationTable) -> str:
    """Hash the relation sequence so a rebuilt table can be compared."""
    digest = hashlib.sha256()
    for key in table.relation_keys():
        digest.update(f"{key[0]}:{key[1]}:{key[2]}:{key[3]};".encode())
    return digest.hexdigest()


def identify_degenerate_predicates(
    table: RelationTable,
    bundle: SplitBundle,
    predicate_names: Sequence[str],
    min_train: int = MIN_PREDICATE_SUPPORT,
    min_eval: int = MIN_EVAL_SUPPORT,
) -> dict[str, str]:
    """Predicates too rare to support a per-predicate claim, with the reason."""
    train_counts: dict[int, int] = {}
    for row in bundle.train:
        train_counts[table.pred_id[row]] = train_counts.get(table.pred_id[row], 0) + 1
    eval_counts: dict[int, int] = {}
    for row in bundle.eval:
        eval_counts[table.pred_id[row]] = eval_counts.get(table.pred_id[row], 0) + 1

    excluded: dict[str, str] = {}
    for pred_id in sorted(set(table.pred_id)):
        name = predicate_names[pred_id]
        n_train = train_counts.get(pred_id, 0)
        n_eval = eval_counts.get(pred_id, 0)
        reasons = []
        if n_train < min_train:
            reasons.append(f"n_train={n_train} < {min_train}")
        if n_eval < min_eval:
            reasons.append(f"n_eval={n_eval} < {min_eval}")
        if reasons:
            excluded[name] = "; ".join(reasons)
    return excluded


def _rows_by_predicate(table: RelationTable, rows: Sequence[int]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[table.pred_id[row]] = counts.get(table.pred_id[row], 0) + 1
    return counts


def class_support(
    table: RelationTable,
    rows: Sequence[int],
    cmap: Any,
) -> dict[str, int]:
    """Class-level counts in one label space.

    Support must be measured per label space, not once in VG50: at L2_entail the
    on-family children fold into ``on``, so classes like ``lying on`` that look
    under-supported in VG50 are not classes at all there.  Filtering on VG50
    counts and then macro-averaging across spaces would compare different class
    sets and confound Δ_ontology.
    """
    counts: dict[int, int] = {}
    for row in rows:
        class_idx = cmap.remap(table.pred_id[row])
        counts[class_idx] = counts.get(class_idx, 0) + 1
    return {cmap.class_names[c]: n for c, n in sorted(counts.items())}


def excluded_classes(
    train_support: dict[str, int],
    eval_support: dict[str, int],
    min_train: int = MIN_PREDICATE_SUPPORT,
    min_eval: int = MIN_EVAL_SUPPORT,
) -> dict[str, str]:
    """Classes too rare for a per-class claim, with the reason."""
    excluded: dict[str, str] = {}
    for name in sorted(set(train_support) | set(eval_support)):
        reasons = []
        n_train = train_support.get(name, 0)
        n_eval = eval_support.get(name, 0)
        if n_train < min_train:
            reasons.append(f"n_train={n_train} < {min_train}")
        if n_eval < min_eval:
            reasons.append(f"n_eval={n_eval} < {min_eval}")
        if reasons:
            excluded[name] = "; ".join(reasons)
    return excluded


def _label_space_support(
    table: RelationTable,
    bundle: SplitBundle,
    canonical_maps: dict[str, Any] | None,
    min_train: int,
    min_eval: int,
) -> dict[str, Any]:
    """Per-label-space class support and degeneracy for this split."""
    if not canonical_maps:
        return {}
    out: dict[str, Any] = {}
    for level, cmap in canonical_maps.items():
        train_support = class_support(table, bundle.train, cmap)
        eval_support = class_support(table, bundle.eval, cmap)
        excluded = excluded_classes(train_support, eval_support, min_train, min_eval)
        out[level] = {
            "n_classes": cmap.n_classes,
            "n_classes_usable": cmap.n_classes - len(excluded),
            "train_class_counts": train_support,
            "eval_class_counts": eval_support,
            "excluded_classes": excluded,
        }

    # Macro metrics are only comparable between two spaces when both are scored
    # over the same class set.  Comparisons are therefore pairwise and always
    # pool the finer space's probabilities onto the coarser space's classes
    # (fine->coarse summation is exact for argmax and NLL; the reverse is not
    # defined). Supporting this is why support is recorded per space rather than
    # once in VG50: the on-family children are separate VG50 classes but a single
    # L2 class, so a fixed VG50 threshold would filter different class sets.
    out["comparability_rule"] = (
        "pairwise comparisons pool the finer space onto the coarser space's "
        "classes and use the coarser space's usable-class set; micro accuracy "
        "and NLL are comparable for any pair, macro accuracy only after pooling"
    )
    return out


def bundle_to_payload(
    bundle: SplitBundle,
    table: RelationTable,
    predicate_names: Sequence[str],
    order_sha: str,
    config: OODSplitConfig,
    min_train: int = MIN_PREDICATE_SUPPORT,
    min_eval: int = MIN_EVAL_SUPPORT,
    canonical_maps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    degenerate = identify_degenerate_predicates(
        table, bundle, predicate_names, min_train, min_eval
    )
    excluded = set(degenerate)
    study_eval_counts: dict[str, int] = {}
    pred_counts = _rows_by_predicate(table, bundle.eval)
    for pred_id, count in sorted(pred_counts.items()):
        name = predicate_names[pred_id]
        if name in STUDY_PREDICATES:
            study_eval_counts[name] = count

    return {
        "_meta": status_block(
            split_id=bundle.split_id,
            config=config.as_dict(),
            relation_order_sha256=order_sha,
            min_predicate_support=min_train,
            min_eval_support=min_eval,
            note=(
                "relations are [image_id, sub_idx, obj_idx, pred_id] index tuples "
                "into the deterministic relation table built from rel.json; "
                "relation_order_sha256 must match before these indices are used"
            ),
        ),
        "split_id": bundle.split_id,
        "counts": {
            "n_train": len(bundle.train),
            "n_dev": len(bundle.dev),
            "n_eval": len(bundle.eval),
            "n_images_train": len({table.image_id[r] for r in bundle.train}),
            "n_images_eval": len({table.image_id[r] for r in bundle.eval}),
            "n_held_out_pairs": len(bundle.held_out_pairs),
            "n_dropped_in_affected_images": bundle.dropped_in_affected_images,
        },
        "pair_policy": {
            "study_predicates": [
                p for p in STUDY_PREDICATES if p not in excluded
            ],
            "study_predicates_excluded": [
                p for p in STUDY_PREDICATES if p in excluded
            ],
            "held_out_pairs": [list(p) for p in bundle.held_out_pairs],
            "excluded_predicates": degenerate,
            "eval_predicate_counts": {
                predicate_names[p]: c for p, c in sorted(pred_counts.items())
            },
            "study_predicate_eval_counts": study_eval_counts,
            "excluded_predicates_vg50": degenerate,
        },
        "label_spaces": _label_space_support(table, bundle, canonical_maps, min_train, min_eval),
        "notes": bundle.notes,
        "train": bundle.train,
        "dev": bundle.dev,
        "eval": bundle.eval,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "splits"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pair-count", type=int, default=DEFAULT_CONFIG.max_pair_count)
    parser.add_argument("--min-pair-count", type=int, default=DEFAULT_CONFIG.min_pair_count)
    parser.add_argument(
        "--image-cap-fraction", type=float, default=DEFAULT_CONFIG.image_cap_fraction
    )
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_CONFIG.dev_fraction)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument(
        "--with-test-natural",
        action="store_true",
        help="also compute the zero-damage confirmation set on the official test split",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_true",
        help="skip per-label-space class support (vg50 only)",
    )
    args = parser.parse_args(argv)

    config = OODSplitConfig(
        max_pair_count=args.max_pair_count,
        min_pair_count=args.min_pair_count,
        image_cap_fraction=args.image_cap_fraction,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    data_root = Path(SAMPLE_ROOT if args.dry_run else args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_splits] data_root={data_root} config={config.as_dict()}")
    index = load_vg_index(data_root, "train")
    table = build_relation_table(index)
    predicate_names = index.predicate_names
    name_to_id = {n: i for i, n in enumerate(predicate_names) if i > 0}
    study_ids = [name_to_id[p] for p in STUDY_PREDICATES if p in name_to_id]
    order_sha = relation_order_sha256(table)

    canonical_maps: dict[str, Any] = {}
    if not args.no_canonical:
        from tools.ontology_probe.canonical_map import identity_map, load_canonical_map

        canonical_maps["vg50"] = identity_map(predicate_names)
        for level in ("L1_noise", "L2_entail"):
            canonical_maps[level] = load_canonical_map(level=level)
        print(f"[build_splits] label spaces: {sorted(canonical_maps)}")

    bundles: dict[str, SplitBundle] = {}
    audits: dict[str, Any] = {}
    payloads: dict[str, Any] = {}

    for bundle in (
        build_pair_ood_split(table, index, predicate_names, study_ids, config),
        build_pair_known_split(table, index, predicate_names, config),
    ):
        audit = audit_split(table, bundle, predicate_names)
        audits[bundle.split_id] = audit
        bundles[bundle.split_id] = bundle
        payload = bundle_to_payload(
            bundle, table, predicate_names, order_sha, config,
            canonical_maps=canonical_maps,
        )
        payload["unseen_subset"] = compute_unseen_subset(table, bundle)
        payloads[bundle.split_id] = payload
        write_json(output_dir / f"split_{bundle.split_id}.json", payload)
        print(
            f"[build_splits] {bundle.split_id}: train={len(bundle.train)} "
            f"dev={len(bundle.dev)} eval={len(bundle.eval)} "
            f"held_out_pairs={len(bundle.held_out_pairs)} AUDIT PASSED"
        )
        study_counts = payload["pair_policy"]["study_predicate_eval_counts"]
        print(f"[build_splits]   study-predicate eval support: {study_counts}")
        if payload["pair_policy"]["excluded_predicates"]:
            print(
                f"[build_splits]   excluded predicates: "
                f"{payload['pair_policy']['excluded_predicates']}"
            )

    # Free the train-side structures before touching the test split.
    del table, index
    gc.collect()

    if args.with_test_natural:
        # The pair_ood train pair set was computed against the released train
        # table, so recompute it here and drop it again before loading test.
        train_index = load_vg_index(data_root, "train")
        train_table = build_relation_table(train_index)
        seen_pairs = {
            train_table.undirected_pairs()[r] for r in payloads["pair_ood"]["train"]
        }
        del train_index, train_table
        gc.collect()

        test_index = load_vg_index(data_root, "test")
        test_table = build_relation_table(test_index)
        test_pairs = test_table.undirected_pairs()
        natural = [r for r in range(len(test_table)) if test_pairs[r] not in seen_pairs]
        test_names = test_index.predicate_names
        counts: dict[str, int] = {}
        for row in natural:
            name = test_names[test_table.pred_id[row]]
            counts[name] = counts.get(name, 0) + 1
        write_json(
            output_dir / "split_ood_test_natural.json",
            {
                "_meta": status_block(
                    split_id="ood_test_natural",
                    relation_order_sha256=relation_order_sha256(test_table),
                    purpose=(
                        "official test relations whose pair is unseen in the "
                        "pair_ood train subset; zero training damage, used as "
                        "secondary confirmation"
                    ),
                ),
                "split_id": "ood_test_natural",
                "counts": {"n_eval": len(natural)},
                "eval": natural,
                "predicate_counts": dict(sorted(counts.items())),
            },
        )
        print(
            f"[build_splits] ood_test_natural: {len(natural)} eval relations "
            f"(pair unseen in pair_ood train)"
        )

    write_json(output_dir / "split_audit.json", status_block(audits=audits))
    write_json(
        output_dir / "splits_index.json",
        status_block(
            relation_order_sha256=order_sha,
            config=config.as_dict(),
            files={
                name: {
                    "path": f"split_{name}.json",
                    "sha256": sha256_file(output_dir / f"split_{name}.json"),
                    "split_id": name,
                }
                for name in payloads
            },
            data_root=str(data_root),
            source_sha256={
                "train.json": sha256_file(data_root / "train.json"),
                "rel.json": sha256_file(data_root / "rel.json"),
            },
        ),
    )
    print(f"[build_splits] wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
