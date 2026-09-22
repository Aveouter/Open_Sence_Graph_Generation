"""The ablation matrix: {label space} x {split} x {probe}.

``--stage prior`` runs the GPU-free half (B0, B1_lookup) and is the point at
which the visual half can still be re-scoped: if a pair lookup already explains
the pair-OOD eval set, more features will not answer the research question.

Comparability rules enforced here:

* the prior for every cell is built from that split's own train subset, and the
  IID cell trains on the *same* reduced subset as pair-OOD so the two differ
  only in what they are evaluated on;
* macro metrics between two label spaces are computed over the coarser space's
  usable classes, with the finer space's probabilities pooled onto them
  (fine->coarse summation is exact for argmax and NLL);
* every cell also reports ``native`` (its own label space) alongside ``pooled``.

Usage::

    python tools/ontology_probe/eval_matrix.py --stage prior --data-root data/VisualGenome
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.canonical_map import (
    CanonicalMap,
    identity_map,
    load_canonical_map,
)
from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    hbt_group,
    sha256_file,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.prior_baselines import (
    B0Predictor,
    B1LookupPredictor,
    build_prior_table,
    calibrate_prior,
    conflict_mask,
    pair_ambiguity,
)
from tools.ontology_probe.probe_metrics import summarise
from tools.ontology_probe.splits import load_split
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    build_relation_table,
    load_vg_index,
)

MIN_MACRO_SUPPORT = 50
LEVELS = ("vg50", "L1_noise", "L2_entail")


def usable_classes(
    labels: Sequence[int], n_classes: int, min_support: int = MIN_MACRO_SUPPORT
) -> list[int]:
    """Classes with at least ``min_support`` eval rows.

    Defined on the *eval* set actually being scored, so the macro denominator is
    a property of the measurement rather than of a build-time artifact.
    """
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(c for c in range(n_classes) if counts.get(c, 0) >= min_support)


def _names_for(cmap: CanonicalMap) -> list[str]:
    return list(cmap.class_names)


def _group_map(cmap: CanonicalMap) -> dict[str, str]:
    """Class name -> VG head/body/tail, taken from the class's finest member."""
    return {
        cmap.class_names[i]: hbt_group(min(cmap.members[i]))
        for i in range(cmap.n_classes)
    }


def evaluate_predictor(
    predictor: Any,
    sub_table: RelationTable,
    rows: Sequence[int],
    labels: Sequence[int],
    cmap: CanonicalMap,
) -> dict[str, Any]:
    """Score one predictor in one label space."""
    probs = predictor.predict_probs(sub_table, rows)
    classes = usable_classes(labels, cmap.n_classes)
    report = summarise(
        labels,
        probs,
        classes=classes,
        class_names=_names_for(cmap),
        class_to_group=_group_map(cmap),
    )
    report["n_usable_classes"] = len(classes)
    report["n_total_classes"] = cmap.n_classes
    report["predictor"] = predictor.describe()
    return report


def _pool_to(
    probs: Sequence[Sequence[float]],
    src: CanonicalMap,
    dst: CanonicalMap,
) -> list[list[float]]:
    """Pool a finer space's probabilities onto a coarser space's classes."""
    return [dst.pool_probs(row) for row in probs]


def run_split(
    split_name: str,
    train_table: RelationTable,
    train_rows: Sequence[int],
    dev_table: RelationTable,
    dev_rows: Sequence[int],
    eval_table: RelationTable,
    eval_rows: Sequence[int],
    maps: dict[str, CanonicalMap],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every probe x label-space cell for one split."""
    cells: list[dict[str, Any]] = []
    subsets: dict[str, Any] = {}

    for level in LEVELS:
        cmap = maps[level]
        prior = build_prior_table(
            train_table, train_rows, cmap=cmap, source_split_id=split_name
        )
        dev_labels = [cmap.remap(dev_table.pred_id[r]) for r in dev_rows]
        eval_labels = [cmap.remap(eval_table.pred_id[r]) for r in eval_rows]

        calibration = calibrate_prior(prior, dev_table, dev_rows, dev_labels)
        probes: list[Any] = [
            B0Predictor(prior, temperature=calibration["temperature"]),
            B1LookupPredictor(
                prior,
                alpha=calibration["alpha"],
                temperature=calibration["temperature"],
            ),
        ]
        for probe in probes:
            report = evaluate_predictor(probe, eval_table, eval_rows, eval_labels, cmap)
            cells.append(
                {
                    "cell": f"{level}__{split_name}__{probe.name}",
                    "level": level,
                    "split": split_name,
                    "probe": probe.name,
                    "backend": "prior",
                    "calibration": {
                        "alpha": calibration["alpha"],
                        "temperature": calibration["temperature"],
                        "dev_nll": calibration["dev_nll"],
                    },
                    "metrics": report,
                }
            )

        # C_unseen / C_prior_conflict live in the fine space, then get remapped.
        conflict = conflict_mask(
            prior, eval_table, eval_rows, eval_labels, alpha=calibration["alpha"]
        )
        subsets[level] = {
            "n_eval": len(eval_rows),
            "n_usable_classes": len(usable_classes(eval_labels, cmap.n_classes)),
            "n_prior_conflict": sum(conflict),
            "prior_conflict_row_indices": [
                eval_rows[i] for i, flag in enumerate(conflict) if flag
            ],
            "prior_ambiguity": pair_ambiguity(prior),
            "prior_source_split_id": prior.source_split_id,
            "n_prior_train_relations": prior.n_relations,
        }

    # Pooled comparison: the VG50 predictor scored on the coarser spaces.
    vg50 = maps["vg50"]
    vg50_prior = build_prior_table(
        train_table, train_rows, cmap=vg50, source_split_id=split_name
    )
    calibration = calibrate_prior(
        vg50_prior,
        dev_table,
        dev_rows,
        [vg50.remap(dev_table.pred_id[r]) for r in dev_rows],
    )
    vg50_probe = B1LookupPredictor(
        vg50_prior, alpha=calibration["alpha"], temperature=calibration["temperature"]
    )
    vg50_probs = vg50_probe.predict_probs(eval_table, eval_rows)

    # Each vg50 class is a singleton, so resolve a vg50 class index back to its
    # fine predicate and then into the coarse space. Going through members()
    # rather than assuming identity keeps this correct if vg50 ever changes.
    fine_ids = [eval_table.pred_id[r] for r in eval_rows]
    for level in ("L1_noise", "L2_entail"):
        coarse = maps[level]
        pooled = _pool_to(vg50_probs, vg50, coarse)
        target_labels = [
            coarse.remap(vg50.members[vg50.remap(pred_id)][0]) for pred_id in fine_ids
        ]
        classes = usable_classes(target_labels, coarse.n_classes)
        report = summarise(
            target_labels,
            pooled,
            classes=classes,
            class_names=_names_for(coarse),
            class_to_group=_group_map(coarse),
        )
        report["n_usable_classes"] = len(classes)
        report["n_total_classes"] = coarse.n_classes
        report["predictor"] = {
            "probe": "B1_lookup",
            "pooled_from": "vg50",
            "alpha": calibration["alpha"],
            "temperature": calibration["temperature"],
            "source_split_id": split_name,
        }
        cells.append(
            {
                "cell": f"{level}__{split_name}__B1_lookup_pooled_from_vg50",
                "level": level,
                "split": split_name,
                "probe": "B1_lookup",
                "backend": "prior",
                "pooled_from": "vg50",
                "calibration": {
                    "alpha": calibration["alpha"],
                    "temperature": calibration["temperature"],
                },
                "metrics": report,
            }
        )
    return cells, subsets


def _flat_row(cell: dict[str, Any]) -> dict[str, Any]:
    metrics = cell["metrics"]
    return {
        "cell": cell["cell"],
        "level": cell["level"],
        "split": cell["split"],
        "probe": cell["probe"],
        "backend": cell["backend"],
        "pooled_from": cell.get("pooled_from", ""),
        "n_eval": metrics["n"],
        "n_usable_classes": metrics["n_usable_classes"],
        "n_total_classes": metrics["n_total_classes"],
        "accuracy": metrics["accuracy"],
        "macro_recall": metrics["macro_recall"],
        "nll": metrics["nll"],
        "mR@1": metrics.get("mR@1"),
        "mR@3": metrics.get("mR@3"),
        "mR@5": metrics.get("mR@5"),
        "head_mR@1": (metrics.get("hbt") or {}).get("head"),
        "body_mR@1": (metrics.get("hbt") or {}).get("body"),
        "tail_mR@1": (metrics.get("hbt") or {}).get("tail"),
        "alpha": cell["calibration"].get("alpha"),
        "temperature": cell["calibration"].get("temperature"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "matrix"))
    parser.add_argument("--splits-dir", default=str(DEFAULT_OUTPUT_ROOT / "splits"))
    parser.add_argument("--stage", choices=["prior"], default="prior")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_vg_index(data_root, "train")
    table = build_relation_table(index)
    maps: dict[str, CanonicalMap] = {"vg50": identity_map(index.predicate_names)}
    for level in ("L1_noise", "L2_entail"):
        maps[level] = load_canonical_map(level=level)

    ood = load_split("pair_ood", args.splits_dir, table=table, verify=True)
    known = load_split("pair_known", args.splits_dir, table=table, verify=True)
    print(
        f"[eval_matrix] pair_ood train={len(ood.rows('train'))} eval={len(ood.rows('eval'))}; "
        f"pair_known eval={len(known.rows('eval'))}"
    )

    val_index = load_vg_index(data_root, "val")
    val_table = build_relation_table(val_index)
    val_rows = list(range(len(val_table)))
    print(f"[eval_matrix] iid(val) relations={len(val_rows)}")

    all_cells: list[dict[str, Any]] = []
    all_subsets: dict[str, Any] = {}

    # Controlled comparison: both splits train on the SAME reduced subset, so a
    # difference between them is the eval distribution, not the training volume.
    for split_name, eval_table, eval_rows in (
        ("iid", val_table, val_rows),
        ("pair_ood", table, ood.rows("eval")),
        ("pair_known", table, known.rows("eval")),
    ):
        cells, subsets = run_split(
            split_name,
            train_table=table,
            train_rows=ood.rows("train"),
            dev_table=table,
            dev_rows=ood.rows("dev"),
            eval_table=eval_table,
            eval_rows=eval_rows,
            maps=maps,
        )
        all_cells.extend(cells)
        all_subsets[split_name] = subsets
        for cell in cells:
            m = cell["metrics"]
            print(
                f"[eval_matrix] {cell['cell']:52s} n={m['n']:6d} "
                f"acc={m['accuracy']:.4f} macro={m['macro_recall']:.4f} nll={m['nll']:.4f}"
            )

    write_json(
        output_dir / "prior_cells.json",
        status_block(
            stage="prior",
            cells=all_cells,
            controlled_training_subset="pair_ood.train",
            eval_subsets=all_subsets,
            data_root=str(data_root),
            source_sha256={
                "train.json": sha256_file(data_root / "train.json"),
                "val.json": sha256_file(data_root / "val.json"),
                "rel.json": sha256_file(data_root / "rel.json"),
            },
        ),
    )
    write_csv(
        output_dir / "prior_cells.csv",
        [
            "cell", "level", "split", "probe", "backend", "pooled_from",
            "n_eval", "n_usable_classes", "n_total_classes",
            "accuracy", "macro_recall", "nll", "mR@1", "mR@3", "mR@5",
            "head_mR@1", "body_mR@1", "tail_mR@1", "alpha", "temperature",
        ],
        [_flat_row(c) for c in all_cells],
    )
    write_json(output_dir / "eval_subsets.json", status_block(subsets=all_subsets))

    print(f"[eval_matrix] wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
