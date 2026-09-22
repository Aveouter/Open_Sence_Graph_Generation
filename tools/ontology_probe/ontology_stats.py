"""Step 1: predicate / ontology statistics for VG150.

Answers "how much can an object-pair prior already explain, per predicate?"
before any GPU time is spent:

* frequency, distinct subject-object pair counts, and how many of those pairs
  carry more than one predicate (the genuinely ambiguous ones);
* ``H(R)``, ``H(R | c_s)``, ``H(R | c_o)``, ``H(R | c_s, c_o)`` and the
  corresponding mutual informations, plugin *and* Miller-Madow corrected.

The Miller-Madow variant matters here: 9,783 directed pairs spread over 315,642
train relations makes the plugin conditional entropy biased downward, and
``H(R | pair)`` is exactly the number the whole diagnostic leans on.

Stdlib only.  Usage::

    python tools/ontology_probe/ontology_stats.py --data-root data/VisualGenome \
        --split train --write-predicate-frequencies
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    NUM_VG_OBJECT_SLOTS,
    SAMPLE_ROOT,
    STUDY_PREDICATES,
    hbt_group,
    sha256_file,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    VGIndex,
    build_relation_table,
    load_vg_index,
    validate_index_semantics,
)

LOG2 = math.log(2.0)

__all__ = [
    "entropy_bits",
    "miller_madow_bits",
    "conditional_entropy_bits",
    "compute_dataset_stats",
    "compute_predicate_stats",
    "compute_pair_ambiguity",
    "predicate_frequencies_payload",
]


def entropy_bits(counts: Iterable[float]) -> float:
    """Shannon entropy of a count vector, in bits."""
    values = [c for c in counts if c > 0]
    total = math.fsum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    return -math.fsum((c / total) * math.log(c / total, 2) for c in values)


def miller_madow_bits(counts: Iterable[float]) -> float:
    """Plugin entropy plus the Miller-Madow finite-sample bias correction.

    ``H_MM = H_plugin + (K_observed - 1) / (2 * N * ln 2)``.  The correction is
    always non-negative, which is what makes it the conservative choice when
    reporting ``H(R | pair)``.
    """
    values = [c for c in counts if c > 0]
    total = math.fsum(values)
    if total <= 0:
        return 0.0
    corrections = (len(values) - 1) / (2.0 * total * LOG2)
    return entropy_bits(values) + corrections


def conditional_entropy_bits(
    groups: Mapping[Any, Mapping[int, float]],
) -> tuple[float, float]:
    """Size-weighted ``H(Y | X)`` over ``X`` groups, as (plugin, Miller-Madow)."""
    total = math.fsum(math.fsum(g.values()) for g in groups.values())
    if total <= 0:
        return 0.0, 0.0
    plugin = 0.0
    corrected = 0.0
    for counts in groups.values():
        weight = math.fsum(counts.values()) / total
        if weight <= 0:
            continue
        plugin += weight * entropy_bits(counts.values())
        corrected += weight * miller_madow_bits(counts.values())
    return plugin, corrected


def _group_by(
    keys: Iterable[int],
    predicate_ids: Iterable[int],
) -> dict[int, collections.Counter]:
    groups: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for key, pred in zip(keys, predicate_ids, strict=True):
        groups[key][pred] += 1
    return dict(groups)


def _argmax_predicate(counts: Mapping[int, float]) -> int:
    # Ties break on the smallest predicate id so results are deterministic
    # regardless of dict insertion order.
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def compute_dataset_stats(table: RelationTable, index: VGIndex) -> dict[str, Any]:
    """Corpus-level entropy and ambiguity summary for one split."""
    preds = list(table.pred_id)
    n = len(preds)
    if n == 0:
        raise ValueError("empty relation table")

    h_r = entropy_bits(collections.Counter(preds).values())
    h_r_mm = miller_madow_bits(collections.Counter(preds).values())

    by_cs = _group_by(table.c_s, preds)
    by_co = _group_by(table.c_o, preds)
    by_pair = _group_by(table.directed_pairs(), preds)
    h_cs, h_cs_mm = conditional_entropy_bits(by_cs)
    h_co, h_co_mm = conditional_entropy_bits(by_co)
    h_pair, h_pair_mm = conditional_entropy_bits(by_pair)

    directed = table.directed_pairs()
    undirected = table.undirected_pairs()
    pair_counts = collections.Counter(directed)
    undirected_counts = collections.Counter(undirected)
    preds_per_undirected: dict[int, set[int]] = collections.defaultdict(set)
    for i in range(n):
        preds_per_undirected[undirected[i]].add(preds[i])

    support_histogram = collections.Counter()
    for count in pair_counts.values():
        bucket = "1" if count == 1 else "2-5" if count <= 5 else "6-20" if count <= 20 else "21-100" if count <= 100 else "100+"
        support_histogram[bucket] += 1

    hbt_counts: collections.Counter = collections.Counter(
        hbt_group(p) for p in set(preds)
    )

    return status_block(
        split=index.split,
        n_images=len(index.image_ids),
        n_images_with_relations=len({i for i in table.image_id}),
        n_objects=index.n_objects(),
        n_relations=n,
        n_distinct_directed_pairs=len(pair_counts),
        n_distinct_undirected_pairs=len(undirected_counts),
        n_undirected_pairs_multi_relation=sum(
            1 for v in preds_per_undirected.values() if len(v) > 1
        ),
        pair_support_histogram=dict(support_histogram),
        relations_in_singleton_pairs=sum(
            c for p, c in pair_counts.items() if c == 1
        ),
        H_R_bits=h_r,
        H_R_bits_miller_madow=h_r_mm,
        H_R_given_c_s_bits=h_cs,
        H_R_given_c_s_bits_miller_madow=h_cs_mm,
        H_R_given_c_o_bits=h_co,
        H_R_given_c_o_bits_miller_madow=h_co_mm,
        H_R_given_pair_bits=h_pair,
        H_R_given_pair_bits_miller_madow=h_pair_mm,
        I_R_c_s_bits=h_r - h_cs,
        I_R_c_o_bits=h_r - h_co,
        I_R_pair_bits=h_r - h_pair,
        n_distinct_subject_classes=len(by_cs),
        n_distinct_object_classes=len(by_co),
        hbt_predicate_counts=dict(hbt_counts),
        interpretation=(
            "H_R_given_pair near 0 means object identity alone nearly determines "
            "the predicate, so a pair prior can win without any visual evidence; "
            "I_R_pair is the maximum a pair-only model could explain."
        ),
    )


def compute_predicate_stats(table: RelationTable, index: VGIndex) -> list[dict[str, Any]]:
    """One diagnostic row per predicate."""
    n = len(table)
    preds = table.pred_id
    directed = table.directed_pairs()
    undirected = table.undirected_pairs()

    pred_counts = collections.Counter(preds)
    pair_pred_counts = _group_by(directed, preds)
    undirected_pred_sets: dict[int, set[int]] = collections.defaultdict(set)
    for i in range(n):
        undirected_pred_sets[undirected[i]].add(preds[i])

    # Rows grouped by predicate, so per-predicate means need one pass each.
    rows_by_pred: dict[int, list[int]] = collections.defaultdict(list)
    for i, pred in enumerate(preds):
        rows_by_pred[pred].append(i)

    names = index.object_names
    out: list[dict[str, Any]] = []
    for pred_id in sorted(pred_counts):
        rows = rows_by_pred[pred_id]
        count = len(rows)
        pair_counts = collections.Counter(directed[i] for i in rows)
        ents = [entropy_bits(pair_pred_counts[directed[i]].values()) for i in rows]
        conflicts = sum(
            1
            for i in rows
            if _argmax_predicate(pair_pred_counts[directed[i]]) != pred_id
        )
        ambiguous = sum(1 for i in rows if len(undirected_pred_sets[undirected[i]]) > 1)
        shared = sum(
            1 for pair in pair_counts if len(pair_pred_counts[pair]) > 1
        )
        top_pairs = pair_counts.most_common(5)
        name = index.predicate_name(pred_id)
        out.append(
            status_block(
                predicate=name,
                predicate_id=pred_id,
                hbt=hbt_group(pred_id),
                frequency=count,
                frequency_share=count / n,
                n_distinct_pairs=len(pair_counts),
                n_distinct_pairs_shared_with_other_predicates=shared,
                mean_pair_prior_entropy_bits=math.fsum(ents) / len(ents) if ents else 0.0,
                ambiguity_mass=ambiguous / count,
                self_prior_conflict_rate=conflicts / count,
                is_study_predicate=name in STUDY_PREDICATES,
                top_pairs=[
                    {
                        "c_s": p // NUM_VG_OBJECT_SLOTS,
                        "c_o": p % NUM_VG_OBJECT_SLOTS,
                        "subject": names.get(p // NUM_VG_OBJECT_SLOTS, "?"),
                        "object": names.get(p % NUM_VG_OBJECT_SLOTS, "?"),
                        "count": c,
                    }
                    for p, c in top_pairs
                ],
                caveat=(
                    "mean_pair_prior_entropy_bits and self_prior_conflict_rate are "
                    "computed on the same table they describe (in-sample) and are "
                    "therefore optimistic; the out-of-sample analogue is produced "
                    "by the split/prior stage."
                ),
            )
        )
    return out


def compute_pair_ambiguity(
    table: RelationTable,
    index: VGIndex,
    min_support: int = 20,
    limit: int | None = 500,
) -> list[dict[str, Any]]:
    """Per-pair ambiguity rows, most ambiguous first."""
    preds = table.pred_id
    directed = table.directed_pairs()
    groups = _group_by(directed, preds)
    names = index.object_names

    rows: list[dict[str, Any]] = []
    for pair, counts in groups.items():
        n_rel = sum(counts.values())
        if n_rel < min_support:
            continue
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append(
            {
                "c_s": pair // NUM_VG_OBJECT_SLOTS,
                "c_o": pair % NUM_VG_OBJECT_SLOTS,
                "subject": names.get(pair // NUM_VG_OBJECT_SLOTS, "?"),
                "object": names.get(pair % NUM_VG_OBJECT_SLOTS, "?"),
                "n_relations": n_rel,
                "n_predicates": len(counts),
                "H_R_given_pair_bits": entropy_bits(counts.values()),
                "H_R_given_pair_bits_miller_madow": miller_madow_bits(counts.values()),
                "top_predicate": index.predicate_name(top[0][0]),
                "top_predicate_share": top[0][1] / n_rel,
                "all_predicates": "|".join(
                    f"{index.predicate_name(p)}:{c}" for p, c in top[:8]
                ),
            }
        )
    rows.sort(key=lambda r: (-r["H_R_given_pair_bits"], -r["n_relations"]))
    return rows[:limit] if limit else rows


def predicate_frequencies_payload(
    table: RelationTable,
    index: VGIndex,
) -> dict[str, Any]:
    """Schema consumed by src/core/metrics.py::compute_head_body_tail_mr and friends.

    ``predicate_names`` is indexed by predicate id, so it keeps all 51 entries
    with ``__background__`` at index 0.  ``predicate_frequencies`` deliberately
    omits id 0: background never appears as a ground-truth relation, and the
    head/body/tail ranking in ``compute_head_body_tail_mr`` ranks every entry it
    is given, so a zero-count background would consume a tail slot.
    """
    counts = table.predicate_counts()
    return {
        "_meta": status_block(
            source="rel.json + <split>.json",
            split=index.split,
            note=(
                "predicate_frequencies omits id 0 (__background__) on purpose; "
                "predicate_names retains all 51 entries so index == predicate id."
            ),
        ),
        "predicate_frequencies": {
            str(pid): int(counts[pid]) for pid in range(1, len(index.rel_categories))
        },
        "predicate_names": list(index.rel_categories),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "predicate_stats"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the tracked 10-image data/VisualGenome_sample fixture",
    )
    parser.add_argument("--json-only", action="store_true", help="skip CSV artifacts")
    parser.add_argument("--min-pair-support", type=int, default=20)
    parser.add_argument(
        "--aligned-pairs-only",
        action="store_true",
        help="restrict ambiguity ranking to pairs whose (c_s, c_o) is order-aligned",
    )
    parser.add_argument(
        "--write-predicate-frequencies",
        action="store_true",
        help="also write <data-root>/predicate_frequencies.json",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _maybe_plot(output_dir: Path, predicate_rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"matplotlib unavailable ({exc})"

    ordered = sorted(predicate_rows, key=lambda r: -r["frequency"])[:30]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(ordered)),
        [r["frequency"] for r in ordered],
        color="#4c72b0",
    )
    ax.set_yscale("log")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([r["predicate"] for r in ordered], rotation=90, fontsize=7)
    ax.set_ylabel("train relations (log)")
    ax.set_title("VG150 predicate frequency (top 30)")
    fig.tight_layout()
    fig.savefig(output_dir / "predicate_frequency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [r["frequency"] for r in predicate_rows],
        [r["mean_pair_prior_entropy_bits"] for r in predicate_rows],
        c=[
            {"head": "#c44e52", "body": "#4c72b0", "tail": "#55a868"}.get(r["hbt"], "#999999")
            for r in predicate_rows
        ],
    )
    for r in predicate_rows:
        ax.annotate(r["predicate"], (r["frequency"], r["mean_pair_prior_entropy_bits"]), fontsize=5)
    ax.set_xscale("log")
    ax.set_xlabel("train relations (log)")
    ax.set_ylabel("mean H(P(r | pair)) bits, in-sample")
    ax.set_title("Pair-prior confidence vs predicate frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "pair_prior_entropy.png", dpi=150)
    plt.close(fig)
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(SAMPLE_ROOT if args.dry_run else args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ontology_stats] data_root={data_root} split={args.split}")
    index = load_vg_index(data_root, args.split)
    table = build_relation_table(index)
    print(
        f"[ontology_stats] images={len(index.image_ids)} objects={index.n_objects()} "
        f"relations={len(table)}"
    )

    validation = validate_index_semantics(index, n_images=200, seed=0)
    write_json(output_dir / "index_validation.json", validation)
    if not validation["passed"]:
        print("[ontology_stats] INDEX VALIDATION FAILED -- see index_validation.json")
        return 2
    print(f"[ontology_stats] index validation passed ({validation['relations_checked']} relations)")

    dataset = compute_dataset_stats(table, index)
    dataset["provenance"] = {
        "data_root": str(data_root),
        "sha256": {
            name: sha256_file(data_root / name)
            for name in (f"{args.split}.json", "rel.json")
        },
    }
    predicate_rows = compute_predicate_stats(table, index)
    pair_rows = compute_pair_ambiguity(
        table, index, min_support=args.min_pair_support
    )

    write_json(
        output_dir / "predicate_statistics.json",
        {
            "_meta": status_block(
                split=index.split,
                data_root=str(data_root),
                min_pair_support=args.min_pair_support,
            ),
            "dataset": dataset,
            "predicates": predicate_rows,
        },
    )
    if not args.json_only:
        write_csv(
            output_dir / "predicate_statistics.csv",
            [
                "predicate",
                "predicate_id",
                "hbt",
                "frequency",
                "frequency_share",
                "n_distinct_pairs",
                "n_distinct_pairs_shared_with_other_predicates",
                "mean_pair_prior_entropy_bits",
                "ambiguity_mass",
                "self_prior_conflict_rate",
                "is_study_predicate",
            ],
            predicate_rows,
        )
        write_csv(
            output_dir / "pair_ambiguity.csv",
            [
                "subject",
                "object",
                "c_s",
                "c_o",
                "n_relations",
                "n_predicates",
                "H_R_given_pair_bits",
                "H_R_given_pair_bits_miller_madow",
                "top_predicate",
                "top_predicate_share",
                "all_predicates",
            ],
            pair_rows,
        )

    if not args.no_plots:
        reason = _maybe_plot(output_dir, predicate_rows)
        if reason:
            print(f"[ontology_stats] plots skipped: {reason}")

    if args.write_predicate_frequencies:
        target = data_root / "predicate_frequencies.json"
        write_json(target, predicate_frequencies_payload(table, index))
        print(f"[ontology_stats] wrote {target}")

    print(
        f"[ontology_stats] H(R)={dataset['H_R_bits']:.3f} bits  "
        f"H(R|pair)={dataset['H_R_given_pair_bits']:.3f} "
        f"(MM {dataset['H_R_given_pair_bits_miller_madow']:.3f})  "
        f"I(R;pair)={dataset['I_R_pair_bits']:.3f}"
    )
    print(
        f"[ontology_stats] pairs: {dataset['n_distinct_directed_pairs']} directed, "
        f"{dataset['n_undirected_pairs_multi_relation']} undirected carry >1 predicate"
    )
    print(f"[ontology_stats] wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
