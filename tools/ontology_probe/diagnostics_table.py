"""M10: per-predicate decomposition.

Overall averages hide the result this experiment is most likely to produce: a
small set of predicates where visual evidence genuinely generalises across
fillers, a set that is purely prior-driven, and a set where the annotations
themselves are inconsistent.  Reporting one number would average those into
nothing.

Joins, per predicate:

* corpus statistics -- frequency, ``H(R | pair)``, ambiguity mass;
* ``B2`` and ``B4`` recall on each split, and ``ΔV = B4 - B2``;
* the shuffle drop from M7;
* the visual rescue rate from M8;
* the canonical-space gain from the pooled M9 comparison.

A verdict per predicate is assigned from thresholds recorded in the artifact,
so a reader can re-derive it rather than trusting the label.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (
    DEFAULT_OUTPUT_ROOT,
    hbt_group,
    read_json,
    status_block,
    write_csv,
    write_json,
)

#: Thresholds behind the verdict column. Recorded in the artifact so the labels
#: can be re-derived rather than taken on faith.
VERDICT_RULES: dict[str, Any] = {
    "min_support": 50,
    "delta_visual_min": 0.05,
    "shuffle_drop_min": 0.05,
    "supports_verdict": (
        "vision_generalizes requires ΔV >= delta_visual_min AND a shuffle drop "
        ">= shuffle_drop_min; ΔV without a shuffle drop is not evidence of "
        "using visual correspondence"
    ),
}


def _read_csv_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _cell_per_class(probes_dir: Path, cell: str) -> dict[str, dict[str, Any]]:
    path = probes_dir / cell / "metrics.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get("metrics", {}).get("per_class") or []
    return {row["class_name"]: row for row in rows}


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def build_table(
    output_root: Path,
    split: str,
    space: str,
    seed: int,
) -> list[dict[str, Any]]:
    stats_path = output_root / "predicate_stats" / "predicate_statistics.json"
    stats_payload = read_json(stats_path) if stats_path.exists() else {}
    stats = {row["predicate"]: row for row in stats_payload.get("predicates", [])}

    probes_dir = output_root / "probes"
    b2 = _cell_per_class(probes_dir, f"{space}__{split}__B2__s{seed}")
    b4 = _cell_per_class(probes_dir, f"{space}__{split}__B4__s{seed}")
    b3 = _cell_per_class(probes_dir, f"{space}__{split}__B3__s{seed}")

    shuffle_path = output_root / "shuffle" / "shuffle_by_predicate.csv"
    shuffle: dict[str, dict[str, list[float]]] = {}
    if shuffle_path.exists():
        with shuffle_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant") != "all":
                    continue
                shuffle.setdefault(row["predicate"], {})[row["cell"]] = _as_float(
                    row["shuffle_drop"]
                )

    rescue_path = output_root / "rescue" / "rescue_by_predicate.csv"
    rescue_rows = _read_csv_by(rescue_path, "predicate")

    predicates = sorted(set(stats) | set(b2) | set(b4))
    rows: list[dict[str, Any]] = []
    for name in predicates:
        stat = stats.get(name, {})
        row_b2 = b2.get(name, {})
        row_b4 = b4.get(name, {})
        row_b3 = b3.get(name, {})
        support = int(row_b2.get("support") or row_b4.get("support") or 0)

        b2_recall = _as_float(row_b2.get("accuracy"))
        b4_recall = _as_float(row_b4.get("accuracy"))
        delta_v = (
            b4_recall - b2_recall
            if b2_recall is not None and b4_recall is not None
            else None
        )

        drops = [
            value
            for cell_drops in shuffle.get(name, {}).values()
            if (value := cell_drops) is not None
        ]
        shuffle_drop = sum(drops) / len(drops) if drops else None

        entry: dict[str, Any] = {
            "predicate": name,
            "hbt": stat.get("hbt") or hbt_group(0),
            "frequency": stat.get("frequency"),
            "n_distinct_pairs": stat.get("n_distinct_pairs"),
            "H_R_given_pair_bits": stat.get("mean_pair_prior_entropy_bits"),
            "ambiguity_mass": stat.get("ambiguity_mass"),
            "support": support,
            "b2_recall": b2_recall,
            "b3_recall": _as_float(row_b3.get("accuracy")),
            "b4_recall": b4_recall,
            "delta_visual": delta_v,
            "shuffle_drop": shuffle_drop,
        }
        rescue_row = rescue_rows.get(name)
        entry["vrr"] = _as_float(rescue_row.get("vrr")) if rescue_row else None
        entry["n_base_wrong"] = (
            int(float(rescue_row["n_base_wrong"])) if rescue_row else None
        )
        entry["verdict"] = classify(entry)
        rows.append(entry)

    # Canonical gain is a per-label-space quantity, not a per-predicate one, so
    # it stays in the pooled comparison artifact. Smearing it across predicate
    # rows would make it read as a predicate-level measurement.
    return rows


def classify(row: dict[str, Any]) -> str:
    """Verdict from the recorded thresholds; both conditions are required.

    An earlier version tested only the shuffle drop, so predicates with a
    negligible ``ΔV`` were labelled ``vision_generalizes`` on the strength of a
    large shuffle drop alone -- a gain that is not there cannot be evidence of
    visual generalisation.
    """
    support = row.get("support") or 0
    if support < VERDICT_RULES["min_support"]:
        return "degenerate_support"
    delta_v = row.get("delta_visual")
    drop = row.get("shuffle_drop")
    if delta_v is None:
        return "inconclusive"
    if delta_v < VERDICT_RULES["delta_visual_min"]:
        # Covers both a negative delta and a positive one too small to claim.
        return "prior_or_geometry_dominated"
    if drop is None:
        return "delta_unverified_against_shuffle"
    if drop >= VERDICT_RULES["shuffle_drop_min"]:
        return "vision_generalizes"
    return "delta_without_correspondence"


def render_markdown(rows: list[dict[str, Any]], split: str, space: str) -> str:
    lines = [
        f"# Per-predicate diagnostics ({space}, {split})",
        "",
        "| Predicate | B2 | B4 | ΔV | Shuffle drop | VRR | H(R\\|pair) | Support | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any, places: int = 3) -> str:
        number = _as_float(value)
        return "—" if number is None else f"{number:.{places}f}"

    for row in rows:
        if row["predicate"].startswith("__"):
            continue
        lines.append(
            "| {p} | {b2} | {b4} | {dv} | {sd} | {vrr} | {h} | {s} | {v} |".format(
                p=row["predicate"],
                b2=fmt(row.get("b2_recall")),
                b4=fmt(row.get("b4_recall")),
                dv=fmt(row.get("delta_visual"), 4),
                sd=fmt(row.get("shuffle_drop"), 4),
                vrr=fmt(row.get("vrr")),
                h=fmt(row.get("H_R_given_pair_bits"), 2),
                s=row.get("support") or "—",
                v=row.get("verdict"),
            )
        )
    counts: dict[str, int] = {}
    for row in rows:
        if not row["predicate"].startswith("__"):
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    lines += ["", "## Verdict counts", ""]
    for verdict, count in sorted(counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines += [
        "",
        "## Rules",
        "",
        f"`{VERDICT_RULES['supports_verdict']}`",
        f"min_support = {VERDICT_RULES['min_support']}, "
        f"delta_visual_min = {VERDICT_RULES['delta_visual_min']}, "
        f"shuffle_drop_min = {VERDICT_RULES['shuffle_drop_min']}",
        "",
        "`__canonical_gain__` is a label-space-level quantity and lives in the",
        "pooled comparison artifact, not in the per-predicate rows.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "diagnostics"))
    parser.add_argument("--split", default="pair_ood")
    parser.add_argument("--space", default="vg50")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_table(Path(args.output_root), args.split, args.space, args.seed)
    write_csv(
        output_dir / f"predicate_diagnostics_{args.space}_{args.split}.csv",
        [
            "predicate", "hbt", "frequency", "n_distinct_pairs",
            "H_R_given_pair_bits", "ambiguity_mass", "support",
            "b2_recall", "b3_recall", "b4_recall", "delta_visual",
            "shuffle_drop", "vrr", "n_base_wrong", "verdict",
        ],
        rows,
    )
    markdown = render_markdown(rows, args.split, args.space)
    (output_dir / f"predicate_diagnostics_{args.space}_{args.split}.md").write_text(
        markdown, encoding="utf-8"
    )
    write_json(
        output_dir / f"predicate_diagnostics_{args.space}_{args.split}.json",
        status_block(
            split=args.split, space=args.space, seed=args.seed,
            rules=VERDICT_RULES, rows=rows,
        ),
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
