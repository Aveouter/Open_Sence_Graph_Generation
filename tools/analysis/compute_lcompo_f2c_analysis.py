#!/usr/bin/env python3
"""Connect fine-to-coarse collapse with LCompo-SGG seen/unseen composition splits.

Labels each exported relation as seen or unseen using existing composition splits,
then computes collapse metrics stratified by composition group.

Usage:
    python tools/analysis/compute_lcompo_f2c_analysis.py --dry-run
    python tools/analysis/compute_lcompo_f2c_analysis.py --split_id 0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

COMPOSITION_SPLITS_DIR = _PROJECT_ROOT / "data" / "VisualGenome" / "composition_splits"
SEMANTIC_MAP_PATH = _PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"
PRED_FREQ_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
DEFAULT_PREDICTIONS_PATH = (
    _PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse" /
    "Motifs" / "PredCLS" / "relation_predictions.jsonl"
)
GT_THRESHOLD = 30


def load_composition_split(split_id=0):
    """Load composition split JSON. Returns {predicate: {seen: [(s,o),...], unseen: [(s,o),...]}}."""
    path = COMPOSITION_SPLITS_DIR / f"split_{split_id}" / "composition_split.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["composition_split"], data["metadata"]


def load_predictions(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_semantic_map():
    with open(SEMANTIC_MAP_PATH, "r") as f:
        data = json.load(f)
    data.pop("_meta", None)
    return data


def load_predicate_frequencies():
    with open(PRED_FREQ_PATH, "r") as f:
        data = json.load(f)
    return data["predicate_frequencies"], data["predicate_names"]


def label_relations(rows, composition_split):
    """Label each row as seen, unseen, or unmatched in the composition split.

    Returns three lists of rows: seen, unseen, unmatched.
    """
    seen_rows = []
    unseen_rows = []
    unmatched_rows = []

    for r in rows:
        if not r.get("matched_pair_found"):
            unmatched_rows.append(r)
            continue

        pred = r["gt_predicate_name"]
        subj = r["subject_class_id"]
        obj = r["object_class_id"]

        if pred not in composition_split:
            unmatched_rows.append(r)
            continue

        comp_data = composition_split[pred]
        pair = [subj, obj]

        # Check if this (subject, object) pair is in seen or unseen
        if pair in comp_data.get("seen", []):
            seen_rows.append(r)
        elif pair in comp_data.get("unseen", []):
            unseen_rows.append(r)
        else:
            unmatched_rows.append(r)

    return seen_rows, unseen_rows, unmatched_rows


def compute_group_metrics(rows, main_map, predicate_names, group_name):
    """Compute collapse metrics for a group of rows."""
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)

    results = []
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        fine_rows = by_gt.get(fine_name, [])
        gt_count = len(fine_rows)
        if gt_count == 0:
            continue

        collapsed = sum(1 for r in fine_rows if r["pred_predicate_name"] == parent)
        correct = sum(1 for r in fine_rows if r["pred_predicate_name"] == fine_name)
        errors = gt_count - correct

        results.append({
            "predicate": fine_name,
            "parent": parent,
            "group": group_name,
            "gt_count": gt_count,
            "recall_at_1": correct / gt_count,
            "collapse_rate": collapsed / gt_count,
            "parent_share": collapsed / errors if errors > 0 else 0.0,
            "qualitative_only": gt_count < GT_THRESHOLD,
        })

    # Micro average
    total_gt = sum(r["gt_count"] for r in results)
    total_collapsed = sum(r["collapse_rate"] * r["gt_count"] for r in results)
    total_errors = sum(
        (1 - r["recall_at_1"]) * r["gt_count"] for r in results)
    macro_collapse = np.mean([r["collapse_rate"] for r in results]) if results else 0

    return {
        "group": group_name,
        "n_predicates": len(results),
        "total_gt": total_gt,
        "micro_collapse_rate": total_collapsed / total_gt if total_gt > 0 else 0,
        "micro_parent_share": total_collapsed / total_errors if total_errors > 0 else 0,
        "macro_collapse_rate": macro_collapse,
        "predicate_details": results,
    }


def generate_lcompo_report(seen_metrics, unseen_metrics, unmatched_count, metadata,
                           output_path):
    """Generate LCompo-F2C report."""
    lines = []
    lines.append("# LCompo-F2C: Composition-Aware Collapse Report")
    lines.append("")
    lines.append(f"**Split ratio**: {metadata.get('split_ratio', 'N/A')}")
    lines.append(f"**GT threshold**: {GT_THRESHOLD}")
    lines.append("")

    lines.append("## Seen vs Unseen Collapse")
    lines.append("")
    lines.append("| Group | #Predicates | Total GT | Micro Collapse | Macro Collapse |")
    lines.append("|---|---|---:|---:|---:|")

    for m in [seen_metrics, unseen_metrics]:
        lines.append(
            f"| {m['group']} | {m['n_predicates']} | {m['total_gt']} | "
            f"{m['micro_collapse_rate']:.3f} | {m['macro_collapse_rate']:.3f} |"
        )
    lines.append("")

    lines.append(f"- Unmatched relations (not in split): {unmatched_count}")
    lines.append("")

    # Per-predicate seen vs unseen
    lines.append("## Per-Predicate Seen vs Unseen")
    lines.append("")
    lines.append("| Predicate | Seen GT | Seen Collapse | Unseen GT | Unseen Collapse | Qualitative |")
    lines.append("|---|---|---:|---:|---:|---:|")

    seen_lookup = {r["predicate"]: r for r in seen_metrics["predicate_details"]}
    unseen_lookup = {r["predicate"]: r for r in unseen_metrics["predicate_details"]}

    all_preds = sorted(set(list(seen_lookup.keys()) + list(unseen_lookup.keys())))
    for pred in all_preds:
        s = seen_lookup.get(pred, {})
        u = unseen_lookup.get(pred, {})
        s_gt = s.get("gt_count", 0)
        u_gt = u.get("gt_count", 0)
        qual = "yes" if (s_gt + u_gt) < GT_THRESHOLD else ""
        lines.append(
            f"| {pred} | {s_gt} | {s.get('collapse_rate', 0):.3f} | "
            f"{u_gt} | {u.get('collapse_rate', 0):.3f} | {qual} |"
        )

    lines.append("")

    # Assessment
    lines.append("## Assessment")
    lines.append("")

    seen_collapse = seen_metrics["micro_collapse_rate"]
    unseen_collapse = unseen_metrics["micro_collapse_rate"]

    if unseen_collapse > seen_collapse * 1.2 and unseen_metrics["total_gt"] >= 5:
        lines.append(
            "**LCompo evidence: SUPPORTS** — Unseen collapse rate "
            f"({unseen_collapse:.3f}) exceeds seen collapse rate "
            f"({seen_collapse:.3f}), suggesting composition pressure "
            "amplifies fine-to-coarse collapse."
        )
    elif unseen_metrics["total_gt"] < 5:
        lines.append(
            "**LCompo evidence: INCONCLUSIVE** — Insufficient unseen samples "
            f"(n={unseen_metrics['total_gt']}) for reliable comparison."
        )
    else:
        lines.append(
            "**LCompo evidence: NEUTRAL** — Collapse rates are similar for "
            f"seen ({seen_collapse:.3f}) and unseen ({unseen_collapse:.3f}) "
            "compositions."
        )

    lines.append("")
    lines.append("> Note: Based on synthetic/dry-run data. Real assessment requires model predictions.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="LCompo-F2C seen/unseen collapse analysis"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use default synthetic predictions")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to relation_predictions.jsonl")
    parser.add_argument("--split_id", type=int, default=0,
                        help="Composition split ID (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    predictions_path = (
        Path(args.predictions) if args.predictions else DEFAULT_PREDICTIONS_PATH
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / "outputs" / "analysis" / "lcompo_f2c" / "Motifs" / "PredCLS"
    )

    print("=" * 70)
    print("LCompo-F2C Seen/Unseen Collapse Analysis")
    print("=" * 70)
    print(f"  Predictions: {predictions_path}")
    print(f"  Split ID:    {args.split_id}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load
    print("[1/4] Loading inputs...")
    rows = load_predictions(predictions_path)
    composition_split, metadata = load_composition_split(args.split_id)
    semantic_map = load_semantic_map()
    predicate_freqs, predicate_names = load_predicate_frequencies()
    main_map = {
        k: v for k, v in semantic_map.items()
        if v.get("confidence") == "strong" and v.get("use_in_main") is True
    }
    print(f"      {len(rows)} prediction rows")
    print(f"      {len(composition_split)} predicates in composition split")
    print(f"      {len(main_map)} main semantic mappings")
    print()

    # Label
    print("[2/4] Labeling relations as seen/unseen...")
    seen_rows, unseen_rows, unmatched_rows = label_relations(rows, composition_split)
    print(f"      Seen:     {len(seen_rows)}")
    print(f"      Unseen:   {len(unseen_rows)}")
    print(f"      Unmatched: {len(unmatched_rows)}")
    print()

    # Compute metrics
    print("[3/4] Computing stratified collapse metrics...")
    seen_metrics = compute_group_metrics(seen_rows, main_map, predicate_names, "seen")
    unseen_metrics = compute_group_metrics(unseen_rows, main_map, predicate_names, "unseen")
    print(f"      Seen:   {seen_metrics['n_predicates']} predicates, "
          f"collapse={seen_metrics['micro_collapse_rate']:.3f}")
    print(f"      Unseen: {unseen_metrics['n_predicates']} predicates, "
          f"collapse={unseen_metrics['micro_collapse_rate']:.3f}")
    print()

    # Write
    print("[4/4] Writing outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seen/unseen JSONL
    seen_jsonl = output_dir / "seen_unseen_relation_predictions.jsonl"
    with open(seen_jsonl, "w") as f:
        for r in seen_rows + unseen_rows:
            r_labeled = dict(r)
            r_labeled["composition_group"] = "seen" if r in seen_rows else "unseen"
            f.write(json.dumps(r_labeled) + "\n")
    print(f"      Wrote: {seen_jsonl}")

    # Collapse metrics CSV
    metrics_csv = output_dir / "seen_unseen_collapse_metrics.csv"
    with open(metrics_csv, "w") as f:
        f.write("predicate,parent,group,gt_count,recall_at_1,collapse_rate,"
                "parent_share,qualitative_only\n")
        for m in [seen_metrics, unseen_metrics]:
            for r in m["predicate_details"]:
                f.write(
                    f"{r['predicate']},{r['parent']},{r['group']},{r['gt_count']},"
                    f"{r['recall_at_1']},{r['collapse_rate']},{r['parent_share']},"
                    f"{r['qualitative_only']}\n"
                )
    print(f"      Wrote: {metrics_csv}")

    # Collapse vs composition gap CSV
    gap_csv = output_dir / "collapse_vs_composition_gap.csv"
    with open(gap_csv, "w") as f:
        f.write("predicate,seen_collapse,unseen_collapse,gap\n")
        seen_l = {r["predicate"]: r for r in seen_metrics["predicate_details"]}
        unseen_l = {r["predicate"]: r for r in unseen_metrics["predicate_details"]}
        for pred in sorted(set(list(seen_l.keys()) + list(unseen_l.keys()))):
            sc = seen_l.get(pred, {}).get("collapse_rate", 0)
            uc = unseen_l.get(pred, {}).get("collapse_rate", 0)
            gap = uc - sc
            f.write(f"{pred},{sc},{uc},{gap}\n")
    print(f"      Wrote: {gap_csv}")

    # Summary JSON
    summary_json = output_dir / "tail_unseen_summary.json"
    with open(summary_json, "w") as f:
        json.dump({
            "seen": {
                "n_predicates": seen_metrics["n_predicates"],
                "total_gt": seen_metrics["total_gt"],
                "micro_collapse_rate": seen_metrics["micro_collapse_rate"],
                "macro_collapse_rate": seen_metrics["macro_collapse_rate"],
            },
            "unseen": {
                "n_predicates": unseen_metrics["n_predicates"],
                "total_gt": unseen_metrics["total_gt"],
                "micro_collapse_rate": unseen_metrics["micro_collapse_rate"],
                "macro_collapse_rate": unseen_metrics["macro_collapse_rate"],
            },
            "unmatched_count": len(unmatched_rows),
            "composition_split_id": args.split_id,
            "split_ratio": metadata.get("split_ratio"),
        }, f, indent=2)
    print(f"      Wrote: {summary_json}")

    # Report
    generate_lcompo_report(
        seen_metrics, unseen_metrics, len(unmatched_rows), metadata,
        output_dir / "lcompo_f2c_report.md",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
