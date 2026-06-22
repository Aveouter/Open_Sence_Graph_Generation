#!/usr/bin/env python3
"""Connect fine-to-coarse collapse with LCompo-SGG seen/unseen composition splits.

Labels each exported relation as seen or unseen using existing composition splits,
then computes collapse metrics stratified by composition group.

Usage:
    python tools/analysis/compute_lcompo_f2c_analysis.py --dry-run
    python tools/analysis/compute_lcompo_f2c_analysis.py --split_id 0
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

COMPOSITION_SPLITS_DIR = _PROJECT_ROOT / "data" / "VisualGenome" / "composition_splits"
SEMANTIC_MAP_PATH = _PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"
PRED_FREQ_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
COMPOSITION_STATS_PATH = (
    _PROJECT_ROOT / "reports" / "composition" / "predicate_composition_stats.csv"
)
DEFAULT_PREDICTIONS_PATH = (
    _PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse" /
    "Motifs" / "PredCLS" / "relation_predictions.jsonl"
)
GT_THRESHOLD = 30


def safe_mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


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


def load_composition_stats():
    if not COMPOSITION_STATS_PATH.exists():
        return {}
    stats = {}
    with open(COMPOSITION_STATS_PATH, "r") as f:
        for row in csv.DictReader(f):
            stats[row["predicate"]] = {
                "frequency": int(row["frequency"]),
                "unique_compositions": int(row["unique_compositions"]),
                "coverage_ratio": float(row["coverage_ratio"]),
                "unique_images": int(row["unique_images"]),
                "hbt_group": row["group"].lower(),
            }
    return stats


def build_dry_run_predictions(main_map, composition_split, predicate_names,
                              rows_per_group=8):
    rows = []
    for pred_idx, (fine_name, mapping) in enumerate(main_map.items()):
        if fine_name not in composition_split:
            continue
        parent = mapping["parent"]
        fine_id = predicate_names.index(fine_name)
        parent_id = predicate_names.index(parent)
        split_data = composition_split[fine_name]

        for group_name in ["seen", "unseen"]:
            pairs = split_data.get(group_name, [])[:rows_per_group]
            for pair_idx, (subj, obj) in enumerate(pairs):
                pred_name = parent if pair_idx % 3 == 0 else fine_name
                pred_id = parent_id if pred_name == parent else fine_id
                rows.append({
                    "image_id": 300000 + pred_idx,
                    "relation_index": len(rows),
                    "subject_class_id": subj,
                    "object_class_id": obj,
                    "gt_predicate_id": fine_id,
                    "gt_predicate_name": fine_name,
                    "pred_predicate_id": pred_id,
                    "pred_predicate_name": pred_name,
                    "matched_pair_found": True,
                })

    rows.append({
        "image_id": 399999,
        "relation_index": len(rows),
        "subject_class_id": 1,
        "object_class_id": 1,
        "gt_predicate_id": predicate_names.index("on"),
        "gt_predicate_name": "on",
        "pred_predicate_id": predicate_names.index("on"),
        "pred_predicate_name": "on",
        "matched_pair_found": False,
    })
    return rows


def label_relations(rows, composition_split):
    """Label each row as seen, unseen, or unmatched in the composition split.

    Returns labeled rows plus three group lists.
    """
    split_lookup = {}
    for pred, comp_data in composition_split.items():
        split_lookup[pred] = {
            "seen": {tuple(pair) for pair in comp_data.get("seen", [])},
            "unseen": {tuple(pair) for pair in comp_data.get("unseen", [])},
        }

    labeled_rows = []
    seen_rows = []
    unseen_rows = []
    unmatched_rows = []

    for r in rows:
        r_labeled = dict(r)
        if not r.get("matched_pair_found"):
            r_labeled["composition_group"] = "unmatched"
            r_labeled["unmatched_reason"] = "export_unmatched_pair"
            unmatched_rows.append(r_labeled)
            labeled_rows.append(r_labeled)
            continue

        pred = r["gt_predicate_name"]
        subj = r.get("subject_class_id")
        obj = r.get("object_class_id")

        if subj is None or obj is None:
            r_labeled["composition_group"] = "unmatched"
            r_labeled["unmatched_reason"] = "missing_subject_object_id"
            unmatched_rows.append(r_labeled)
            labeled_rows.append(r_labeled)
            continue

        if pred not in split_lookup:
            r_labeled["composition_group"] = "unmatched"
            r_labeled["unmatched_reason"] = "predicate_not_in_composition_split"
            unmatched_rows.append(r_labeled)
            labeled_rows.append(r_labeled)
            continue

        pair = (subj, obj)
        pred_lookup = split_lookup[pred]

        if pair in pred_lookup["seen"]:
            r_labeled["composition_group"] = "seen"
            r_labeled["unmatched_reason"] = None
            seen_rows.append(r_labeled)
        elif pair in pred_lookup["unseen"]:
            r_labeled["composition_group"] = "unseen"
            r_labeled["unmatched_reason"] = None
            unseen_rows.append(r_labeled)
        else:
            r_labeled["composition_group"] = "unmatched"
            r_labeled["unmatched_reason"] = "pair_not_in_split_for_predicate"
            unmatched_rows.append(r_labeled)

        labeled_rows.append(r_labeled)

    return labeled_rows, seen_rows, unseen_rows, unmatched_rows


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


def compute_hbt_metrics(labeled_rows, main_map, predicate_names, composition_stats):
    hbt_metrics = {}
    for group in ["head", "body", "tail"]:
        group_rows = [
            r for r in labeled_rows
            if r.get("composition_group") in {"seen", "unseen"}
            and composition_stats.get(r.get("gt_predicate_name"), {}).get("hbt_group") == group
        ]
        hbt_metrics[group] = compute_group_metrics(
            group_rows, main_map, predicate_names, group)
    return hbt_metrics


def summarize_labeling(labeled_rows):
    counts = defaultdict(int)
    reasons = defaultdict(int)
    for row in labeled_rows:
        group = row.get("composition_group", "unmatched")
        counts[group] += 1
        if group == "unmatched":
            reasons[row.get("unmatched_reason") or "unknown"] += 1
    return {
        "seen": counts["seen"],
        "unseen": counts["unseen"],
        "unmatched": counts["unmatched"],
        "unmatched_reasons": dict(sorted(reasons.items())),
    }


def build_gap_rows(seen_metrics, unseen_metrics, composition_stats):
    seen_l = {r["predicate"]: r for r in seen_metrics["predicate_details"]}
    unseen_l = {r["predicate"]: r for r in unseen_metrics["predicate_details"]}
    rows = []
    for pred in sorted(set(list(seen_l.keys()) + list(unseen_l.keys()))):
        sc = seen_l.get(pred, {}).get("collapse_rate", 0)
        uc = unseen_l.get(pred, {}).get("collapse_rate", 0)
        stat = composition_stats.get(pred, {})
        rows.append({
            "predicate": pred,
            "seen_collapse": sc,
            "unseen_collapse": uc,
            "gap": uc - sc,
            "coverage_ratio": stat.get("coverage_ratio"),
            "hbt_group": stat.get("hbt_group"),
            "frequency": stat.get("frequency"),
        })
    return rows


def save_placeholder_plot(output_path, title, message):
    if not _HAS_MPL:
        print(f"      WARNING: matplotlib not available, skipping {output_path.name}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Wrote: {output_path}")


def plot_seen_vs_unseen(seen_metrics, unseen_metrics, output_path):
    if not _HAS_MPL:
        return
    labels = ["seen", "unseen"]
    values = [
        seen_metrics["micro_collapse_rate"],
        unseen_metrics["micro_collapse_rate"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Micro Collapse Rate")
    ax.set_title("Seen vs Unseen Collapse")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Wrote: {output_path}")


def plot_tail_unseen(tail_metrics, tail_unseen_metrics, output_path):
    if not _HAS_MPL:
        return
    labels = ["tail all", "tail unseen"]
    values = [
        tail_metrics["micro_collapse_rate"],
        tail_unseen_metrics["micro_collapse_rate"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["tab:gray", "tab:red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Micro Collapse Rate")
    ax.set_title("Tail-Unseen Collapse")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Wrote: {output_path}")


def plot_coverage_gap(gap_rows, output_path):
    if not _HAS_MPL:
        return
    points = [
        r for r in gap_rows
        if r.get("coverage_ratio") is not None
    ]
    if not points:
        save_placeholder_plot(
            output_path,
            "Collapse Gap vs Coverage Ratio",
            "No predicates have both collapse gap and composition coverage statistics.",
        )
        return
    x = [r["coverage_ratio"] for r in points]
    y = [r["gap"] for r in points]
    labels = [r["predicate"] for r in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color="tab:purple")
    for label, xx, yy in zip(labels, x, y):
        ax.annotate(label, (xx, yy), fontsize=7, alpha=0.8)
    ax.set_xlabel("Composition Coverage Ratio")
    ax.set_ylabel("Unseen - Seen Collapse")
    ax.set_title("Collapse Gap vs Coverage Ratio")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Wrote: {output_path}")


def generate_lcompo_report(seen_metrics, unseen_metrics, hbt_metrics,
                           tail_unseen_metrics, label_summary, metadata,
                           output_path, dry_run=False):
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

    lines.append(f"- Seen relations: {label_summary['seen']}")
    lines.append(f"- Unseen relations: {label_summary['unseen']}")
    lines.append(f"- Unmatched relations: {label_summary['unmatched']}")
    lines.append("")

    if label_summary["unmatched_reasons"]:
        lines.append("### Unmatched Reasons")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for reason, count in label_summary["unmatched_reasons"].items():
            lines.append(f"| {reason} | {count} |")
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

    lines.append("## Head / Body / Tail Collapse")
    lines.append("")
    lines.append("| Group | #Predicates | Total GT | Micro Collapse | Macro Collapse |")
    lines.append("|---|---:|---:|---:|---:|")
    for group in ["head", "body", "tail"]:
        m = hbt_metrics[group]
        lines.append(
            f"| {group} | {m['n_predicates']} | {m['total_gt']} | "
            f"{m['micro_collapse_rate']:.3f} | {m['macro_collapse_rate']:.3f} |"
        )
    lines.append("")

    lines.append("## Tail-Unseen Collapse")
    lines.append("")
    lines.append("| Group | #Predicates | Total GT | Micro Collapse | Macro Collapse |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| tail_unseen | {tail_unseen_metrics['n_predicates']} | "
        f"{tail_unseen_metrics['total_gt']} | "
        f"{tail_unseen_metrics['micro_collapse_rate']:.3f} | "
        f"{tail_unseen_metrics['macro_collapse_rate']:.3f} |"
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
    if dry_run:
        lines.append("> Note: Based on synthetic dry-run data. Real assessment requires model predictions.")
    else:
        lines.append("> Note: Low-sample runs validate plumbing but should not be used as paper evidence.")

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

    use_synthetic = args.dry_run and not args.predictions
    predictions_path = Path(args.predictions) if args.predictions else DEFAULT_PREDICTIONS_PATH
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / "outputs" / "analysis" / "lcompo_f2c" / "Motifs" / "PredCLS"
    )

    print("=" * 70)
    print("LCompo-F2C Seen/Unseen Collapse Analysis")
    print("=" * 70)
    print(f"  Predictions: {'<synthetic dry-run fixture>' if use_synthetic else predictions_path}")
    print(f"  Split ID:    {args.split_id}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load
    print("[1/4] Loading inputs...")
    composition_split, metadata = load_composition_split(args.split_id)
    semantic_map = load_semantic_map()
    predicate_freqs, predicate_names = load_predicate_frequencies()
    composition_stats = load_composition_stats()
    main_map = {
        k: v for k, v in semantic_map.items()
        if v.get("confidence") == "strong" and v.get("use_in_main") is True
    }
    if use_synthetic:
        rows = build_dry_run_predictions(main_map, composition_split, predicate_names)
    elif not predictions_path.exists():
        print(f"      ERROR: Predictions file not found: {predictions_path}")
        print("      Run Task 04 first or use --dry-run")
        sys.exit(1)
    else:
        rows = load_predictions(predictions_path)
    print(f"      {len(rows)} prediction rows")
    print(f"      {len(composition_split)} predicates in composition split")
    print(f"      {len(main_map)} main semantic mappings")
    print(f"      {len(composition_stats)} predicates in composition stats")
    print()

    # Label
    print("[2/4] Labeling relations as seen/unseen...")
    labeled_rows, seen_rows, unseen_rows, unmatched_rows = label_relations(
        rows, composition_split)
    label_summary = summarize_labeling(labeled_rows)
    print(f"      Seen:     {len(seen_rows)}")
    print(f"      Unseen:   {len(unseen_rows)}")
    print(f"      Unmatched: {len(unmatched_rows)}")
    for reason, count in label_summary["unmatched_reasons"].items():
        print(f"        {reason}: {count}")
    print()

    # Compute metrics
    print("[3/4] Computing stratified collapse metrics...")
    seen_metrics = compute_group_metrics(seen_rows, main_map, predicate_names, "seen")
    unseen_metrics = compute_group_metrics(unseen_rows, main_map, predicate_names, "unseen")
    hbt_metrics = compute_hbt_metrics(
        labeled_rows, main_map, predicate_names, composition_stats)
    tail_unseen_rows = [
        r for r in unseen_rows
        if composition_stats.get(r.get("gt_predicate_name"), {}).get("hbt_group") == "tail"
    ]
    tail_unseen_metrics = compute_group_metrics(
        tail_unseen_rows, main_map, predicate_names, "tail_unseen")
    gap_rows = build_gap_rows(seen_metrics, unseen_metrics, composition_stats)
    print(f"      Seen:   {seen_metrics['n_predicates']} predicates, "
          f"collapse={seen_metrics['micro_collapse_rate']:.3f}")
    print(f"      Unseen: {unseen_metrics['n_predicates']} predicates, "
          f"collapse={unseen_metrics['micro_collapse_rate']:.3f}")
    for group, metric in hbt_metrics.items():
        print(f"      {group}: {metric['n_predicates']} predicates, "
              f"gt={metric['total_gt']}, collapse={metric['micro_collapse_rate']:.3f}")
    print()

    # Write
    print("[4/4] Writing outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seen/unseen JSONL
    seen_jsonl = output_dir / "seen_unseen_relation_predictions.jsonl"
    with open(seen_jsonl, "w") as f:
        for r in labeled_rows:
            f.write(json.dumps(r) + "\n")
    print(f"      Wrote: {seen_jsonl}")

    # Collapse metrics CSV
    metrics_csv = output_dir / "seen_unseen_collapse_metrics.csv"
    with open(metrics_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "predicate", "parent", "group", "gt_count", "recall_at_1",
            "collapse_rate", "parent_share", "qualitative_only",
        ])
        for m in [seen_metrics, unseen_metrics] + [
            hbt_metrics[group] for group in ["head", "body", "tail"]
        ]:
            for r in m["predicate_details"]:
                writer.writerow([
                    r["predicate"], r["parent"], r["group"], r["gt_count"],
                    r["recall_at_1"], r["collapse_rate"], r["parent_share"],
                    r["qualitative_only"],
                ])
    print(f"      Wrote: {metrics_csv}")

    # Collapse vs composition gap CSV
    gap_csv = output_dir / "collapse_vs_composition_gap.csv"
    with open(gap_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "predicate", "seen_collapse", "unseen_collapse", "gap",
            "coverage_ratio", "hbt_group", "frequency",
        ])
        for row in gap_rows:
            writer.writerow([
                row["predicate"], row["seen_collapse"], row["unseen_collapse"],
                row["gap"], row.get("coverage_ratio"), row.get("hbt_group"),
                row.get("frequency"),
            ])
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
            "label_summary": label_summary,
            "head_body_tail": {
                group: {
                    "n_predicates": hbt_metrics[group]["n_predicates"],
                    "total_gt": hbt_metrics[group]["total_gt"],
                    "micro_collapse_rate": hbt_metrics[group]["micro_collapse_rate"],
                    "macro_collapse_rate": hbt_metrics[group]["macro_collapse_rate"],
                }
                for group in ["head", "body", "tail"]
            },
            "tail_unseen": {
                "n_predicates": tail_unseen_metrics["n_predicates"],
                "total_gt": tail_unseen_metrics["total_gt"],
                "micro_collapse_rate": tail_unseen_metrics["micro_collapse_rate"],
                "macro_collapse_rate": tail_unseen_metrics["macro_collapse_rate"],
            },
            "composition_split_id": args.split_id,
            "split_ratio": metadata.get("split_ratio"),
        }, f, indent=2)
    print(f"      Wrote: {summary_json}")

    figures_dir = output_dir / "figures"
    plot_seen_vs_unseen(
        seen_metrics, unseen_metrics,
        figures_dir / "collapse_seen_vs_unseen.png",
    )
    plot_tail_unseen(
        hbt_metrics["tail"], tail_unseen_metrics,
        figures_dir / "tail_unseen_collapse.png",
    )
    plot_coverage_gap(
        gap_rows,
        figures_dir / "collapse_vs_coverage_ratio.png",
    )

    # Report
    generate_lcompo_report(
        seen_metrics, unseen_metrics, hbt_metrics, tail_unseen_metrics,
        label_summary, metadata,
        output_dir / "lcompo_f2c_report.md",
        dry_run=args.dry_run,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
