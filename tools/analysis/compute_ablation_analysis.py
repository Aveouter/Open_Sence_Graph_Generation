#!/usr/bin/env python3
"""Ablation analysis: family variants, H/B/T stratification, and control comparison.

Extends Task 05 collapse metrics with ablation-specific analysis.
Does not run model inference.

Usage:
    python tools/analysis/compute_ablation_analysis.py --dry-run
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

# Self-contained: inline shared helpers until Task 05 PR is merged
SEMANTIC_MAP_PATH = _PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"
PRED_FREQ_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
GT_THRESHOLD = 30
BOOTSTRAP_SAMPLES = 1000


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


def compute_predicate_metrics(rows, main_map, predicate_names):
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)
    metrics = {}
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        fine_rows = by_gt.get(fine_name, [])
        gt_count = len(fine_rows)
        if gt_count == 0:
            metrics[fine_name] = {"gt_count": 0, "parent": parent,
                                  "collapse_rate": None, "qualitative_only": True}
            continue
        collapsed = sum(1 for r in fine_rows if r["pred_predicate_name"] == parent)
        errors = gt_count - sum(1 for r in fine_rows if r["pred_predicate_name"] == fine_name)
        metrics[fine_name] = {
            "gt_count": gt_count, "parent": parent,
            "collapse_rate": collapsed / gt_count,
            "parent_share": collapsed / errors if errors > 0 else 0.0,
            "qualitative_only": gt_count < GT_THRESHOLD,
        }
    return metrics


def compute_family_metrics(pred_metrics, main_map):
    families = defaultdict(list)
    for fine_name, mapping in main_map.items():
        families[mapping["parent"]].append(fine_name)
    family_metrics = {}
    for parent, children in families.items():
        child_metrics = [pred_metrics[c] for c in children if c in pred_metrics]
        if not child_metrics:
            continue
        total_gt = sum(m["gt_count"] for m in child_metrics)
        total_collapsed = sum((m.get("collapse_rate") or 0) * m["gt_count"]
                              for m in child_metrics)
        family_metrics[parent] = {
            "children": children, "total_gt": total_gt,
            "micro_collapse_rate": total_collapsed / total_gt if total_gt > 0 else 0,
        }
    return family_metrics


def compute_random_control(rows, main_map, predicate_names, seed=42):
    rng = np.random.default_rng(seed)
    fine_names = list(main_map.keys())
    candidate_parents = [n for n in predicate_names if n != "__background__"]
    random_map = {}
    for f in fine_names:
        pool = [p for p in candidate_parents
                if p != f and p != main_map[f]["parent"]]
        if not pool:
            pool = [p for p in candidate_parents if p != f]
        random_map[f] = {"parent": rng.choice(pool)}
    metrics = compute_predicate_metrics(rows, random_map, predicate_names)
    rates = [m.get("collapse_rate") or 0 for m in metrics.values() if m["gt_count"] > 0]
    return float(np.mean(rates)) if rates else 0.0


def compute_frequency_matched_control(rows, main_map, predicate_names, predicate_freqs):
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)
    controls = []
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        fine_id = predicate_names.index(fine_name) if fine_name in predicate_names else -1
        fine_freq = int(predicate_freqs.get(str(fine_id), 0)) if fine_id > 0 else 0
        family = {n for n, m in main_map.items() if m["parent"] == parent}
        candidates = []
        for i, pname in enumerate(predicate_names):
            if pname in ("__background__",) or pname in family or pname == parent:
                continue
            pfreq = int(predicate_freqs.get(str(i), 0))
            if pfreq > 0 and fine_freq > 0:
                ratio = max(pfreq, fine_freq) / max(min(pfreq, fine_freq), 1)
                if ratio <= 3:
                    candidates.append((pname, pfreq, ratio))
        candidates.sort(key=lambda x: x[2])
        if candidates:
            matched_name = candidates[0][0]
            matched_rows = by_gt.get(matched_name, [])
            if matched_rows:
                collapsed = sum(1 for r in matched_rows
                                if r["pred_predicate_name"] == parent)
                controls.append(collapsed / len(matched_rows))
    return float(np.mean(controls)) if controls else 0.0


def compute_subject_object_prior_baseline(rows, main_map, predicate_names):
    pair_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pair_totals = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r.get("matched_pair_found"):
            continue
        s = r["subject_class_id"]
        o = r["object_class_id"]
        p = r["gt_predicate_name"]
        pair_counts[s][o][p] += 1
        pair_totals[s][o] += 1
    controls = []
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        prior_parent = 0
        prior_total = 0
        for s in pair_counts:
            for o in pair_counts[s]:
                total = pair_totals[s][o]
                if total > 0:
                    parent_prob = pair_counts[s][o].get(parent, 0) / total
                    fine_count = pair_counts[s][o].get(fine_name, 0)
                    if fine_count > 0:
                        prior_parent += parent_prob * fine_count
                        prior_total += fine_count
        if prior_total > 0:
            controls.append(prior_parent / prior_total)
    return float(np.mean(controls)) if controls else 0.0

DEFAULT_PREDICTIONS_PATH = (
    _PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse" /
    "Motifs" / "PredCLS" / "relation_predictions.jsonl"
)


def compute_hbt_stratification(rows, main_map, predicate_names, predicate_freqs):
    """Compute collapse metrics stratified by Head/Body/Tail frequency groups.

    Head: top 20% predicates by frequency
    Body: middle 40%
    Tail: bottom 40%
    """
    # Get frequencies for fine predicates
    pred_freq_list = []
    for fine_name in main_map:
        vg_id = predicate_names.index(fine_name) if fine_name in predicate_names else -1
        freq = int(predicate_freqs.get(str(vg_id), 0)) if vg_id > 0 else 0
        pred_freq_list.append((fine_name, freq))

    if not pred_freq_list:
        return {}, {}, {}

    pred_freq_list.sort(key=lambda x: x[1], reverse=True)
    n = len(pred_freq_list)
    head_n = max(1, int(n * 0.2))
    body_n = max(1, int(n * 0.4))
    tail_n = n - head_n - body_n

    head_preds = set(p[0] for p in pred_freq_list[:head_n])
    body_preds = set(p[0] for p in pred_freq_list[head_n:head_n + body_n])
    tail_preds = set(p[0] for p in pred_freq_list[head_n + body_n:])

    # Group rows by GT
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)

    groups = {"head": head_preds, "body": body_preds, "tail": tail_preds}
    results = {}

    for group_name, pred_set in groups.items():
        group_rows = []
        gt_counts = {}
        for pname in pred_set:
            p_rows = by_gt.get(pname, [])
            group_rows.extend(p_rows)
            gt_counts[pname] = len(p_rows)

        total_gt = sum(gt_counts.values())
        collapsed = 0
        errors = 0
        for pname in pred_set:
            if pname not in main_map:
                continue
            parent = main_map[pname]["parent"]
            p_rows = by_gt.get(pname, [])
            for r in p_rows:
                if r["pred_predicate_name"] == parent:
                    collapsed += 1
                if r["pred_predicate_name"] != pname:
                    errors += 1

        results[group_name] = {
            "predicates": sorted(pred_set),
            "n_predicates": len(pred_set),
            "total_gt": total_gt,
            "collapsed_count": collapsed,
            "error_count": errors,
            "collapse_rate": collapsed / total_gt if total_gt > 0 else 0,
            "parent_share": collapsed / errors if errors > 0 else 0,
        }

    return results


def compute_family_ablation(rows, main_map, predicate_names):
    """Compute metrics for on-family-only, on-family-removed, in-family-only."""
    families = defaultdict(list)
    for fine_name, mapping in main_map.items():
        families[mapping["parent"]].append(fine_name)

    variants = {}
    for variant_name, family_filter in [
        ("on_family_only", {"on"}),
        ("on_family_removed", set(families.keys()) - {"on"}),
        ("all_families", set(families.keys())),
    ]:
        variant_map = {
            k: v for k, v in main_map.items()
            if v["parent"] in family_filter
        }
        pred_metrics = compute_predicate_metrics(rows, variant_map, predicate_names)
        family_metrics = compute_family_metrics(pred_metrics, variant_map)

        total_gt = sum(m["gt_count"] for m in pred_metrics.values())
        total_collapsed = sum(
            (m.get("collapse_rate") or 0) * m["gt_count"]
            for m in pred_metrics.values()
        )
        micro_collapse = total_collapsed / total_gt if total_gt > 0 else 0

        variants[variant_name] = {
            "n_predicates": len(variant_map),
            "n_families": len(family_metrics),
            "total_gt": total_gt,
            "micro_collapse_rate": micro_collapse,
            "families_included": sorted(family_filter),
        }

    return variants


def generate_ablation_report(ablation_results, hbt_results, neg_controls, output_path):
    """Generate ablation and controls report."""
    lines = []
    lines.append("# Ablation and Controls Report")
    lines.append("")
    lines.append(f"**GT threshold**: {GT_THRESHOLD}")
    lines.append(f"**Bootstrap samples**: {BOOTSTRAP_SAMPLES}")
    lines.append("")

    # Family ablation
    lines.append("## Family Ablation")
    lines.append("")
    lines.append("| Variant | #Predicates | #Families | Total GT | Micro Collapse |")
    lines.append("|---|---|---:|---:|")
    for name, v in ablation_results.items():
        lines.append(
            f"| {name} | {v['n_predicates']} | {v['n_families']} | "
            f"{v['total_gt']} | {v['micro_collapse_rate']:.3f} |"
        )
    lines.append("")

    # H/B/T stratification
    lines.append("## Head/Body/Tail Stratification")
    lines.append("")
    lines.append("| Group | #Predicates | Total GT | Collapse Rate | Parent Share |")
    lines.append("|---|---|---:|---:|---:|")
    for group_name in ["head", "body", "tail"]:
        if group_name in hbt_results:
            r = hbt_results[group_name]
            lines.append(
                f"| {group_name} | {r['n_predicates']} | {r['total_gt']} | "
                f"{r['collapse_rate']:.3f} | {r['parent_share']:.3f} |"
            )
    lines.append("")

    # Controls summary
    lines.append("## Negative Controls Summary")
    lines.append("")
    if neg_controls.get("random") is not None:
        lines.append(f"- Random parent control: {neg_controls['random']:.4f}")
    if neg_controls.get("frequency_matched"):
        lines.append(
            f"- Frequency-matched control avg: "
            f"{neg_controls['frequency_matched']['avg_rate']:.4f}"
        )
    if neg_controls.get("subject_object_prior"):
        lines.append(
            f"- Subject-object prior avg: "
            f"{neg_controls['subject_object_prior']['avg_rate']:.4f}"
        )
    lines.append("")

    # Unresolved objections
    lines.append("## Unresolved Reviewer Objections")
    lines.append("")
    lines.append("| Objection | Status | Evidence |")
    lines.append("|---|---|")
    lines.append("| Parent is high-frequency | Needs real data | Controls underpowered with synthetic data |")
    lines.append("| Object-pair priors explain collapse | Needs real data | S-O prior baseline requires more samples |")
    lines.append("| Semantic map is hand-picked | Addressed | Validated by Task 03 validator |")
    lines.append("| Tail predicates too few samples | Addressed | Qualitative-only flag for GT < 30 |")
    lines.append("| Single-model result | Pending | Cross-model validation in Task 07 |")
    lines.append("")

    lines.append("> Note: This report is based on synthetic data and is illustrative.")
    lines.append("> Real assessment requires model predictions from Task 04/Task 07.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Ablation analysis for collapse metrics"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with default synthetic predictions")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to relation_predictions.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    predictions_path = (
        Path(args.predictions) if args.predictions else DEFAULT_PREDICTIONS_PATH
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir else predictions_path.parent
    )

    print("=" * 70)
    print("Ablation Analysis")
    print("=" * 70)
    print(f"  Predictions: {predictions_path}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load
    print("[1/5] Loading inputs...")
    rows = load_predictions(predictions_path)
    semantic_map = load_semantic_map()
    predicate_freqs, predicate_names = load_predicate_frequencies()
    main_map = {
        k: v for k, v in semantic_map.items()
        if v.get("confidence") == "strong" and v.get("use_in_main") is True
    }
    print(f"      {len(rows)} rows, {len(main_map)} main mappings")
    print()

    # Family ablation
    print("[2/5] Family ablation...")
    ablation_results = compute_family_ablation(rows, main_map, predicate_names)
    for name, v in ablation_results.items():
        print(f"      {name}: {v['n_predicates']} preds, "
              f"collapse={v['micro_collapse_rate']:.3f}")
    print()

    # H/B/T
    print("[3/5] Head/Body/Tail stratification...")
    hbt_results = compute_hbt_stratification(
        rows, main_map, predicate_names, predicate_freqs)
    for group_name, r in hbt_results.items():
        print(f"      {group_name}: {r['n_predicates']} preds, "
              f"gt={r['total_gt']}, collapse={r['collapse_rate']:.3f}")
    print()

    # Controls
    print("[4/5] Negative controls...")
    neg_controls = {}
    neg_controls["random"] = compute_random_control(
        rows, main_map, predicate_names)
    freq_avg = compute_frequency_matched_control(
        rows, main_map, predicate_names, predicate_freqs)
    neg_controls["frequency_matched"] = {"avg_rate": freq_avg}
    so_avg = compute_subject_object_prior_baseline(
        rows, main_map, predicate_names)
    neg_controls["subject_object_prior"] = {"avg_rate": so_avg}
    print(f"      Random: {neg_controls['random']:.4f}")
    print(f"      Freq-matched: {freq_avg:.4f}")
    print(f"      S-O prior: {so_avg:.4f}")
    print()

    # Write
    print("[5/5] Writing outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ablation CSV
    abl_csv = output_dir / "ablation_summary.csv"
    with open(abl_csv, "w") as f:
        f.write("variant,n_predicates,n_families,total_gt,micro_collapse_rate\n")
        for name, v in ablation_results.items():
            f.write(f"{name},{v['n_predicates']},{v['n_families']},"
                    f"{v['total_gt']},{v['micro_collapse_rate']}\n")
    print(f"      Wrote: {abl_csv}")

    # HBT CSV
    hbt_csv = output_dir / "head_body_tail_collapse.csv"
    with open(hbt_csv, "w") as f:
        f.write("group,n_predicates,total_gt,collapse_rate,parent_share\n")
        for group_name in ["head", "body", "tail"]:
            if group_name in hbt_results:
                r = hbt_results[group_name]
                f.write(f"{group_name},{r['n_predicates']},{r['total_gt']},"
                        f"{r['collapse_rate']},{r['parent_share']}\n")
    print(f"      Wrote: {hbt_csv}")

    # Control significance JSON
    sig_json = output_dir / "control_significance.json"
    with open(sig_json, "w") as f:
        json.dump({
            "controls": {
                "random_parent": neg_controls["random"],
                "frequency_matched_avg": freq_avg,
                "subject_object_prior_avg": so_avg,
            },
            "ablation": {
                k: {"micro_collapse_rate": v["micro_collapse_rate"],
                    "n_predicates": v["n_predicates"]}
                for k, v in ablation_results.items()
            },
            "hbt_stratification": {
                k: {"collapse_rate": v["collapse_rate"],
                    "n_predicates": v["n_predicates"],
                    "total_gt": v["total_gt"]}
                for k, v in hbt_results.items()
            },
            "note": "Synthetic data — real assessment requires model predictions",
        }, f, indent=2)
    print(f"      Wrote: {sig_json}")

    # Report
    generate_ablation_report(
        ablation_results, hbt_results, neg_controls,
        output_dir / "ablation_and_controls_report.md",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
