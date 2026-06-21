#!/usr/bin/env python3
"""Compute collapse metrics, controls, CIs, and visualizations from relation predictions.

Consumes Task 03 (semantic map) and Task 04 (relation_predictions.jsonl) outputs.
Does not run model inference.

Usage:
    python tools/analysis/compute_collapse_metrics.py --dry-run
    python tools/analysis/compute_collapse_metrics.py \
        --predictions outputs/analysis/fine_to_coarse/Motifs/PredCLS/relation_predictions.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Pre-import matplotlib to avoid slow first-import inside plot functions
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

SEMANTIC_MAP_PATH = _PROJECT_ROOT / "configs" / "predicate_semantic_map_vg150.json"
PRED_FREQ_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
DEFAULT_PREDICTIONS_PATH = (
    _PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse" /
    "Motifs" / "PredCLS" / "relation_predictions.jsonl"
)

GT_THRESHOLD = 30
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 95
TOP_K = 5


# ---- Data loading ----

def load_semantic_map():
    with open(SEMANTIC_MAP_PATH, "r") as f:
        data = json.load(f)
    meta = data.pop("_meta", {})
    return data, meta


def load_predictions(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predicate_frequencies():
    with open(PRED_FREQ_PATH, "r") as f:
        data = json.load(f)
    return data["predicate_frequencies"], data["predicate_names"]


# ---- Core metrics ----

def compute_predicate_metrics(rows, main_map, predicate_names):
    """Compute per-predicate collapse metrics.

    Returns dict: predicate_name -> {gt_count, recall, collapse_rate, parent_share, ...}
    """
    # Group rows by GT predicate
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
            metrics[fine_name] = {
                "gt_count": 0,
                "parent": parent,
                "recall_at_1": None,
                "collapse_rate": None,
                "parent_share": None,
                "parent_above_fine_rate": None,
                "fine_in_topk_rate": None,
                "qualitative_only": True,
            }
            continue

        # Recall@1: GT=f, Pred=f
        correct = sum(1 for r in fine_rows if r["pred_predicate_name"] == fine_name)

        # CollapseRate: GT=f, Pred=parent
        collapsed = sum(1 for r in fine_rows if r["pred_predicate_name"] == parent)

        # ParentShare: among errors, how many go to parent
        errors = sum(1 for r in fine_rows if r["pred_predicate_name"] != fine_name)
        parent_share = collapsed / errors if errors > 0 else 0.0

        # ParentAboveFine: score(parent) > score(fine)
        fine_vg_id = predicate_names.index(fine_name) if fine_name in predicate_names else -1
        parent_vg_id = predicate_names.index(parent) if parent in predicate_names else -1
        parent_above = 0
        if fine_vg_id > 0 and parent_vg_id > 0:
            fine_score_idx = fine_vg_id - 1
            parent_score_idx = parent_vg_id - 1
            for r in fine_rows:
                scores = r.get("predicate_scores_all", [])
                if (len(scores) > max(fine_score_idx, parent_score_idx) and
                        scores[parent_score_idx] > scores[fine_score_idx]):
                    parent_above += 1

        # FineInTopK: GT=f and f in top-k
        fine_in_topk = 0
        if fine_name in predicate_names:
            fine_vg_id = predicate_names.index(fine_name)
            for r in fine_rows:
                topk = r.get("topk_predicate_ids", [])
                if fine_vg_id in topk:
                    fine_in_topk += 1

        metrics[fine_name] = {
            "gt_count": gt_count,
            "parent": parent,
            "recall_at_1": correct / gt_count,
            "collapse_rate": collapsed / gt_count,
            "parent_share": parent_share,
            "parent_above_fine_rate": parent_above / gt_count,
            "fine_in_topk_rate": fine_in_topk / gt_count,
            "qualitative_only": gt_count < GT_THRESHOLD,
        }

    return metrics


def compute_family_metrics(pred_metrics, main_map):
    """Compute family-level micro and macro averages."""
    families = defaultdict(list)
    for fine_name, mapping in main_map.items():
        families[mapping["parent"]].append(fine_name)

    family_metrics = {}
    for parent, children in families.items():
        child_metrics = [pred_metrics[c] for c in children if c in pred_metrics]
        if not child_metrics:
            continue

        total_gt = sum(m["gt_count"] for m in child_metrics)
        total_collapsed = sum(
            (m.get("collapse_rate") or 0) * m["gt_count"] for m in child_metrics
        )
        total_errors = sum(
            (1 - (m.get("recall_at_1") or 0)) * m["gt_count"] for m in child_metrics
        )

        macro_recall = np.mean([m.get("recall_at_1") or 0 for m in child_metrics])
        macro_collapse = np.mean([m.get("collapse_rate") or 0 for m in child_metrics])
        macro_parent_share = np.mean([m.get("parent_share") or 0 for m in child_metrics])

        family_metrics[parent] = {
            "children": children,
            "total_gt": total_gt,
            "micro_recall_at_1": sum(
                (m.get("recall_at_1") or 0) * m["gt_count"] for m in child_metrics
            ) / total_gt if total_gt > 0 else 0,
            "micro_collapse_rate": total_collapsed / total_gt if total_gt > 0 else 0,
            "micro_parent_share": total_collapsed / total_errors if total_errors > 0 else 0,
            "macro_recall_at_1": macro_recall,
            "macro_collapse_rate": macro_collapse,
            "macro_parent_share": macro_parent_share,
        }

    return family_metrics


# ---- Negative controls ----

def compute_random_control(rows, main_map, predicate_names, seed=42):
    """Random parent control: shuffle fine->parent mappings.

    For each fine predicate, randomly assign a parent from the pool of all
    predicate names (excluding itself). This prevents infinite loops when
    many predicates share the same parent (e.g. on-family).
    """
    rng = np.random.default_rng(seed)
    fine_names = list(main_map.keys())
    # Candidate pool: all VG150 predicate names except background and the fine name itself
    candidate_parents = [
        n for n in predicate_names
        if n != "__background__"
    ]

    random_map = {}
    for f in fine_names:
        # Pick a random parent that is NOT the fine predicate itself and NOT the true parent
        pool = [p for p in candidate_parents if p != f and p != main_map[f]["parent"]]
        if not pool:
            pool = [p for p in candidate_parents if p != f]  # fallback: only exclude self
        random_parent = rng.choice(pool)
        random_map[f] = {"parent": random_parent}

    metrics = compute_predicate_metrics(rows, random_map, predicate_names)
    rates = [m.get("collapse_rate") or 0 for m in metrics.values() if m["gt_count"] > 0]
    return np.mean(rates) if rates else 0.0


def compute_frequency_matched_control(rows, main_map, predicate_names, predicate_freqs):
    """Frequency-matched negative control.

    For each fine predicate, find a non-family predicate with similar frequency
    and compute its collapse rate to the same parent.
    """
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)

    controls = []
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        fine_id = predicate_names.index(fine_name) if fine_name in predicate_names else -1
        fine_freq = int(predicate_freqs.get(str(fine_id), 0)) if fine_id > 0 else 0

        # Find frequency-similar predicates not in this family
        family = {n for n, m in main_map.items() if m["parent"] == parent}
        candidates = []
        for i, pname in enumerate(predicate_names):
            if pname in ("__background__",) or pname in family or pname == parent:
                continue
            pfreq = int(predicate_freqs.get(str(i), 0))
            if pfreq > 0 and fine_freq > 0:
                ratio = max(pfreq, fine_freq) / max(min(pfreq, fine_freq), 1)
                if ratio <= 3:  # within 3x frequency
                    candidates.append((pname, pfreq, ratio))

        # Pick the closest frequency match
        candidates.sort(key=lambda x: x[2])
        if candidates:
            matched_name = candidates[0][0]
            matched_rows = by_gt.get(matched_name, [])
            if matched_rows:
                collapsed = sum(1 for r in matched_rows if r["pred_predicate_name"] == parent)
                rate = collapsed / len(matched_rows)
                controls.append({
                    "fine_predicate": fine_name,
                    "parent": parent,
                    "control_predicate": matched_name,
                    "control_gt_count": len(matched_rows),
                    "control_freq": candidates[0][1],
                    "fine_freq": fine_freq,
                    "control_collapse_rate": rate,
                })

    avg_rate = np.mean([c["control_collapse_rate"] for c in controls]) if controls else 0.0
    return avg_rate, controls


def compute_sibling_control(rows, main_map, predicate_names):
    """Semantic sibling control: compare parent errors with sibling errors."""
    by_gt = defaultdict(list)
    for r in rows:
        if r.get("matched_pair_found"):
            by_gt[r["gt_predicate_name"]].append(r)

    families = defaultdict(list)
    for fine_name, mapping in main_map.items():
        families[mapping["parent"]].append(fine_name)

    controls = []
    for parent, siblings in families.items():
        if len(siblings) < 2:
            continue
        for fine_name in siblings:
            fine_rows = by_gt.get(fine_name, [])
            if not fine_rows:
                continue
            # Sibling errors: GT=fine, Pred=any other sibling
            sibling_names = [s for s in siblings if s != fine_name]
            sibling_errors = sum(
                1 for r in fine_rows
                if r["pred_predicate_name"] in sibling_names
            )
            parent_errors = sum(
                1 for r in fine_rows
                if r["pred_predicate_name"] == parent
            )
            controls.append({
                "fine_predicate": fine_name,
                "parent": parent,
                "siblings": sibling_names,
                "gt_count": len(fine_rows),
                "parent_error_rate": parent_errors / len(fine_rows),
                "sibling_error_rate": sibling_errors / len(fine_rows),
            })

    return controls


def compute_subject_object_prior_baseline(rows, main_map, predicate_names):
    """Estimate collapse rate from subject-object pair priors.

    Uses empirical P(predicate | subject_class, object_class) from the data.
    """
    # Build co-occurrence counts from the prediction rows (using GT predicates)
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

    # Compute prior-based collapse rate
    controls = []
    for fine_name, mapping in main_map.items():
        parent = mapping["parent"]
        # For each (s,o) pair seen with this fine predicate, what would the prior predict?
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
            controls.append({
                "fine_predicate": fine_name,
                "parent": parent,
                "prior_collapse_rate": prior_parent / prior_total,
                "prior_total": prior_total,
            })

    avg_rate = np.mean([c["prior_collapse_rate"] for c in controls]) if controls else 0.0
    return avg_rate, controls


# ---- Bootstrap CI ----

def bootstrap_collapse_rate(rows, fine_name, parent, n_samples=BOOTSTRAP_SAMPLES):
    """Bootstrap 95% CI for collapse rate."""
    fine_rows = [r for r in rows if r.get("matched_pair_found")
                 and r["gt_predicate_name"] == fine_name]
    n = len(fine_rows)
    if n < 5:
        return None, None, None

    collapsed = np.array([
        1 if r["pred_predicate_name"] == parent else 0
        for r in fine_rows
    ])

    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_samples):
        sample = rng.choice(collapsed, size=n, replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, (100 - BOOTSTRAP_CI) / 2)
    upper = np.percentile(means, 100 - (100 - BOOTSTRAP_CI) / 2)
    return float(lower), float(np.mean(means)), float(upper)


# ---- Visualizations ----

def plot_on_family_confusion(rows, main_map, predicate_names, output_path):
    """Plot on-family confusion heatmap."""
    if not _HAS_MPL:
        print("      WARNING: matplotlib not available, skipping plot")
        return

    on_children = [f for f, m in main_map.items()
                   if m["parent"] == "on" and m.get("confidence") == "strong"]
    if not on_children:
        print("      WARNING: no on-family children in main map, skipping plot")
        return

    # Build confusion matrix: rows=GT, cols=Pred
    labels = ["on"] + on_children + ["near", "behind", "in front of"]
    label_idx = {l: i for i, l in enumerate(labels)}

    matrix = np.zeros((len(on_children), len(labels)))
    row_counts = np.zeros(len(on_children))

    for r in rows:
        if not r.get("matched_pair_found"):
            continue
        gt = r["gt_predicate_name"]
        pred = r["pred_predicate_name"]
        if gt in on_children:
            ri = on_children.index(gt)
            row_counts[ri] += 1
            if pred in label_idx:
                ci = label_idx[pred]
                matrix[ri, ci] += 1

    # Normalize by row
    for i in range(len(on_children)):
        if row_counts[i] > 0:
            matrix[i] /= row_counts[i]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(on_children)))
    ax.set_yticklabels(on_children, fontsize=9)
    ax.set_xlabel("Predicted Predicate")
    ax.set_ylabel("GT Predicate")
    ax.set_title("on-family Confusion (row-normalized)")

    # Annotate cells
    for i in range(len(on_children)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}" if val > 0 else "",
                    ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {output_path}")


def plot_negative_control_comparison(pred_metrics, neg_controls, output_path):
    """Plot main collapse rates vs negative controls."""
    if not _HAS_MPL:
        return

    # Collect per-predicate data for bar chart
    pred_names = []
    main_rates = []
    freq_control_rates = []
    sibling_rates = []

    freq_lookup = {}
    if neg_controls.get("frequency_matched"):
        for c in neg_controls["frequency_matched"]["details"]:
            freq_lookup[c["fine_predicate"]] = c["control_collapse_rate"]

    sibling_lookup = {}
    if neg_controls.get("sibling"):
        for c in neg_controls["sibling"]:
            sibling_lookup[c["fine_predicate"]] = c["sibling_error_rate"]

    for name, m in pred_metrics.items():
        if m["gt_count"] < GT_THRESHOLD:
            continue
        pred_names.append(name)
        main_rates.append(m.get("collapse_rate") or 0)
        freq_control_rates.append(freq_lookup.get(name, 0))
        sibling_rates.append(sibling_lookup.get(name, 0))

    if not pred_names:
        print("      WARNING: no predicates with sufficient samples for control plot")
        return

    x = np.arange(len(pred_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, main_rates, width, label="Model Collapse Rate", color="tab:red")
    ax.bar(x, freq_control_rates, width, label="Freq-Matched Control", color="tab:blue")
    ax.bar(x + width, sibling_rates, width, label="Sibling Error Rate", color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(pred_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Collapse Rate vs Negative Controls")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {output_path}")


# ---- Report generation ----

def generate_report(
    pred_metrics, family_metrics, neg_controls, pred_names,
    main_map, output_path,
):
    """Generate fine_to_coarse_collapse_report.md."""
    lines = []
    lines.append("# Fine-to-Coarse Collapse Report")
    lines.append("")
    lines.append(f"**GT threshold**: {GT_THRESHOLD} (predicates below this are qualitative only)")
    lines.append(f"**Bootstrap**: {BOOTSTRAP_SAMPLES} samples, {BOOTSTRAP_CI}% CI")
    lines.append("")

    # Predicate-level table
    lines.append("## Predicate-Level Metrics")
    lines.append("")
    lines.append("| Predicate | Parent | #GT | Recall@1 ↑ | Collapse ↓ | Parent Share ↓ | ParentAboveFine ↓ | Qualitative |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for name, m in pred_metrics.items():
        if m["gt_count"] == 0:
            continue
        qual = "yes" if m.get("qualitative_only") else ""
        lines.append(
            f"| {name} | {m['parent']} | {m['gt_count']} | "
            f"{m.get('recall_at_1') or 0:.3f} | "
            f"{m.get('collapse_rate') or 0:.3f} | "
            f"{m.get('parent_share') or 0:.3f} | "
            f"{m.get('parent_above_fine_rate') or 0:.3f} | "
            f"{qual} |"
        )

    lines.append("")

    # Family-level table
    lines.append("## Family-Level Metrics")
    lines.append("")
    lines.append("| Family | Children | Total #GT | Micro Recall ↑ | Micro Collapse ↓ | Macro Recall ↑ | Macro Collapse ↓ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for parent, m in family_metrics.items():
        n_children = len(m["children"])
        lines.append(
            f"| {parent}-family | {n_children} | {m['total_gt']} | "
            f"{m['micro_recall_at_1']:.3f} | "
            f"{m['micro_collapse_rate']:.3f} | "
            f"{m['macro_recall_at_1']:.3f} | "
            f"{m['macro_collapse_rate']:.3f} |"
        )

    lines.append("")

    # Negative controls
    lines.append("## Negative Controls")
    lines.append("")

    if neg_controls.get("random"):
        lines.append(f"- **Random parent control**: {neg_controls['random']:.4f}")
    if neg_controls.get("frequency_matched"):
        fm = neg_controls["frequency_matched"]
        lines.append(f"- **Frequency-matched control** (avg): {fm['avg_rate']:.4f}")
    if neg_controls.get("subject_object_prior"):
        so = neg_controls["subject_object_prior"]
        lines.append(f"- **Subject-object prior baseline** (avg): {so['avg_rate']:.4f}")

    lines.append("")

    # Go / Weak-Go / No-Go assessment
    lines.append("## Go / Weak-Go / No-Go Assessment")
    lines.append("")

    # Simple heuristic: if model collapse substantially exceeds controls, Go
    model_avg = np.mean([
        m.get("collapse_rate") or 0 for m in pred_metrics.values()
        if m["gt_count"] >= GT_THRESHOLD
    ])
    random_avg = neg_controls.get("random", 0)
    freq_avg = neg_controls.get("frequency_matched", {}).get("avg_rate", 0)
    so_avg = neg_controls.get("subject_object_prior", {}).get("avg_rate", 0)

    if model_avg > 0 and random_avg > 0:
        ratio_random = model_avg / random_avg
    else:
        ratio_random = 1.0

    if model_avg > 0 and freq_avg > 0:
        ratio_freq = model_avg / freq_avg
    else:
        ratio_freq = 1.0

    lines.append(f"- Model avg collapse rate: {model_avg:.4f}")
    lines.append(f"- Random control: {random_avg:.4f} (ratio: {ratio_random:.2f}x)")
    lines.append(f"- Freq-matched control: {freq_avg:.4f} (ratio: {ratio_freq:.2f}x)")
    lines.append("")

    if ratio_random > 2.0 and ratio_freq > 1.5:
        lines.append("**Assessment: Go** — Model collapse substantially exceeds controls.")
        lines.append("Evidence supports Predicate Orthogonality Bias hypothesis.")
    elif ratio_random > 1.5 or ratio_freq > 1.2:
        lines.append("**Assessment: Weak-Go** — Some evidence but controls are close.")
        lines.append("Need more data points or cross-model validation.")
    else:
        lines.append("**Assessment: No-Go** — Model collapse does not clearly exceed controls.")
        lines.append("Effect may be explained by frequency or object-pair priors.")

    lines.append("")
    lines.append("> Note: This assessment is based on synthetic/dry-run data and is illustrative only.")
    lines.append("> Real assessment requires model predictions on actual test data.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"      Saved: {output_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Compute collapse metrics from relation predictions"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with default synthetic predictions")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to relation_predictions.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Resolve paths
    if args.predictions:
        predictions_path = Path(args.predictions)
    else:
        predictions_path = DEFAULT_PREDICTIONS_PATH

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = predictions_path.parent

    figures_dir = output_dir / "figures"

    print("=" * 70)
    print("Collapse Metrics Computation")
    print("=" * 70)
    print(f"  Predictions: {predictions_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Dry-run:     {args.dry_run}")
    print()

    # 1. Load inputs
    print("[1/8] Loading inputs...")
    if not predictions_path.exists():
        print(f"      ERROR: Predictions file not found: {predictions_path}")
        print(f"      Run Task 04 first or use --dry-run")
        sys.exit(1)

    rows = load_predictions(predictions_path)
    semantic_map, _ = load_semantic_map()
    predicate_freqs, predicate_names = load_predicate_frequencies()
    print(f"      Loaded {len(rows)} prediction rows")
    print(f"      Loaded {len(semantic_map)} semantic mappings")
    print()

    # Filter main mappings
    main_map = {
        k: v for k, v in semantic_map.items()
        if v.get("confidence") == "strong" and v.get("use_in_main") is True
    }
    print(f"      Main mappings (strong + use_in_main): {len(main_map)}")
    print()

    # 2. Compute predicate metrics
    print("[2/8] Computing predicate-level metrics...")
    pred_metrics = compute_predicate_metrics(rows, main_map, predicate_names)
    n_qual = sum(1 for m in pred_metrics.values() if m.get("qualitative_only"))
    n_quant = len(pred_metrics) - n_qual
    print(f"      {n_quant} quantitative, {n_qual} qualitative-only predicates")
    print()

    # 3. Compute family metrics
    print("[3/8] Computing family-level metrics...")
    family_metrics = compute_family_metrics(pred_metrics, main_map)
    for parent, m in family_metrics.items():
        print(f"      {parent}-family: {len(m['children'])} children, "
              f"micro collapse={m['micro_collapse_rate']:.3f}")
    print()

    # 4. Negative controls
    print("[4/8] Computing negative controls...")
    neg_controls = {}

    random_rate = compute_random_control(rows, main_map, predicate_names)
    neg_controls["random"] = random_rate
    print(f"      Random parent control:   {random_rate:.4f}")

    freq_avg, freq_details = compute_frequency_matched_control(
        rows, main_map, predicate_names, predicate_freqs)
    neg_controls["frequency_matched"] = {"avg_rate": freq_avg, "details": freq_details}
    print(f"      Freq-matched control:    {freq_avg:.4f} ({len(freq_details)} pairs)")

    sibling_controls = compute_sibling_control(rows, main_map, predicate_names)
    neg_controls["sibling"] = sibling_controls
    print(f"      Sibling control:          {len(sibling_controls)} pairs")

    so_avg, so_details = compute_subject_object_prior_baseline(
        rows, main_map, predicate_names)
    neg_controls["subject_object_prior"] = {"avg_rate": so_avg, "details": so_details}
    print(f"      Subject-object prior:    {so_avg:.4f} ({len(so_details)} pairs)")
    print()

    # 5. Bootstrap CI
    print("[5/8] Computing bootstrap CI...")
    for name, m in pred_metrics.items():
        if m["gt_count"] < 5:
            continue
        lo, mean, hi = bootstrap_collapse_rate(rows, name, m["parent"])
        m["collapse_ci_lower"] = lo
        m["collapse_ci_upper"] = hi
        if lo is not None:
            print(f"      {name}: {mean:.3f} [{lo:.3f}, {hi:.3f}] (n={m['gt_count']})")
    print()

    # 6. Write CSVs
    print("[6/8] Writing CSV outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predicate-level CSV
    csv_path = output_dir / "predicate_level_collapse.csv"
    with open(csv_path, "w") as f:
        f.write("predicate,parent,gt_count,recall_at_1,collapse_rate,parent_share,"
                "parent_above_fine_rate,fine_in_topk_rate,qualitative_only,"
                "collapse_ci_lower,collapse_ci_upper\n")
        for name, m in pred_metrics.items():
            f.write(
                f"{name},{m['parent']},{m['gt_count']},"
                f"{m.get('recall_at_1') or ''},{m.get('collapse_rate') or ''},"
                f"{m.get('parent_share') or ''},{m.get('parent_above_fine_rate') or ''},"
                f"{m.get('fine_in_topk_rate') or ''},{m.get('qualitative_only')},"
                f"{m.get('collapse_ci_lower') or ''},{m.get('collapse_ci_upper') or ''}\n"
            )
    print(f"      Wrote: {csv_path}")

    # Family-level CSV
    fam_csv = output_dir / "family_level_collapse.csv"
    with open(fam_csv, "w") as f:
        f.write("family,n_children,total_gt,micro_recall,micro_collapse,"
                "micro_parent_share,macro_recall,macro_collapse,macro_parent_share\n")
        for parent, m in family_metrics.items():
            f.write(
                f"{parent},{len(m['children'])},{m['total_gt']},"
                f"{m['micro_recall_at_1']},{m['micro_collapse_rate']},"
                f"{m['micro_parent_share']},"
                f"{m['macro_recall_at_1']},{m['macro_collapse_rate']},"
                f"{m['macro_parent_share']}\n"
            )
    print(f"      Wrote: {fam_csv}")

    # Negative controls CSV
    neg_csv = output_dir / "negative_controls.csv"
    with open(neg_csv, "w") as f:
        f.write("control_type,avg_rate,n_pairs\n")
        f.write(f"random_parent,{neg_controls['random']},1\n")
        f.write(f"frequency_matched,{neg_controls['frequency_matched']['avg_rate']},"
                f"{len(neg_controls['frequency_matched']['details'])}\n")
        f.write(f"subject_object_prior,{neg_controls['subject_object_prior']['avg_rate']},"
                f"{len(neg_controls['subject_object_prior']['details'])}\n")
    print(f"      Wrote: {neg_csv}")

    # Metrics JSON
    metrics_json = output_dir / "collapse_metrics.json"
    serializable_metrics = {}
    for name, m in pred_metrics.items():
        serializable_metrics[name] = {
            k: (float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in m.items()
        }
    with open(metrics_json, "w") as f:
        json.dump({
            "predicate_metrics": serializable_metrics,
            "family_metrics": {k: {kk: float(vv) if isinstance(vv, (np.floating,)) else vv
                                   for kk, vv in v.items()}
                               for k, v in family_metrics.items()},
            "negative_controls": {
                "random": neg_controls["random"],
                "frequency_matched_avg": neg_controls["frequency_matched"]["avg_rate"],
                "subject_object_prior_avg": neg_controls["subject_object_prior"]["avg_rate"],
            },
        }, f, indent=2)
    print(f"      Wrote: {metrics_json}")

    # Statistical reliability JSON
    stats_json = output_dir / "statistical_reliability.json"
    with open(stats_json, "w") as f:
        json.dump({
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence_level": BOOTSTRAP_CI,
            "gt_threshold": GT_THRESHOLD,
            "quantitative_predicates": n_quant,
            "qualitative_only_predicates": n_qual,
        }, f, indent=2)
    print(f"      Wrote: {stats_json}")
    print()

    # 7. Visualizations
    print("[7/8] Generating visualizations...")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_on_family_confusion(
        rows, semantic_map, predicate_names,
        figures_dir / "on_family_confusion.png",
    )
    plot_negative_control_comparison(
        pred_metrics, neg_controls,
        figures_dir / "negative_control_comparison.png",
    )
    print()

    # 8. Report
    print("[8/8] Generating report...")
    generate_report(
        pred_metrics, family_metrics, neg_controls,
        predicate_names, main_map,
        output_dir / "fine_to_coarse_collapse_report.md",
    )
    print()

    print("=" * 70)
    print("Collapse Metrics Summary")
    print("=" * 70)
    print(f"  Quantitative predicates:  {n_quant}")
    print(f"  Qualitative-only:         {n_qual}")
    print(f"  Families:                 {len(family_metrics)}")
    print(f"  Random control:           {neg_controls['random']:.4f}")
    print(f"  Freq-matched control:     {neg_controls['frequency_matched']['avg_rate']:.4f}")
    print(f"  S-O prior control:        {neg_controls['subject_object_prior']['avg_rate']:.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
