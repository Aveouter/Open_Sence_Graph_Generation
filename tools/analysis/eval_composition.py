"""
Per-predicate evaluation and composition gap analysis for SGG models.

Usage:
  # After training a model, evaluate and compute composition gap:
  python tools/analysis/eval_composition.py \
    --eval_results outputs/runs/<method>/<run>/eval/sgdet/metrics.json \
    --composition_split data/VisualGenome/composition_splits/split_0/composition_split.json \
    --output reports/composition/<method>_comp_gap.json
"""
import json
import argparse
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_composition_split(split_path: str):
    """Load and flatten composition split into lookup tables."""
    with open(split_path) as f:
        data = json.load(f)

    seen = {}   # pred -> set of (s, o)
    unseen = {}  # pred -> set of (s, o)

    for pred, split_data in data["composition_split"].items():
        seen[pred] = set(tuple(c) for c in split_data["seen"])
        unseen[pred] = set(tuple(c) for c in split_data["unseen"])

    return seen, unseen, data.get("metadata", {}), data.get("group_statistics", {})


def compute_composition_gap(per_predicate_recall: dict,
                            composition_seen: dict,
                            composition_unseen: dict,
                            predicate_groups: dict = None):
    """
    计算每个 predicate 的 composition gap。

    Args:
        per_predicate_recall: {pred_name: {"recall@50": 0.XX, "recall@100": 0.YY, ...}}
        composition_seen: {pred_name: set of (s, o) in train}
        composition_unseen: {pred_name: set of (s, o) in test only}
        predicate_groups: {pred_name: "Head"/"Body"/"Tail"}

    Returns:
        DataFrame with per-predicate gaps
    """
    rows = []
    for pred, metrics in per_predicate_recall.items():
        if pred == "__background__":
            continue
        n_seen = len(composition_seen.get(pred, set()))
        n_unseen = len(composition_unseen.get(pred, set()))
        n_total = n_seen + n_unseen

        row = {
            "predicate": pred,
            "n_seen_compositions": n_seen,
            "n_unseen_compositions": n_unseen,
            "n_total_compositions": n_total,
            "composition_coverage": n_seen / n_total if n_total > 0 else 0,
            "group": predicate_groups.get(pred, "Tail") if predicate_groups else "Unknown",
        }

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Add gap columns (placeholder — actual gap requires seen/unseen recall separately)
    # For now, we compute the expected compositional diversity penalty
    df["log_total_comp"] = df["n_total_compositions"].apply(lambda x: math.log(x + 1))
    df["log_unseen_comp"] = df["n_unseen_compositions"].apply(lambda x: math.log(x + 1))

    return df


def plot_composition_gap(df: pd.DataFrame, output_dir: Path, tag: str = ""):
    """Generate composition gap visualization."""
    if len(df) == 0:
        return

    colors = {"Head": "#e74c3c", "Body": "#f39c12", "Tail": "#3498db"}
    recall_cols = [c for c in df.columns if c.startswith("recall@") or c.startswith("R@")]

    if not recall_cols:
        print("No recall columns found in dataframe, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: n_compositions vs recall
    ax = axes[0]
    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        ax.scatter(gdf["n_total_compositions"], gdf[recall_cols[0]],
                  c=colors[group], label=group, alpha=0.7, s=60,
                  edgecolors="black", linewidth=0.5)
        for _, row in gdf.iterrows():
            ax.annotate(row["predicate"], (row["n_total_compositions"], row[recall_cols[0]]),
                      fontsize=6, alpha=0.7,
                      textcoords="offset points", xytext=(3, 2))
    ax.set_xlabel("Total (S,O) Compositions", fontsize=11)
    ax.set_ylabel(recall_cols[0], fontsize=11)
    ax.set_title("Composition Count vs Recall", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: composition_coverage vs recall
    ax = axes[1]
    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        ax.scatter(gdf["composition_coverage"], gdf[recall_cols[0]],
                  c=colors[group], label=group, alpha=0.7, s=60,
                  edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Composition Coverage (seen/total)", fontsize=11)
    ax.set_ylabel(recall_cols[0], fontsize=11)
    ax.set_title("Composition Coverage vs Recall", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Bar chart of recall by group
    ax = axes[2]
    groups = ["Head", "Body", "Tail"]
    x_pos = np.arange(len(groups))
    means = []
    stds = []
    for group in groups:
        gdf = df[df["group"] == group]
        if len(gdf) > 0 and recall_cols[0] in gdf.columns:
            means.append(gdf[recall_cols[0]].mean())
            stds.append(gdf[recall_cols[0]].std())
        else:
            means.append(0)
            stds.append(0)

    bars = ax.bar(x_pos, means, color=[colors[g] for g in groups],
                  yerr=stds, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel(f"Mean {recall_cols[0]}", fontsize=11)
    ax.set_title(f"Mean {recall_cols[0]} by Predicate Group", fontsize=13)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    suffix = f"_{tag}" if tag else ""
    plt.suptitle(f"Composition Gap Analysis{suffix}", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_dir / f"composition_gap{suffix}.png", dpi=150)
    print(f"Saved: {output_dir / f'composition_gap{suffix}.png'}")


def compute_group_summary(df: pd.DataFrame):
    """Compute per-group summary statistics."""
    recall_cols = [c for c in df.columns if c.startswith("recall@") or c.startswith("R@")]

    summary = {}
    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        summary[group] = {
            "n_predicates": len(gdf),
            "mean_n_compositions": float(gdf["n_total_compositions"].mean()),
            "mean_composition_coverage": float(gdf["composition_coverage"].mean()),
        }
        for col in recall_cols:
            summary[group][f"mean_{col}"] = float(gdf[col].mean())
            summary[group][f"std_{col}"] = float(gdf[col].std())

    return summary


def main():
    parser = argparse.ArgumentParser(description="Composition Gap Analysis for SGG")
    parser.add_argument("--eval_results", type=str, default=None,
                        help="Path to eval metrics JSON (from the framework's save_eval_results)")
    parser.add_argument("--per_predicate_json", type=str, default=None,
                        help="Alternative: per-predicate recall JSON "
                             "(format: {pred_name: {'recall@50': ..., 'recall@100': ...}})")
    parser.add_argument("--composition_split", type=str,
                        default="data/VisualGenome/composition_splits/split_0/composition_split.json")
    parser.add_argument("--coverage_csv", type=str,
                        default="reports/composition/predicate_composition_stats.csv")
    parser.add_argument("--output_dir", type=str, default="reports/composition")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load composition split
    comp_seen, comp_unseen, metadata, group_stats = load_composition_split(args.composition_split)
    print(f"Loaded composition split: {len(comp_seen)} predicates, "
          f"{sum(len(v) for v in comp_seen.values())} seen, "
          f"{sum(len(v) for v in comp_unseen.values())} unseen")

    # Load predicate H/B/T groups
    pred_groups = {}
    stats_df = pd.read_csv(args.coverage_csv)
    for _, row in stats_df.iterrows():
        pred_groups[row["predicate"]] = row["group"]

    # Load recall data
    per_predicate_recall = {}
    if args.per_predicate_json:
        with open(args.per_predicate_json) as f:
            per_predicate_recall = json.load(f)
        print(f"Loaded per-predicate recall for {len(per_predicate_recall)} predicates")
    elif args.eval_results:
        with open(args.eval_results) as f:
            eval_data = json.load(f)
        # Extract per-predicate recall if available
        if "per_predicate" in eval_data:
            per_predicate_recall = eval_data["per_predicate"]
        else:
            print("Warning: eval_results JSON does not contain per_predicate data.")
            print("Run with --per_predicate_json for per-predicate recall.")
    else:
        print("No recall data provided. Generating composition statistics only.")

    # Compute gap analysis
    if per_predicate_recall:
        df = compute_composition_gap(per_predicate_recall, comp_seen, comp_unseen, pred_groups)
        if len(df) > 0:
            df.to_csv(output_dir / f"composition_gap{'_' + args.tag if args.tag else ''}.csv", index=False)
            summary = compute_group_summary(df)
            print("\nGroup Summary:")
            print(json.dumps(summary, indent=2))
            plot_composition_gap(df, output_dir, args.tag)

            # Save summary
            with open(output_dir / f"composition_gap_summary{'_' + args.tag if args.tag else ''}.json", "w") as f:
                json.dump(summary, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
