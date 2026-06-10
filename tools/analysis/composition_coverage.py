"""
统计 VG150 中每个 predicate 的:
- frequency
- unique (subject_class, object_class) count
- coverage_ratio = unique_so / frequency
- H/B/T group based on frequency

适配 VG150 COCO 格式 + rel.json
"""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse
import math
import numpy as np

# --- Config ---
DATA_ROOT = Path("data/VisualGenome")
OUTPUT_DIR = Path("reports/composition")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_root: Path, split: str = "train"):
    """Load VG150 data in COCO format + rel.json."""
    # Load annotations to get category_id per annotation
    ann_file = data_root / f"{split}.json"
    with open(ann_file) as f:
        coco_data = json.load(f)

    # Build annotation_id -> category_id mapping
    ann_id_to_cat = {}
    ann_id_to_img = {}
    for ann in coco_data["annotations"]:
        ann_id_to_cat[ann["id"]] = ann["category_id"]
        ann_id_to_img[ann["id"]] = ann["image_id"]

    # Build category_id -> category_name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Load relations from rel.json
    rel_file = data_root / "rel.json"
    with open(rel_file) as f:
        rel_data = json.load(f)

    predicate_names = rel_data["rel_categories"]  # index 0 = __background__

    relations = rel_data.get(split, {})
    if isinstance(relations, list):
        # some formats use list
        relations = {img_id: rels for img_id, rels in relations}

    # Build parsed relations
    parsed_relations = []
    missing_ann_ids = set()
    skipped = 0

    for img_id_str, rel_list in relations.items():
        img_id = int(img_id_str)
        for rel in rel_list:
            subj_ann_id, obj_ann_id, pred_id = rel

            if subj_ann_id not in ann_id_to_cat:
                missing_ann_ids.add(subj_ann_id)
                skipped += 1
                continue
            if obj_ann_id not in ann_id_to_cat:
                missing_ann_ids.add(obj_ann_id)
                skipped += 1
                continue

            subj_cat = ann_id_to_cat[subj_ann_id]
            obj_cat = ann_id_to_cat[obj_ann_id]
            pred_name = predicate_names[pred_id] if pred_id < len(predicate_names) else f"unknown_{pred_id}"

            parsed_relations.append({
                "image_id": img_id,
                "subject_class": subj_cat,
                "subject_name": cat_id_to_name.get(subj_cat, f"cls_{subj_cat}"),
                "object_class": obj_cat,
                "object_name": cat_id_to_name.get(obj_cat, f"cls_{obj_cat}"),
                "predicate_id": pred_id,
                "predicate": pred_name,
            })

    if missing_ann_ids:
        print(f"  Warning: {len(missing_ann_ids)} annotation IDs not found in {split}.json, skipped {skipped} relations")

    return parsed_relations, cat_id_to_name, predicate_names


def assign_group_by_rank(rank: int, n_predicates: int) -> str:
    """
    VG150: 50 predicates.
    Head: rank 1-5 (top 10%)
    Body: rank 6-20 (10%-40%)
    Tail: rank 21-50 (40%-100%)
    """
    if rank < 5:
        return "Head"
    elif rank < 20:
        return "Body"
    else:
        return "Tail"


def assign_group_by_freq_cumsum(sorted_freqs, idx, cumsum):
    """
    Alternative: based on cumulative frequency.
    Head: top 50% of total frequency mass
    Body: 50%-90%
    Tail: 90%-100%
    """
    ratio = cumsum[idx] / cumsum[-1] if cumsum[-1] > 0 else 1.0
    if ratio < 0.50:
        return "Head"
    elif ratio < 0.90:
        return "Body"
    else:
        return "Tail"


def main():
    parser = argparse.ArgumentParser(description="VG150 Composition Coverage Analysis")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--split", type=str, default="train", help="Which split to analyze")
    parser.add_argument("--grouping", type=str, default="rank", choices=["rank", "cumulative"],
                        help="How to assign H/B/T groups")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split} split from {data_root}...")
    relations, cat_id_to_name, predicate_names = load_data(data_root, args.split)
    print(f"Loaded {len(relations)} relations")

    # --- 统计每个 predicate ---
    predicate_stats = defaultdict(lambda: {
        "freq": 0,
        "compositions": set(),
        "images": set(),
    })

    for rel in relations:
        pred = rel["predicate"]
        comp = (rel["subject_class"], rel["object_class"])
        predicate_stats[pred]["freq"] += 1
        predicate_stats[pred]["compositions"].add(comp)
        predicate_stats[pred]["images"].add(rel["image_id"])

    # --- Filter out __background__ ---
    if "__background__" in predicate_stats:
        del predicate_stats["__background__"]

    # --- Build DataFrame ---
    rows = []
    for pred, stats in sorted(predicate_stats.items(), key=lambda x: -x[1]["freq"]):
        freq = stats["freq"]
        comp_count = len(stats["compositions"])
        coverage_ratio = comp_count / freq if freq > 0 else 0
        rows.append({
            "predicate": pred,
            "frequency": freq,
            "unique_compositions": comp_count,
            "coverage_ratio": coverage_ratio,
            "unique_images": len(stats["images"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    # --- 计算 cumulative frequency ---
    freq_vals = df["frequency"].values
    cumsum = freq_vals.cumsum()

    # --- 标注 H/B/T ---
    n_pred = len(df)
    if args.grouping == "rank":
        df["group"] = [assign_group_by_rank(i, n_pred) for i in range(n_pred)]
    else:
        df["group"] = [assign_group_by_freq_cumsum(freq_vals, i, cumsum) for i in range(n_pred)]

    # --- 输出统计 ---
    df.to_csv(output_dir / "predicate_composition_stats.csv", index=False)
    print(f"\nTotal predicates (excl background): {len(df)}")

    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            print(f"\n{group}: (empty)")
            continue
        print(f"\n{group} ({len(gdf)} predicates):")
        print(f"  Total frequency: {gdf['frequency'].sum()} ({gdf['frequency'].sum()/df['frequency'].sum()*100:.1f}%)")
        print(f"  Mean frequency: {gdf['frequency'].mean():.1f}")
        print(f"  Median frequency: {gdf['frequency'].median():.1f}")
        print(f"  Mean unique compositions: {gdf['unique_compositions'].mean():.1f}")
        print(f"  Median unique compositions: {gdf['unique_compositions'].median():.1f}")
        print(f"  Mean coverage ratio: {gdf['coverage_ratio'].mean():.4f}")
        print(f"  Median coverage ratio: {gdf['coverage_ratio'].median():.4f}")
        print(f"  Coverage ratio range: [{gdf['coverage_ratio'].min():.4f}, {gdf['coverage_ratio'].max():.4f}]")

    # --- 绘图 1: frequency vs unique compositions ---
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"Head": "#e74c3c", "Body": "#f39c12", "Tail": "#3498db"}
    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        ax.scatter(gdf["frequency"], gdf["unique_compositions"],
                   c=colors[group], label=group, alpha=0.7, s=80, edgecolors="black", linewidth=0.5)
        for _, row in gdf.iterrows():
            ax.annotate(row["predicate"], (row["frequency"], row["unique_compositions"]),
                        fontsize=6.5, alpha=0.75,
                        textcoords="offset points", xytext=(5, 3))

    ax.set_xlabel("Predicate Frequency (log scale)", fontsize=13)
    ax.set_ylabel("Unique (Subject, Object) Compositions (log scale)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_title("VG150: Predicate Frequency vs Composition Diversity", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "freq_vs_composition_coverage.png", dpi=150)
    print(f"\nSaved: {output_dir / 'freq_vs_composition_coverage.png'}")

    # --- 绘图 2: coverage ratio 分布 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, group in enumerate(["Head", "Body", "Tail"]):
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        ax.hist(gdf["coverage_ratio"], bins=25, alpha=0.55, label=group, color=colors[group], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Composition Coverage Ratio (unique SO / frequency)", fontsize=12)
    ax.set_ylabel("Predicate Count", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title("Distribution of Composition Coverage Ratio by H/B/T Group", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_ratio_distribution.png", dpi=150)
    print(f"Saved: {output_dir / 'coverage_ratio_distribution.png'}")

    # --- 绘图 3: Unique compositions distribution ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, group in enumerate(["Head", "Body", "Tail"]):
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        ax.hist(gdf["unique_compositions"], bins=25, alpha=0.55, label=group, color=colors[group], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Unique (Subject, Object) Compositions", fontsize=12)
    ax.set_ylabel("Predicate Count", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title("Distribution of Unique Composition Count by H/B/T Group", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "unique_compositions_distribution.png", dpi=150)
    print(f"Saved: {output_dir / 'unique_compositions_distribution.png'}")

    # --- 绘图 4: Bar chart of top/bottom predicates ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 10 by frequency
    top10 = df.head(10)
    ax = axes[0]
    bars = ax.bar(range(len(top10)), top10["frequency"], color=[colors[g] for g in top10["group"]])
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10["predicate"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Top 10 Most Frequent Predicates", fontsize=13)
    for bar, comp in zip(bars, top10["unique_compositions"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f"comp={comp}", ha='center', fontsize=7, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')

    # Top 10 by composition count
    top10_comp = df.sort_values("unique_compositions", ascending=False).head(10)
    ax = axes[1]
    bars = ax.bar(range(len(top10_comp)), top10_comp["unique_compositions"], color=[colors[g] for g in top10_comp["group"]])
    ax.set_xticks(range(len(top10_comp)))
    ax.set_xticklabels(top10_comp["predicate"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Unique Compositions", fontsize=12)
    ax.set_title("Top 10 Most Compositionally Diverse Predicates", fontsize=13)
    for bar, freq in zip(bars, top10_comp["frequency"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"freq={freq}", ha='center', fontsize=7, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "top_predicates_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'top_predicates_comparison.png'}")

    # --- 输出 key observations 用于论文 ---
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS FOR PAPER")
    print("=" * 60)

    print(f"1. Total relations: {df['frequency'].sum():,}")
    print(f"2. Total unique (S,O) compositions across all predicates: "
          f"{sum(len(predicate_stats[p]['compositions']) for p in predicate_stats)}")

    print(f"\n3. Coverage ratio range: [{df['coverage_ratio'].min():.4f}, {df['coverage_ratio'].max():.4f}]")
    print(f"4. Mean coverage ratio: {df['coverage_ratio'].mean():.4f}")

    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) == 0:
            continue
        print(f"\n5. {group}:")
        print(f"   Predicates: {list(gdf['predicate'].values)}")
        print(f"   Avg freq: {gdf['frequency'].mean():.0f}, "
              f"Avg comp: {gdf['unique_compositions'].mean():.0f}, "
              f"Avg coverage: {gdf['coverage_ratio'].mean():.4f}")

    # 找同频率区间但不同 composition coverage 的例子
    mid_freq = df[(df["frequency"] > 50) & (df["frequency"] < 500)]
    if len(mid_freq) >= 2:
        max_comp = mid_freq.loc[mid_freq["unique_compositions"].idxmax()]
        min_comp = mid_freq.loc[mid_freq["unique_compositions"].idxmin()]
        print(f"\n6. Same freq range (50-500), different coverage:")
        print(f"   High coverage: {max_comp['predicate']} "
              f"(freq={max_comp['frequency']}, comp={max_comp['unique_compositions']}, "
              f"ratio={max_comp['coverage_ratio']:.4f})")
        print(f"   Low coverage: {min_comp['predicate']} "
              f"(freq={min_comp['frequency']}, comp={min_comp['unique_compositions']}, "
              f"ratio={min_comp['coverage_ratio']:.4f})")

    # Composition gap ratio: how much of each predicate's comp space is covered
    print(f"\n7. Predicates with lowest composition coverage (most susceptible to shortcut):")
    bottom10 = df.nsmallest(10, "coverage_ratio")
    for _, row in bottom10.iterrows():
        print(f"   {row['predicate']}: freq={row['frequency']}, comp={row['unique_compositions']}, "
              f"ratio={row['coverage_ratio']:.4f}")

    print(f"\n8. Predicates with highest composition coverage (most compositional):")
    top10 = df.nlargest(10, "coverage_ratio")
    for _, row in top10.iterrows():
        print(f"   {row['predicate']}: freq={row['frequency']}, comp={row['unique_compositions']}, "
              f"ratio={row['coverage_ratio']:.4f}")


if __name__ == "__main__":
    main()
