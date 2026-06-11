"""
计算 frequency / composition coverage / recall 之间的相关性。

用法:
  python tools/analysis/compute_correlations.py --stats_csv reports/composition/predicate_composition_stats.csv
"""
import json
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def compute_correlations(stats_df: pd.DataFrame, recall_data: dict, output_dir: Path):
    """
    Args:
        stats_df: DataFrame with columns [predicate, frequency, unique_compositions, coverage_ratio, group]
        recall_data: dict mapping predicate -> {"recall@50": ..., "recall@100": ...}
    """
    # Build recall dataframe
    recall_rows = []
    for pred, metrics in recall_data.items():
        if isinstance(metrics, dict):
            recall_rows.append({
                "predicate": pred,
                "recall@50": metrics.get("recall@50", metrics.get("R@50", 0)),
                "recall@100": metrics.get("recall@100", metrics.get("R@100", 0)),
            })
    recall_df = pd.DataFrame(recall_rows)

    if len(recall_df) == 0:
        print("No recall data provided. Skipping correlation analysis.")
        return

    # Merge
    df = stats_df.merge(recall_df, on="predicate", how="inner")
    print(f"Merged {len(df)} predicates with recall data")

    if len(df) < 5:
        print("Too few predicates for correlation analysis.")
        return

    # --- Correlation analysis ---
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: Predicate Stats vs Recall")
    print("=" * 70)

    for recall_col in ["recall@50", "recall@100"]:
        if recall_col not in df.columns or df[recall_col].isna().all():
            continue

        print(f"\n--- {recall_col} ---")
        valid = df[df[recall_col].notna() & (df[recall_col] >= 0)]

        for feature in ["frequency", "unique_compositions", "coverage_ratio"]:
            r_pearson, p_pearson = pearsonr(valid[feature], valid[recall_col])
            r_spearman, p_spearman = spearmanr(valid[feature], valid[recall_col])
            print(f"  {feature}:")
            print(f"    Pearson r={r_pearson:.4f}, p={p_pearson:.2e}")
            print(f"    Spearman r={r_spearman:.4f}, p={p_spearman:.2e}")

    # --- Partial correlation: recall ~ coverage | log_freq ---
    df["log_freq"] = df["frequency"].apply(lambda x: math.log(x + 1))

    if "recall@50" in df.columns:
        print("\n--- Partial Correlation: recall@50 ~ coverage_ratio | log_freq ---")
        try:
            import statsmodels.api as sm
            valid = df[["recall@50", "coverage_ratio", "log_freq"]].dropna()
            if len(valid) > 10:
                # Regress coverage_ratio on log_freq, get residuals
                X_cov = sm.add_constant(valid["log_freq"])
                model_cov = sm.OLS(valid["coverage_ratio"], X_cov).fit()
                cov_resid = model_cov.resid

                # Regress recall on log_freq, get residuals
                model_rec = sm.OLS(valid["recall@50"], X_cov).fit()
                rec_resid = model_rec.resid

                r_partial, p_partial = pearsonr(cov_resid, rec_resid)
                print(f"  Partial r (coverage | freq) vs recall: r={r_partial:.4f}, p={p_partial:.2e}")
                print(f"  (Higher |r| means composition coverage matters beyond frequency)")
            else:
                print(f"  Not enough data points ({len(valid)})")
        except ImportError:
            print("  statsmodels not available, skipping")

    # --- Per-group correlation ---
    print("\n--- Per-Group Correlations ---")
    for group in ["Head", "Body", "Tail"]:
        gdf = df[df["group"] == group]
        if len(gdf) < 3:
            continue
        print(f"\n  {group} ({len(gdf)} predicates):")

        if "recall@50" in df.columns:
            valid = gdf[gdf["recall@50"].notna()]
            if len(valid) >= 3:
                for feature in ["frequency", "unique_compositions", "coverage_ratio"]:
                    r_s, p_s = spearmanr(valid[feature], valid["recall@50"])
                    print(f"    {feature} vs recall@50: Spearman r={r_s:.4f}, p={p_s:.2e}")

    # --- Plot: scatter with regression lines ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"Head": "#e74c3c", "Body": "#f39c12", "Tail": "#3498db"}
    for i, feature in enumerate(["frequency", "unique_compositions", "coverage_ratio"]):
        ax = axes[i]
        for group in ["Head", "Body", "Tail"]:
            gdf = df[(df["group"] == group) & df["recall@50"].notna()]
            if len(gdf) == 0:
                continue
            ax.scatter(gdf[feature], gdf["recall@50"],
                      c=colors[group], label=group, alpha=0.7, s=60,
                      edgecolors="black", linewidth=0.5)
            # Annotate
            for _, row in gdf.iterrows():
                ax.annotate(row["predicate"], (row[feature], row["recall@50"]),
                          fontsize=6, alpha=0.7,
                          textcoords="offset points", xytext=(3, 2))

        if feature == "frequency":
            ax.set_xscale("log")
        ax.set_xlabel(feature.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Recall@50", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Predicate Properties vs Recall@50", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_recall_vs_features.png", dpi=150)
    print(f"\nSaved: {output_dir / 'correlation_recall_vs_features.png'}")

    # --- Summary for paper ---
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)

    if "recall@50" in df.columns:
        valid = df[df["recall@50"].notna()]
        # Top predictor
        best_r = -1
        best_feature = ""
        for feature in ["frequency", "unique_compositions", "coverage_ratio"]:
            r, p = spearmanr(valid[feature], valid["recall@50"])
            if abs(r) > abs(best_r):
                best_r = r
                best_feature = feature
        print(f"1. Strongest predictor of recall: {best_feature} (Spearman r={best_r:.4f})")

        # Head vs Tail recall gap
        for group in ["Head", "Body", "Tail"]:
            gdf = valid[valid["group"] == group]
            if len(gdf) > 0:
                print(f"2. {group} mean recall@50: {gdf['recall@50'].mean():.4f}")

        # Within-group: does composition coverage predict recall after controlling for frequency?
        for group in ["Head", "Body", "Tail"]:
            gdf = valid[valid["group"] == group]
            if len(gdf) >= 5:
                r, p = spearmanr(gdf["coverage_ratio"], gdf["recall@50"])
                print(f"3. {group} coverage_ratio ~ recall@50: Spearman r={r:.4f}, p={p:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Compute correlations between predicate stats and recall")
    parser.add_argument("--stats_csv", type=str, default="reports/composition/predicate_composition_stats.csv")
    parser.add_argument("--recall_json", type=str, default=None,
                        help="JSON file with per-predicate recall (format: {pred: {'recall@50': ..., ...}})")
    parser.add_argument("--output_dir", type=str, default="reports/composition")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stats
    stats_df = pd.read_csv(args.stats_csv)
    print(f"Loaded {len(stats_df)} predicates from {args.stats_csv}")

    # Load recall if provided
    recall_data = {}
    if args.recall_json and Path(args.recall_json).exists():
        with open(args.recall_json) as f:
            recall_data = json.load(f)
        print(f"Loaded recall data for {len(recall_data)} predicates")

    compute_correlations(stats_df, recall_data, output_dir)


if __name__ == "__main__":
    main()
