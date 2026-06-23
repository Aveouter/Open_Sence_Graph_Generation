#!/usr/bin/env python3
"""Compute score-space POB diagnosis: Delta = score(parent) - score(fine).

Reads relation prediction JSONL files and frozen POB mappings, computes per-sample
score gaps, aggregate statistics, and statistical tests comparing collapse vs
non-collapse gap distributions.

Usage:
    # Single model
    python tools/analysis/compute_pob_logit_gap.py \
        --predictions outputs/analysis/strong_go/Motifs_full/PredCLS/relation_predictions.jsonl \
        --output_dir reports/pob_logit_gap/Motifs \
        --mappings configs/pob_strong_mappings.json

    # All three models (from repo root)
    for model in Motifs_full RelTR_full EGTR_full; do
        python tools/analysis/compute_pob_logit_gap.py \
            --predictions outputs/analysis/strong_go/${model}/PredCLS/relation_predictions.jsonl \
            --output_dir reports/pob_logit_gap/${model}
    done
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_MAPPINGS_PATH = _PROJECT_ROOT / "configs" / "pob_strong_mappings.json"
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 95
BOOTSTRAP_SEED = 42
TOP_K = 5


# ---- Utilities ----

def safe_mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def bootstrap_ci(values, n_bootstrap=BOOTSTRAP_SAMPLES, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED):
    """Compute bootstrap confidence interval for the mean."""
    vals = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))],
                      dtype=np.float64)
    if len(vals) < 10:
        return None, None
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    low_idx = int((100 - ci) / 2 / 100 * n_bootstrap)
    high_idx = int((100 + ci) / 2 / 100 * n_bootstrap) - 1
    return float(means[low_idx]), float(means[high_idx])


def cohens_d(x, y):
    """Cohen's d effect size."""
    x = np.asarray([float(v) for v in x if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    y = np.asarray([float(v) for v in y if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std < 1e-12:
        return 0.0 if abs(np.mean(x) - np.mean(y)) < 1e-12 else float("inf")
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cliffs_delta(x, y):
    """Cliff's delta non-parametric effect size."""
    x = np.asarray([float(v) for v in x if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    y = np.asarray([float(v) for v in y if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    if len(x) < 1 or len(y) < 1:
        return None
    # Use sampling for large arrays
    if len(x) * len(y) > 1_000_000:
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        idx_x = rng.choice(len(x), size=min(len(x), 1000), replace=False)
        idx_y = rng.choice(len(y), size=min(len(y), 1000), replace=False)
        x_samp, y_samp = x[idx_x], y[idx_y]
    else:
        x_samp, y_samp = x, y
    x_col = x_samp[:, None]
    y_row = y_samp[None, :]
    greater = np.sum(x_col > y_row)
    less = np.sum(x_col < y_row)
    n = len(x_samp) * len(y_samp)
    return float((greater - less) / n)


def mann_whitney_u(x, y):
    """Mann-Whitney U statistic and approximate p-value using normal approximation."""
    x = np.asarray([float(v) for v in x if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    y = np.asarray([float(v) for v in y if v is not None and np.isfinite(float(v))],
                   dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return None, None
    nx, ny = len(x), len(y)
    # Rank all values
    all_vals = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(all_vals)) + 1.0
    # Handle ties: average rank
    sorted_order = np.argsort(all_vals)
    sorted_vals = all_vals[sorted_order]
    tied_ranks = np.ones_like(ranks)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg_rank = np.mean(np.arange(i, j) + 1.0)
            for k in range(i, j):
                tied_ranks[sorted_order[k]] = avg_rank
        else:
            tied_ranks[sorted_order[i]] = i + 1.0
        i = j
    Rx = np.sum(tied_ranks[:nx])
    U1 = Rx - nx * (nx + 1) / 2.0
    U2 = nx * ny - U1
    U = min(U1, U2)
    # Normal approximation
    mean_u = nx * ny / 2.0
    # Correction for ties
    unique_vals, counts = np.unique(all_vals, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / ((nx + ny) * (nx + ny - 1))
    std_u = np.sqrt((nx * ny / 12.0) * ((nx + ny + 1) - tie_correction))
    if std_u < 1e-12:
        return float(U), 0.5
    z = (U - mean_u) / std_u
    # Two-tailed p-value using normal approximation
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return float(U), float(p)


# ---- Data loading ----

def load_mappings(mappings_path):
    """Load frozen POB strong mappings. Returns list of (fine_name, fine_id, parent_name, parent_id, family_status)."""
    with open(mappings_path, "r") as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    source = meta.get("source")
    expected_sha = meta.get("source_sha256")
    if source and expected_sha:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = _PROJECT_ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Mapping source not found for SHA validation: {source_path}")
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Mapping source SHA mismatch for {source_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    mappings = []
    for section_key, status in [("primary_on_family", "primary"), ("boundary_in_family", "boundary")]:
        if section_key in data:
            section = data[section_key]
            parent = section["parent"]
            parent_id = section["parent_id"]
            for m in section["mappings"]:
                mappings.append({
                    "fine": m["child"],
                    "fine_id": m["child_id"],
                    "parent": m["parent"],
                    "parent_id": m["parent_id"],
                    "family": "on-family" if parent == "on" else "in-family",
                    "status": status,
                })
    return mappings


def load_predicate_id_map():
    """Build name->id map from VG150 rel_categories."""
    rel_json = _PROJECT_ROOT / "data" / "VisualGenome" / "rel.json"
    with open(rel_json, "r") as f:
        data = json.load(f)
    rel_cats = data["rel_categories"]
    # rel_categories[0] = __background__, [1..50] = predicates
    # In predicate_scores_all, index i corresponds to rel_categories[i+1]
    name_to_id = {name: idx for idx, name in enumerate(rel_cats)}
    return name_to_id


# ---- Gap computation ----

def compute_gaps(predictions_path, mappings, predicate_id_map):
    """Iterate prediction JSONL and compute Delta = score(parent) - score(fine) for relevant samples.

    Returns:
        samples: list of dicts with per-sample data
        gaps_by_mapping: dict mapping fine_name -> list of gap values
        collapse_gaps_by_mapping: dict mapping fine_name -> list of gap values for collapse samples only
        non_collapse_gaps_by_mapping: dict mapping fine_name -> list of gap values for non-collapse
        correct_gaps_by_mapping: dict mapping fine_name -> list of gap values for correct fine predictions
    """
    mapping_lookup = {m["fine_id"]: m for m in mappings}
    relevant_fine_ids = set(mapping_lookup.keys())

    samples = []
    gaps_by_mapping = defaultdict(list)
    collapse_gaps_by_mapping = defaultdict(list)
    non_collapse_gaps_by_mapping = defaultdict(list)
    correct_gaps_by_mapping = defaultdict(list)

    with open(predictions_path, "r") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            gt_id = rec["gt_predicate_id"]
            if gt_id not in relevant_fine_ids:
                continue

            mapping = mapping_lookup[gt_id]
            fine_name = mapping["fine"]
            parent_name = mapping["parent"]
            parent_id = mapping["parent_id"]
            fine_id = mapping["fine_id"]

            scores = rec.get("predicate_scores_all")
            if scores is None or len(scores) < max(parent_id, fine_id):
                print(f"ERROR: line {line_idx}: missing or too-short predicate_scores_all for {fine_name}")
                continue

            # scores array is 0-indexed, but predicate IDs are 1-indexed in rel_categories
            # predicate_scores_all[i] corresponds to rel_categories[i+1]
            # So score for predicate with id=X is at index X-1
            score_parent = float(scores[parent_id - 1]) if parent_id > 0 else None
            score_fine = float(scores[fine_id - 1]) if fine_id > 0 else None

            if score_parent is None or score_fine is None:
                continue

            delta = score_parent - score_fine
            parent_above_fine = 1 if score_parent > score_fine else 0

            pred_id = rec.get("pred_predicate_id", -1)
            pred_name = rec.get("pred_predicate_name", None)
            is_collapse = 1 if (pred_name == parent_name) else 0
            is_correct = 1 if (pred_name == fine_name) else 0

            topk_ids = rec.get("topk_predicate_ids", [])
            fine_in_topk = 1 if fine_id in topk_ids else 0

            sample = {
                "line_idx": line_idx,
                "image_id": rec["image_id"],
                "gt_predicate": fine_name,
                "gt_predicate_id": gt_id,
                "parent_predicate": parent_name,
                "parent_predicate_id": parent_id,
                "pred_predicate": pred_name,
                "pred_predicate_id": pred_id,
                "score_parent": score_parent,
                "score_fine": score_fine,
                "delta": delta,
                "parent_above_fine": parent_above_fine,
                "is_collapse": is_collapse,
                "is_correct": is_correct,
                "fine_in_topk": fine_in_topk,
                "subject_name": rec.get("subject_class_name", ""),
                "object_name": rec.get("object_class_name", ""),
            }
            samples.append(sample)
            gaps_by_mapping[fine_name].append(delta)
            if is_collapse:
                collapse_gaps_by_mapping[fine_name].append(delta)
            else:
                non_collapse_gaps_by_mapping[fine_name].append(delta)
            if is_correct:
                correct_gaps_by_mapping[fine_name].append(delta)

    return samples, gaps_by_mapping, collapse_gaps_by_mapping, non_collapse_gaps_by_mapping, correct_gaps_by_mapping


# ---- Statistics ----

def compute_stats(values, label=""):
    """Compute descriptive statistics for a list of values."""
    vals = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))],
                      dtype=np.float64)
    n = len(vals)
    if n == 0:
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "min": None, "max": None, "q25": None, "q75": None,
            "ci95_low": None, "ci95_high": None,
        }
    mean_v = float(np.mean(vals))
    median_v = float(np.median(vals))
    std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    ci_low, ci_high = bootstrap_ci(vals, n_bootstrap=BOOTSTRAP_SAMPLES, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED)
    return {
        "n": n, "mean": mean_v, "median": median_v, "std": std_v,
        "min": min_v, "max": max_v, "q25": q25, "q75": q75,
        "ci95_low": ci_low, "ci95_high": ci_high,
    }


def compute_group_stats(samples, gaps, collapse_gaps, non_collapse_gaps, correct_gaps, parent_above_fine_count, fine_in_topk_count, n_total):
    """Compute all required statistics for a group of samples."""
    stats = {
        "all_gt_fine": compute_stats(gaps),
        "collapse": compute_stats(collapse_gaps),
        "non_collapse": compute_stats(non_collapse_gaps),
        "correct_fine": compute_stats(correct_gaps),
        "parent_above_fine": parent_above_fine_count,
        "parent_above_fine_rate": parent_above_fine_count / n_total if n_total > 0 else None,
        "fine_in_topk": fine_in_topk_count,
        "fine_in_topk_rate": fine_in_topk_count / n_total if n_total > 0 else None,
    }
    return stats


# ---- Statistical tests ----

def run_statistical_tests(collapse_gaps, non_collapse_gaps, fine_name):
    """Compare Delta distributions between collapse and non-collapse groups."""
    c_vals = [float(v) for v in collapse_gaps if v is not None and np.isfinite(float(v))]
    nc_vals = [float(v) for v in non_collapse_gaps if v is not None and np.isfinite(float(v))]
    nc, nnc = len(c_vals), len(nc_vals)

    results = {
        "mapping": fine_name,
        "fine_predicate": fine_name,
        "n_collapse": nc,
        "n_non_collapse": nnc,
        "mean_collapse": safe_mean(c_vals),
        "mean_non_collapse": safe_mean(nc_vals),
        "median_collapse": float(np.median(c_vals)) if nc > 0 else None,
        "median_non_collapse": float(np.median(nc_vals)) if nnc > 0 else None,
    }

    if nc >= 2 and nnc >= 2:
        results["mean_difference"] = results["mean_collapse"] - results["mean_non_collapse"]
        results["mean_diff"] = results["mean_difference"]
        results["median_difference"] = results["median_collapse"] - results["median_non_collapse"]
        results["cohens_d"] = cohens_d(c_vals, nc_vals)
        u_stat, p_value = mann_whitney_u(c_vals, nc_vals)
        results["mann_whitney_u"] = u_stat
        results["mann_whitney_p"] = p_value
        results["cliffs_delta"] = cliffs_delta(c_vals, nc_vals)

        # Bootstrap CI for mean difference
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        diffs = []
        for _ in range(BOOTSTRAP_SAMPLES):
            c_sample = rng.choice(c_vals, size=nc, replace=True)
            nc_sample = rng.choice(nc_vals, size=nnc, replace=True)
            diffs.append(np.mean(c_sample) - np.mean(nc_sample))
        diffs = np.sort(diffs)
        low_idx = int((100 - BOOTSTRAP_CI) / 2 / 100 * BOOTSTRAP_SAMPLES)
        high_idx = int((100 + BOOTSTRAP_CI) / 2 / 100 * BOOTSTRAP_SAMPLES) - 1
        results["mean_diff_ci95_low"] = float(diffs[low_idx])
        results["mean_diff_ci95_high"] = float(diffs[high_idx])
        results["insufficient_data"] = False
    else:
        results["mean_difference"] = None
        results["mean_diff"] = None
        results["median_difference"] = None
        results["cohens_d"] = None
        results["mann_whitney_u"] = None
        results["mann_whitney_p"] = None
        results["cliffs_delta"] = None
        results["mean_diff_ci95_low"] = None
        results["mean_diff_ci95_high"] = None
        results["insufficient_data"] = True
        results["missing_test_reason"] = (
            f"Insufficient samples: collapse={nc}, non_collapse={nnc}. "
            "Need >=2 in each group for statistical tests."
        )

    return results


# ---- Output writing ----

def write_outputs(output_dir, model_name, samples, gaps_by_mapping, collapse_gaps_by_mapping,
                  non_collapse_gaps_by_mapping, correct_gaps_by_mapping, mappings, all_test_results):
    """Write all output files for one model."""
    os.makedirs(output_dir, exist_ok=True)

    # --- gap_summary.csv ---
    summary_rows = []
    all_gaps = []
    all_collapse_gaps = []
    all_non_collapse_gaps = []
    all_correct_gaps = []
    total_parent_above = 0
    total_fine_in_topk = 0
    total_n = 0
    total_collapse_n = 0

    for m in mappings:
        fine = m["fine"]
        gaps = gaps_by_mapping.get(fine, [])
        c_gaps = collapse_gaps_by_mapping.get(fine, [])
        nc_gaps = non_collapse_gaps_by_mapping.get(fine, [])
        corr_gaps = correct_gaps_by_mapping.get(fine, [])

        n = len(gaps)
        n_collapse = len(c_gaps)
        collapse_rate = n_collapse / n if n > 0 else None
        paf_count = sum(1 for s in samples if s["gt_predicate"] == fine and s["parent_above_fine"])
        fitk_count = sum(1 for s in samples if s["gt_predicate"] == fine and s["fine_in_topk"])

        stats_all = compute_stats(gaps)
        stats_collapse = compute_stats(c_gaps)
        stats_nc = compute_stats(nc_gaps)

        row = {
            "model": model_name,
            "fine_predicate": fine,
            "parent_predicate": m["parent"],
            "family": m["family"],
            "status": m["status"],
            "n": n,
            "collapse_rate": collapse_rate,
            "mean_delta": stats_all["mean"],
            "median_delta": stats_all["median"],
            "std_delta": stats_all["std"],
            "min_delta": stats_all["min"],
            "max_delta": stats_all["max"],
            "q25_delta": stats_all["q25"],
            "q75_delta": stats_all["q75"],
            "ci95_low": stats_all["ci95_low"],
            "ci95_high": stats_all["ci95_high"],
            "parent_above_fine": paf_count,
            "parent_above_fine_rate": paf_count / n if n > 0 else None,
            "fine_in_topk": fitk_count,
            "fine_in_topk_rate": fitk_count / n if n > 0 else None,
            "mean_delta_collapse": stats_collapse["mean"],
            "mean_delta_non_collapse": stats_nc["mean"],
        }
        summary_rows.append(row)

        all_gaps.extend(gaps)
        all_collapse_gaps.extend(c_gaps)
        all_non_collapse_gaps.extend(nc_gaps)
        all_correct_gaps.extend(corr_gaps)
        total_parent_above += paf_count
        total_fine_in_topk += fitk_count
        total_n += n
        total_collapse_n += n_collapse

    # Aggregate rows
    on_family_gaps = []
    on_family_collapse = []
    on_family_non_collapse = []
    in_family_gaps = []
    in_family_collapse = []
    in_family_non_collapse = []

    for m in mappings:
        fine = m["fine"]
        if m["family"] == "on-family":
            on_family_gaps.extend(gaps_by_mapping.get(fine, []))
            on_family_collapse.extend(collapse_gaps_by_mapping.get(fine, []))
            on_family_non_collapse.extend(non_collapse_gaps_by_mapping.get(fine, []))
        else:
            in_family_gaps.extend(gaps_by_mapping.get(fine, []))
            in_family_collapse.extend(collapse_gaps_by_mapping.get(fine, []))
            in_family_non_collapse.extend(non_collapse_gaps_by_mapping.get(fine, []))

    # On-family aggregate
    on_family_stats = compute_stats(on_family_gaps)
    on_family_collapse_stats = compute_stats(on_family_collapse)
    on_family_nc_stats = compute_stats(on_family_non_collapse)
    on_family_paf = sum(1 for s in samples if s["parent_above_fine"] and any(
        m["fine"] == s["gt_predicate"] and m["family"] == "on-family" for m in mappings))
    on_family_fitk = sum(1 for s in samples if s["fine_in_topk"] and any(
        m["fine"] == s["gt_predicate"] and m["family"] == "on-family" for m in mappings))
    on_family_n = len(on_family_gaps)
    on_family_cn = len(on_family_collapse)

    summary_rows.append({
        "model": model_name,
        "fine_predicate": "ON-FAMILY AGGREGATE",
        "parent_predicate": "on",
        "family": "on-family",
        "status": "aggregate",
        "n": on_family_n,
        "collapse_rate": on_family_cn / on_family_n if on_family_n > 0 else None,
        "mean_delta": on_family_stats["mean"],
        "median_delta": on_family_stats["median"],
        "std_delta": on_family_stats["std"],
        "min_delta": on_family_stats["min"],
        "max_delta": on_family_stats["max"],
        "q25_delta": on_family_stats["q25"],
        "q75_delta": on_family_stats["q75"],
        "ci95_low": on_family_stats["ci95_low"],
        "ci95_high": on_family_stats["ci95_high"],
        "parent_above_fine": on_family_paf,
        "parent_above_fine_rate": on_family_paf / on_family_n if on_family_n > 0 else None,
        "fine_in_topk": on_family_fitk,
        "fine_in_topk_rate": on_family_fitk / on_family_n if on_family_n > 0 else None,
        "mean_delta_collapse": on_family_collapse_stats["mean"],
        "mean_delta_non_collapse": on_family_nc_stats["mean"],
    })

    # Write gap_summary.csv
    summary_path = os.path.join(output_dir, "gap_summary.csv")
    fieldnames = [
        "model", "fine_predicate", "parent_predicate", "family", "status",
        "n", "collapse_rate",
        "mean_delta", "median_delta", "std_delta", "min_delta", "max_delta",
        "q25_delta", "q75_delta", "ci95_low", "ci95_high",
        "parent_above_fine", "parent_above_fine_rate",
        "fine_in_topk", "fine_in_topk_rate",
        "mean_delta_collapse", "mean_delta_non_collapse",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Wrote {summary_path} ({len(summary_rows)} rows)")

    # --- gap_by_predicate.csv ---
    by_pred_rows = []
    for m in mappings:
        fine = m["fine"]
        gaps = gaps_by_mapping.get(fine, [])
        n = len(gaps)
        if n == 0:
            continue
        c_gaps = collapse_gaps_by_mapping.get(fine, [])
        vals = np.asarray(gaps, dtype=np.float64)
        paf_count = sum(1 for s in samples if s["gt_predicate"] == fine and s["parent_above_fine"])
        fitk_count = sum(1 for s in samples if s["gt_predicate"] == fine and s["fine_in_topk"])
        cn = len(c_gaps)
        stats = compute_stats(gaps)
        by_pred_rows.append({
            "model": model_name,
            "fine_predicate": fine,
            "parent_predicate": m["parent"],
            "n": n,
            "mean_delta": stats["mean"],
            "median_delta": stats["median"],
            "parent_above_fine": paf_count / n if n > 0 else None,
            "parent_above_fine_numer": paf_count,
            "fine_in_topk": fitk_count / n if n > 0 else None,
            "fine_in_topk_numer": fitk_count,
            "collapse_rate": cn / n if n > 0 else None,
            "collapse_numer": cn,
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
        })
    by_pred_path = os.path.join(output_dir, "gap_by_predicate.csv")
    bp_fieldnames = [
        "model", "fine_predicate", "parent_predicate", "n",
        "mean_delta", "median_delta",
        "parent_above_fine", "parent_above_fine_numer",
        "fine_in_topk", "fine_in_topk_numer",
        "collapse_rate", "collapse_numer",
        "ci95_low", "ci95_high",
    ]
    with open(by_pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bp_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_pred_rows)
    print(f"  Wrote {by_pred_path} ({len(by_pred_rows)} rows)")

    # --- gap_samples.jsonl (first 200 per mapping + all collapse) ---
    samples_path = os.path.join(output_dir, "gap_samples.jsonl")
    written_indices = set()
    with open(samples_path, "w") as f:
        # First write all collapse samples
        for s in samples:
            if s["is_collapse"]:
                f.write(json.dumps(s) + "\n")
                written_indices.add(s["line_idx"])
        # Then write up to 200 non-collapse per mapping
        per_mapping_count = defaultdict(int)
        for s in samples:
            if s["line_idx"] in written_indices:
                continue
            fine = s["gt_predicate"]
            if per_mapping_count[fine] < 200:
                f.write(json.dumps(s) + "\n")
                written_indices.add(s["line_idx"])
                per_mapping_count[fine] += 1
    print(f"  Wrote {samples_path} ({len(written_indices)} samples)")

    return {
        "model": model_name,
        "summary_rows": summary_rows,
        "by_pred_rows": by_pred_rows,
        "total_samples": total_n,
    }


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Compute POB score-space logit gap analysis")
    parser.add_argument("--predictions", required=True,
                        help="Path to relation_predictions.jsonl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for reports (e.g., reports/pob_logit_gap/Motifs)")
    parser.add_argument("--mappings", default=str(DEFAULT_MAPPINGS_PATH),
                        help="Path to pob_strong_mappings.json")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                        help=f"Bootstrap random seed (default: {BOOTSTRAP_SEED})")
    parser.add_argument("--extract_name", default=None,
                        help="Model name override (auto-detected from predictions path if not set)")
    args = parser.parse_args()

    # Note: Bootstrap seed is fixed at 42 per acceptance criteria.
    # All bootstrap_ci calls use the module-level BOOTSTRAP_SEED constant.

    # Validate inputs
    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.mappings):
        print(f"ERROR: Mappings file not found: {args.mappings}")
        sys.exit(1)

    # Auto-detect model name
    if args.extract_name:
        model_name = args.extract_name
    else:
        # Derive from path: .../Motifs_full/... -> Motifs
        p = args.predictions
        if "Motifs" in p:
            model_name = "Motifs"
        elif "RelTR" in p:
            model_name = "RelTR"
        elif "EGTR" in p:
            model_name = "EGTR"
        else:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))

    print(f"=== POB Logit Gap Analysis: {model_name} ===")
    print(f"  Predictions: {args.predictions}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Mappings:    {args.mappings}")
    print(f"  Seed:        {BOOTSTRAP_SEED}")

    # Load data
    mappings = load_mappings(args.mappings)
    predicate_id_map = load_predicate_id_map()
    print(f"  Loaded {len(mappings)} mappings ({sum(1 for m in mappings if m['status']=='primary')} primary, {sum(1 for m in mappings if m['status']=='boundary')} boundary)")

    # Check score-vector availability
    print("  Checking score-vector availability...")
    with open(args.predictions, "r") as f:
        first_line = json.loads(f.readline())
    if "predicate_scores_all" not in first_line:
        print("ERROR: Predictions file lacks 'predicate_scores_all' field. Cannot compute score gaps.")
        print("       This file was not exported with score vectors. Re-export with export_relation_predictions.py.")
        sys.exit(1)
    if len(first_line["predicate_scores_all"]) < 50:
        print(f"ERROR: predicate_scores_all has only {len(first_line['predicate_scores_all'])} dimensions (expected 50).")
        sys.exit(1)
    print(f"  Score vectors available: {len(first_line['predicate_scores_all'])}-dimensional")

    # Compute gaps
    print("  Computing Delta = score(parent) - score(fine)...")
    samples, gaps_by_mapping, collapse_gaps, non_collapse_gaps, correct_gaps = compute_gaps(
        args.predictions, mappings, predicate_id_map
    )
    total_relevant = len(samples)
    print(f"  Processed {total_relevant} GT=fine samples across {len(mappings)} mappings")

    # Run statistical tests per mapping
    print("  Running statistical tests...")
    all_test_results = []
    for m in mappings:
        fine = m["fine"]
        c_gaps = collapse_gaps.get(fine, [])
        nc_gaps = non_collapse_gaps.get(fine, [])
        test_results = run_statistical_tests(c_gaps, nc_gaps, fine)
        all_test_results.append(test_results)

    # Also run aggregate on-family and in-family tests
    on_family_c = []
    on_family_nc = []
    for m in mappings:
        if m["family"] == "on-family":
            on_family_c.extend(collapse_gaps.get(m["fine"], []))
            on_family_nc.extend(non_collapse_gaps.get(m["fine"], []))
    if on_family_c or on_family_nc:
        agg_test = run_statistical_tests(on_family_c, on_family_nc, "ON-FAMILY AGGREGATE")
        all_test_results.append(agg_test)

    # Write outputs
    print("  Writing outputs...")
    result_info = write_outputs(
        args.output_dir, model_name, samples,
        gaps_by_mapping, collapse_gaps, non_collapse_gaps, correct_gaps,
        mappings, all_test_results
    )

    # Write statistical tests CSV
    tests_path = os.path.join(args.output_dir, "statistical_tests.csv")
    test_fieldnames = [
        "mapping", "fine_predicate", "n_collapse", "n_non_collapse",
        "mean_collapse", "mean_non_collapse", "mean_difference", "mean_diff",
        "median_collapse", "median_non_collapse", "median_difference",
        "cohens_d", "mann_whitney_u", "mann_whitney_p", "cliffs_delta",
        "mean_diff_ci95_low", "mean_diff_ci95_high",
        "insufficient_data", "missing_test_reason",
    ]
    with open(tests_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_test_results)
    print(f"  Wrote {tests_path} ({len(all_test_results)} rows)")

    print(f"\n  Done. Outputs in {args.output_dir}/")
    print(f"    gap_summary.csv       - per-predicate + aggregate summary")
    print(f"    gap_by_predicate.csv   - detailed per-predicate stats")
    print(f"    gap_samples.jsonl      - sample-level gap data")
    print(f"    statistical_tests.csv  - collapse vs non-collapse tests")


if __name__ == "__main__":
    main()
