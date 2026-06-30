#!/usr/bin/env python3
"""EXP-002 aggregation: integrity checks, strata tables, paired differences, bootstrap.

Reads the equal-budget evaluation artifacts produced by train_gen_sgg.py for all
12 runs (4 modes x 3 seeds) and emits:
  1. per-run integrity checks
  2. per-seed + 3-seed mean/std strata tables
  3. Progressive vs {SingleSoftmax, Parallel, Shuffled} paired differences
  4. image-level paired bootstrap 95% CI for gate-relevant metrics

Data facts (computed from records) vs mechanism inferences are kept separate in
the final report.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CANON_TRAIN_SHA = "c2e32118c3ac0b7ba134ac95bc6cd40e244b1e330dc7069d9af1e70ec8ff2515"
CANON_VAL_SHA = "6a2fe647c223a0cf54b5e95c74c2ea51e7ecc16e9dbdc64d72cfd3fbd895f9a3"

# (mode_label, dir_prefix, cli_mode)
MODES = [
    ("SingleSoftmax", "single_softmax", "single_softmax"),
    ("Parallel", "parallel", "parallel"),
    ("Progressive", "progressive", "progressive"),
    ("Shuffled", "shuffled", "shuffled_history"),
]
SEEDS = [42, 123, 2027]
RECALL_KS = [1, 3, 5]

# Pre-eval checkpoint (size, mtime) snapshot captured before re-evaluation.
PRE_EVAL_SNAPSHOT = {
    "parallel_s42_5k": (143907327, 1782375294),
    "parallel_s123_5k": (143907327, 1782378789),
    "parallel_s2027_5k": (143907391, 1782378841),
    "progressive_s42_5k": (143907391, 1782375297),
    "progressive_s123_5k": (143907391, 1782385781),
    "progressive_s2027_5k": (143907391, 1782385864),
    "shuffled_s42_5k": (143907391, 1782376421),
    "shuffled_s123_5k": (143907391, 1782386111),
    "shuffled_s2027_5k": (143907391, 1782385923),
}


def load_parent_map():
    import json as _json

    path = _PROJECT_ROOT / "configs" / "pob_strong_mappings.json"
    with open(path) as f:
        data = _json.load(f)
    parent_map = {}
    for item in data["primary_on_family"]["mappings"]:
        child = int(item["child_id"]) - 1
        parent = int(item["parent_id"]) - 1
        parent_map[child] = parent
    return parent_map


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_dir(prefix, seed):
    return _PROJECT_ROOT / "outputs" / "gen_sgg" / f"{prefix}_s{seed}_5k"


def checkpoint_config_shas(prefix, seed):
    """Read train/val manifest SHAs stored inside the checkpoint config."""
    ckpt_path = run_dir(prefix, seed) / "checkpoint.pt"
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    return cfg.get("train_manifest_sha256"), cfg.get("val_manifest_sha256")


def integrity_check(mode_label, prefix, seed, summary, records, parent_map):
    issues = []
    d = run_dir(prefix, seed)

    # 1. all count == 25727
    all_count = summary["all"]["count"]
    if all_count != 25727:
        issues.append(f"all count={all_count} != 25727")

    # 2. mapped-fine count == 1045
    mapped_count = summary["mapped_fine"]["count"]
    if mapped_count != 1045:
        issues.append(f"mapped-fine count={mapped_count} != 1045")

    # 3 & 4. manifest SHAs from checkpoint config
    try:
        tr_sha, val_sha = checkpoint_config_shas(prefix, seed)
        if tr_sha != CANON_TRAIN_SHA:
            issues.append(f"train manifest SHA mismatch: {tr_sha}")
        if val_sha != CANON_VAL_SHA:
            issues.append(f"val manifest SHA mismatch: {val_sha}")
    except Exception as e:  # noqa: BLE001
        issues.append(f"could not read checkpoint SHAs: {e}")

    # 5. Top-K budget == [1,3,5]
    if summary.get("prediction_budget") != [1, 3, 5]:
        issues.append(f"prediction_budget={summary.get('prediction_budget')}")

    # 6. no NaN/Inf in top5_scores
    bad_scores = 0
    for r in records:
        for s in r["top5_scores"]:
            if not math.isfinite(s):
                bad_scores += 1
    if bad_scores:
        issues.append(f"{bad_scores} non-finite top5_scores")

    # 7. Top-K predicates unique (top5_labels distinct)
    non_unique = 0
    for r in records:
        if len(set(r["top5_labels"])) != len(r["top5_labels"]):
            non_unique += 1
    if non_unique:
        issues.append(f"{non_unique} records with non-unique top5_labels")

    # 8. record count == all count
    if len(records) != all_count:
        issues.append(f"record count {len(records)} != all count {all_count}")

    # 9. single_gt + multi_gt counts == all
    sg = summary["single_gt"]["count"]
    mg = summary["multi_gt"]["count"]
    if sg + mg != all_count:
        issues.append(f"single_gt({sg})+multi_gt({mg}) != all({all_count})")

    # 10. existing checkpoint not overwritten (size+mtime unchanged)
    ckpt_path = d / "checkpoint.pt"
    if ckpt_path.exists():
        st = ckpt_path.stat()
        key = f"{prefix}_s{seed}_5k"
        if key in PRE_EVAL_SNAPSHOT:
            exp_size, exp_mtime = PRE_EVAL_SNAPSHOT[key]
            if st.st_size != exp_size or int(st.st_mtime) != exp_mtime:
                issues.append(
                    f"checkpoint changed: size {st.st_size} (exp {exp_size}), "
                    f"mtime {int(st.st_mtime)} (exp {exp_mtime})"
                )

    return issues


def per_image_metrics(records, parent_map):
    """Aggregate records to per-image micro metrics for bootstrap.

    Returns dict: image_id -> metric dict. Micro = sum hits / sum denominator
    over pairs in that image.
    """
    by_img = defaultdict(list)
    for r in records:
        by_img[r["image_id"]].append(r)

    img_metrics = {}
    for img, pairs in by_img.items():
        m = {}
        topk = {kk: [set(p["top5_labels"][:kk]) for p in pairs] for kk in RECALL_KS}
        obs = [set(p["observed_labels"]) for p in pairs]

        # all-pairs recall/precision/f1
        for kk in RECALL_KS:
            hits = sum(len(topk[kk][i] & obs[i]) for i in range(len(pairs)))
            obs_tot = sum(len(o) for o in obs)
            m[f"all_recall@{kk}"] = hits / obs_tot if obs_tot else 0.0
            m[f"all_precision@{kk}"] = hits / (kk * len(pairs))
            rec = m[f"all_recall@{kk}"]
            prec = m[f"all_precision@{kk}"]
            m[f"all_f1@{kk}"] = (
                (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
            )

        # single-GT recall@1 (accuracy), recall@3/5
        sg = [p for p in pairs if len(p["observed_labels"]) == 1]
        if sg:
            for kk in RECALL_KS:
                hits = sum(
                    len(set(p["top5_labels"][:kk]) & set(p["observed_labels"]))
                    for p in sg
                )
                m[f"sg_recall@{kk}"] = hits / len(sg)
            # precision@3/5 for single-GT
            for kk in [3, 5]:
                hits = sum(
                    len(set(p["top5_labels"][:kk]) & set(p["observed_labels"]))
                    for p in sg
                )
                m[f"sg_precision@{kk}"] = hits / (kk * len(sg))
        else:
            for kk in RECALL_KS:
                m[f"sg_recall@{kk}"] = 0.0
            for kk in [3, 5]:
                m[f"sg_precision@{kk}"] = 0.0

        # multi-GT set recall@1/3/5, precision@3/5, f1@3/5, full-set coverage@3/5
        mg = [p for p in pairs if len(p["observed_labels"]) >= 2]
        if mg:
            for kk in RECALL_KS:
                hits = sum(
                    len(set(p["top5_labels"][:kk]) & set(p["observed_labels"]))
                    for p in mg
                )
                obs_tot = sum(len(p["observed_labels"]) for p in mg)
                m[f"mg_recall@{kk}"] = hits / obs_tot if obs_tot else 0.0
            for kk in [3, 5]:
                hits = sum(
                    len(set(p["top5_labels"][:kk]) & set(p["observed_labels"]))
                    for p in mg
                )
                m[f"mg_precision@{kk}"] = hits / (kk * len(mg))
                rec = m[f"mg_recall@{kk}"]
                prec = m[f"mg_precision@{kk}"]
                m[f"mg_f1@{kk}"] = (
                    (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
                )
                m[f"mg_fullset@{kk}"] = sum(
                    float(set(p["observed_labels"]) <= set(p["top5_labels"][:kk]))
                    for p in mg
                ) / len(mg)
        else:
            for kk in RECALL_KS:
                m[f"mg_recall@{kk}"] = 0.0
            for kk in [3, 5]:
                m[f"mg_precision@{kk}"] = m[f"mg_f1@{kk}"] = m[f"mg_fullset@{kk}"] = 0.0

        # mapped-fine: fine recall@1/3/5, F2C@1, parent-only error@1/3/5
        mapped_pairs = []
        for p in pairs:
            for child in p["observed_labels"]:
                if child in parent_map:
                    mapped_pairs.append((p, child, parent_map[child]))
        if mapped_pairs:
            for kk in RECALL_KS:
                m[f"fine_recall@{kk}"] = sum(
                    float(child in set(p["top5_labels"][:kk]))
                    for (p, child, par) in mapped_pairs
                ) / len(mapped_pairs)
                m[f"parent_only_error@{kk}"] = sum(
                    float(
                        par in set(p["top5_labels"][:kk])
                        and child not in set(p["top5_labels"][:kk])
                    )
                    for (p, child, par) in mapped_pairs
                ) / len(mapped_pairs)
            m["f2c@1"] = sum(
                float(p["top5_labels"][0] == par) for (p, child, par) in mapped_pairs
            ) / len(mapped_pairs)
        else:
            for kk in RECALL_KS:
                m[f"fine_recall@{kk}"] = 0.0
                m[f"parent_only_error@{kk}"] = 0.0
            m["f2c@1"] = 0.0

        img_metrics[img] = m
    return img_metrics


def bootstrap_paired_diff(a_metrics, b_metrics, metric, n_boot=10000, seed=0):
    """Image-level paired bootstrap of mean per-image metric difference (a - b).

    a_metrics/b_metrics: image_id -> metric dict. Paired by shared image_id.
    Returns (point_estimate, ci_low, ci_high, n_pairs).
    """
    common = sorted(set(a_metrics) & set(b_metrics))
    diffs = np.array([a_metrics[i][metric] - b_metrics[i][metric] for i in common])
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    # If n is small, bootstrap; else still fine.
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return (point, ci_low, ci_high, n)


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  nan"
    return f"{x:.{p}f}"


def main():
    parent_map = load_parent_map()

    # Load everything
    data = {}  # (mode_label, seed) -> {summary, records, img_metrics}
    integrity = {}
    for mode_label, prefix, cli in MODES:
        for seed in SEEDS:
            d = run_dir(prefix, seed)
            spath = d / "eval_equal_budget_summary.json"
            rpath = d / "eval_equal_budget_records.jsonl"
            if not spath.exists():
                integrity[(mode_label, seed)] = [f"MISSING {spath}"]
                continue
            summary = json.load(open(spath))
            records = load_records(rpath) if rpath.exists() else []
            issues = integrity_check(
                mode_label, prefix, seed, summary, records, parent_map
            )
            integrity[(mode_label, seed)] = issues
            img_m = per_image_metrics(records, parent_map) if records else {}
            data[(mode_label, seed)] = {
                "summary": summary,
                "records": records,
                "img_metrics": img_m,
            }

    report = []
    out = {"integrity": {}, "tables": {}, "paired": {}, "bootstrap": {}}

    # ---- Integrity ----
    report.append("=" * 78)
    report.append("STEP 6: INTEGRITY CHECKS")
    report.append("=" * 78)
    all_ok = True
    for (ml, seed), issues in integrity.items():
        status = "PASS" if not issues else "FAIL"
        if issues:
            all_ok = False
        out["integrity"][f"{ml}_s{seed}"] = issues
        report.append(f"  [{status}] {ml} seed={seed}")
        for iss in issues:
            report.append(f"        - {iss}")
    report.append(
        f"  Overall integrity: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}"
    )

    # ---- Per-seed tables (from summary, the pooled global metrics) ----
    report.append("")
    report.append("=" * 78)
    report.append("STEP 7: RESULTS TABLES (pooled global metrics from summary JSON)")
    report.append("=" * 78)

    def get_s(ml, seed, *path):
        s = data[(ml, seed)]["summary"]
        for p in path:
            s = s[p]
        return s

    # 7a. All pairs: Recall/Precision/F1 @1/3/5
    report.append("\n--- All pairs (n=25,727) ---")
    # simpler per-k blocks
    for metric_prefix, label, keys in [
        ("observed_recall", "Recall", RECALL_KS),
        ("observed_precision", "Precision", RECALL_KS),
        ("observed_f1", "F1", RECALL_KS),
    ]:
        report.append(f"\n  All-pairs {label}:")
        report.append(
            f"    {'Mode':<14}{'seed':>6}"
            + "".join(f"@{k}:>9".replace(":>9", "") for k in keys)
        )
        for ml, prefix, cli in MODES:
            for seed in SEEDS:
                if (ml, seed) not in data:
                    continue
                vals = [get_s(ml, seed, "all", f"{metric_prefix}@{k}") for k in keys]
                report.append(
                    f"    {ml:<14}{seed:>6}" + "".join(f"{v:>9.4f}" for v in vals)
                )
            # 3-seed mean +/- std
            seed_vals = {
                k: [
                    get_s(ml, s, "all", f"{metric_prefix}@{k}")
                    for s in SEEDS
                    if (ml, s) in data
                ]
                for k in keys
            }
            means = {k: np.mean(seed_vals[k]) for k in keys if seed_vals[k]}
            stds = {
                k: np.std(seed_vals[k], ddof=1) if len(seed_vals[k]) > 1 else 0.0
                for k in keys
                if seed_vals[k]
            }
            report.append(
                f"    {ml + ' mean':<14}{'':>6}"
                + "".join(f"{means[k]:>9.4f}" for k in keys if k in means)
            )
            report.append(
                f"    {ml + ' std':<14}{'':>6}"
                + "".join(f"{stds[k]:>9.4f}" for k in keys if k in stds)
            )

    # 7b. Single-GT
    report.append("\n--- Single-GT pairs (n=24,799) ---")
    report.append(
        f"    {'Mode':<14}{'seed':>6}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'P@3':>9}{'P@5':>9}"
    )
    for ml, prefix, cli in MODES:
        for seed in SEEDS:
            if (ml, seed) not in data:
                continue
            r1 = get_s(ml, seed, "single_gt", "observed_recall@1")
            r3 = get_s(ml, seed, "single_gt", "observed_recall@3")
            r5 = get_s(ml, seed, "single_gt", "observed_recall@5")
            p3 = get_s(ml, seed, "single_gt", "observed_precision@3")
            p5 = get_s(ml, seed, "single_gt", "observed_precision@5")
            report.append(
                f"    {ml:<14}{seed:>6}{r1:>9.4f}{r3:>9.4f}{r5:>9.4f}{p3:>9.4f}{p5:>9.4f}"
            )
        r1s = [
            get_s(ml, s, "single_gt", "observed_recall@1")
            for s in SEEDS
            if (ml, s) in data
        ]
        report.append(f"    {ml + ' mean':<14}{'':>6}{np.mean(r1s):>9.4f}")

    # 7c. Multi-GT
    report.append("\n--- Multi-GT pairs (n=928) ---")
    report.append(
        f"    {'Mode':<14}{'seed':>6}{'SetR@1':>9}{'SetR@3':>9}{'SetR@5':>9}{'P@3':>9}{'F1@3':>9}{'Cov@3':>9}{'Cov@5':>9}"
    )
    for ml, prefix, cli in MODES:
        for seed in SEEDS:
            if (ml, seed) not in data:
                continue
            r1 = get_s(ml, seed, "multi_gt", "observed_recall@1")
            r3 = get_s(ml, seed, "multi_gt", "observed_recall@3")
            r5 = get_s(ml, seed, "multi_gt", "observed_recall@5")
            p3 = get_s(ml, seed, "multi_gt", "observed_precision@3")
            f13 = get_s(ml, seed, "multi_gt", "observed_f1@3")
            c3 = get_s(ml, seed, "multi_gt", "full_set_coverage@3")
            c5 = get_s(ml, seed, "multi_gt", "full_set_coverage@5")
            report.append(
                f"    {ml:<14}{seed:>6}{r1:>9.4f}{r3:>9.4f}{r5:>9.4f}{p3:>9.4f}{f13:>9.4f}{c3:>9.4f}{c5:>9.4f}"
            )
        r3s = [
            get_s(ml, s, "multi_gt", "observed_recall@3")
            for s in SEEDS
            if (ml, s) in data
        ]
        report.append(f"    {ml + ' mean':<14}{'':>6}{'':>9}{np.mean(r3s):>9.4f}")

    # 7d. Mapped-fine
    report.append("\n--- Mapped-fine pairs (n=1,045) ---")
    report.append(
        f"    {'Mode':<14}{'seed':>6}{'FineR@1':>9}{'FineR@3':>9}{'FineR@5':>9}{'F2C@1':>9}{'POE@1':>9}{'POE@3':>9}{'POE@5':>9}"
    )
    for ml, prefix, cli in MODES:
        for seed in SEEDS:
            if (ml, seed) not in data:
                continue
            fr1 = get_s(ml, seed, "mapped_fine", "fine_recall@1")
            fr3 = get_s(ml, seed, "mapped_fine", "fine_recall@3")
            fr5 = get_s(ml, seed, "mapped_fine", "fine_recall@5")
            f2c = get_s(ml, seed, "mapped_fine", "f2c@1")
            poe1 = get_s(ml, seed, "mapped_fine", "parent_only_error@1")
            poe3 = get_s(ml, seed, "mapped_fine", "parent_only_error@3")
            poe5 = get_s(ml, seed, "mapped_fine", "parent_only_error@5")
            report.append(
                f"    {ml:<14}{seed:>6}{fr1:>9.4f}{fr3:>9.4f}{fr5:>9.4f}{f2c:>9.4f}{poe1:>9.4f}{poe3:>9.4f}{poe5:>9.4f}"
            )
        fr1s = [
            get_s(ml, s, "mapped_fine", "fine_recall@1")
            for s in SEEDS
            if (ml, s) in data
        ]
        f2cs = [get_s(ml, s, "mapped_fine", "f2c@1") for s in SEEDS if (ml, s) in data]
        report.append(
            f"    {ml + ' mean':<14}{'':>6}{np.mean(fr1s):>9.4f}{'':>9}{'':>9}{np.mean(f2cs):>9.4f}"
        )

    # ---- Paired differences (per-seed, global pooled metrics) ----
    report.append("")
    report.append("=" * 78)
    report.append(
        "PAIRED DIFFERENCES (Progressive - baseline), per seed, global pooled metrics"
    )
    report.append("=" * 78)

    gate_metrics = [
        ("mapped_fine", "fine_recall@1", "mapped fine recall@1"),
        ("mapped_fine", "f2c@1", "F2C@1"),
        ("mapped_fine", "parent_only_error@1", "parent-only err@1"),
        ("single_gt", "observed_recall@1", "single-GT R@1"),
        ("multi_gt", "observed_recall@3", "multi-GT SetR@3"),
        ("multi_gt", "observed_recall@1", "multi-GT SetR@1"),
        ("multi_gt", "observed_recall@5", "multi-GT SetR@5"),
        ("all", "observed_recall@1", "all-pair R@1"),
        ("all", "observed_recall@3", "all-pair R@3"),
        ("all", "observed_recall@5", "all-pair R@5"),
        ("mapped_fine", "fine_recall@3", "mapped fine recall@3"),
        ("mapped_fine", "fine_recall@5", "mapped fine recall@5"),
    ]

    baselines = ["SingleSoftmax", "Parallel", "Shuffled"]
    for base in baselines:
        report.append(f"\n  Progressive - {base}:")
        report.append(
            f"    {'metric':<24}{'s42':>10}{'s123':>10}{'s2027':>10}{'mean':>10}{'dir':>8}"
        )
        for sec, key, label in gate_metrics:
            diffs = []
            for seed in SEEDS:
                if ("Progressive", seed) in data and (base, seed) in data:
                    d = get_s("Progressive", seed, sec, key) - get_s(
                        base, seed, sec, key
                    )
                    diffs.append(d)
            if len(diffs) == 3:
                mean = np.mean(diffs)
                pos = sum(1 for x in diffs if x > 0)
                neg = sum(1 for x in diffs if x < 0)
                dirn = "+" if pos == 3 else ("-" if neg == 3 else "mix")
                report.append(
                    f"    {label:<24}{diffs[0]:>10.4f}{diffs[1]:>10.4f}{diffs[2]:>10.4f}{mean:>10.4f}{dirn:>8}"
                )
                out["paired"].setdefault(base, {})[label] = {
                    "per_seed": diffs,
                    "mean": float(mean),
                    "direction": dirn,
                }

    # ---- Image-level paired bootstrap ----
    report.append("")
    report.append("=" * 78)
    report.append(
        "IMAGE-LEVEL PAIRED BOOTSTRAP 95% CI (per-image micro metrics, n_boot=10000)"
    )
    report.append("=" * 78)
    report.append(
        "  Note: bootstrap point estimate is mean of per-image metrics; may differ"
    )
    report.append("  slightly from pooled global. CI excludes 0 => direction stable.")
    report.append(
        f"    {'comparison':<28}{'metric':<22}{'point':>9}{'CI_low':>9}{'CI_high':>9}{'n_img':>8}"
    )

    boot_metric_map = [
        ("fine_recall@1", "mapped fine recall@1"),
        ("f2c@1", "F2C@1"),
        ("sg_recall@1", "single-GT R@1"),
        ("mg_recall@3", "multi-GT SetR@3"),
        ("mg_recall@1", "multi-GT SetR@1"),
        ("mg_recall@5", "multi-GT SetR@5"),
        ("all_recall@1", "all-pair R@1"),
        ("all_recall@3", "all-pair R@3"),
        ("fine_recall@3", "mapped fine recall@3"),
    ]
    for base in baselines:
        for metric, label in boot_metric_map:
            # use seed 42 as representative for bootstrap CI reporting per seed
            for seed in SEEDS:
                if ("Progressive", seed) in data and (base, seed) in data:
                    a = data[("Progressive", seed)]["img_metrics"]
                    b = data[(base, seed)]["img_metrics"]
                    point, lo, hi, n = bootstrap_paired_diff(a, b, metric, seed=seed)
                    report.append(
                        f"    {'Prog-' + base + ' s' + str(seed):<28}{label:<22}{point:>9.4f}{lo:>9.4f}{hi:>9.4f}{n:>8}"
                    )
                    out["bootstrap"].setdefault(f"Prog-{base}", {}).setdefault(
                        label, {}
                    )[seed] = {
                        "point": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_images": n,
                    }

    # ---- Seed-direction consistency for gate metrics ----
    report.append("")
    report.append("=" * 78)
    report.append("SEED-DIRECTION CONSISTENCY (Progressive - SingleSoftmax, 3 seeds)")
    report.append("=" * 78)
    for sec, key, label in gate_metrics[:6]:
        diffs = []
        for seed in SEEDS:
            if ("Progressive", seed) in data and ("SingleSoftmax", seed) in data:
                diffs.append(
                    get_s("Progressive", seed, sec, key)
                    - get_s("SingleSoftmax", seed, sec, key)
                )
        if len(diffs) == 3:
            pos = sum(1 for x in diffs if x > 0)
            report.append(
                f"  {label:<24} seeds: {[f'{x:+.4f}' for x in diffs]}  positive in {pos}/3"
            )

    report_text = "\n".join(report)
    print(report_text)

    out_path = _PROJECT_ROOT / "outputs" / "gen_sgg" / "exp002_aggregate_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # also write text
    txt_path = _PROJECT_ROOT / "outputs" / "gen_sgg" / "exp002_aggregate_report.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
