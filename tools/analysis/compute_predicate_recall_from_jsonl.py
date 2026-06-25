#!/usr/bin/env python3
"""Compute GT-aligned predicate Recall@K and mean Recall@K from JSONL exports.

This is not full SGG R@K; it evaluates whether the ground-truth predicate appears
in each row's top-K predicate list for GT-aligned relation rows.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def iter_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Compute GT-aligned predicate R@K/mR@K")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {k: 0 for k in args.ks}
    hits = {k: 0 for k in args.ks}
    per_pred_total = defaultdict(int)
    per_pred_hits = {k: defaultdict(int) for k in args.ks}
    rows_used = 0
    rows_skipped = 0
    for row in iter_rows(args.predictions):
        if not row.get("matched_pair_found", True):
            rows_skipped += 1
            continue
        gt = int(row.get("gt_predicate_id", -1))
        if gt <= 0:
            rows_skipped += 1
            continue
        topk = [int(x) for x in row.get("topk_predicate_ids", [])]
        if not topk:
            rows_skipped += 1
            continue
        rows_used += 1
        per_pred_total[gt] += 1
        for k in args.ks:
            totals[k] += 1
            if gt in topk[:k]:
                hits[k] += 1
                per_pred_hits[k][gt] += 1

    summary = {
        "prediction_file": args.predictions,
        "rows_used": rows_used,
        "rows_skipped": rows_skipped,
        "recall": {},
        "mean_recall": {},
    }
    for k in args.ks:
        summary["recall"][f"R@{k}"] = hits[k] / totals[k] if totals[k] else 0.0
        vals = []
        for pred, total in per_pred_total.items():
            vals.append(per_pred_hits[k][pred] / total if total else 0.0)
        summary["mean_recall"][f"mR@{k}"] = sum(vals) / len(vals) if vals else 0.0

    with open(out / "predicate_recall_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out / "predicate_recall_by_class.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["predicate_id", "count"] + [f"R@{k}" for k in args.ks]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pred in sorted(per_pred_total):
            row = {"predicate_id": pred, "count": per_pred_total[pred]}
            for k in args.ks:
                row[f"R@{k}"] = per_pred_hits[k][pred] / per_pred_total[pred]
            writer.writerow(row)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
