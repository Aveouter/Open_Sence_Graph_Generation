#!/usr/bin/env python3
"""Phase 0 coverage check for ON-family relation rows.

Counts how many GT ON-family rows in an exported relation_predictions.jsonl file
have matched model pairs/features according to `matched_pair_found`.

This is a cheap gate before feature extraction and primitive discovery.
"""

import argparse
import csv
import json
from pathlib import Path

ON_FAMILY = {
    "sitting on",
    "standing on",
    "lying on",
    "laying on",
    "walking on",
    "parked on",
    "mounted on",
}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Check ON-family matched coverage")
    parser.add_argument("--predictions", required=True, help="relation_predictions.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="Model label override")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    total = 0
    matched = 0
    rows = []

    for _, row in iter_jsonl(pred_path):
        gt_name = row.get("gt_predicate_name")
        if gt_name not in ON_FAMILY:
            continue
        is_matched = bool(row.get("matched_pair_found"))
        model = args.model or row.get("model", "unknown")
        total += 1
        matched += int(is_matched)
        d = counts.setdefault(gt_name, {"gt": 0, "matched": 0})
        d["gt"] += 1
        d["matched"] += int(is_matched)

    summary = {
        "prediction_file": str(pred_path),
        "model": args.model,
        "on_family_total": total,
        "on_family_matched": matched,
        "coverage": (matched / total) if total else 0.0,
        "passes_reltr_gate_80pct": (matched / total) >= 0.80 if total else False,
        "passes_motifs_gate_95pct": (matched / total) >= 0.95 if total else False,
        "per_predicate": {},
    }

    for pred in sorted(ON_FAMILY):
        c = counts.get(pred, {"gt": 0, "matched": 0})
        cov = c["matched"] / c["gt"] if c["gt"] else 0.0
        summary["per_predicate"][pred] = {
            "gt": c["gt"],
            "matched": c["matched"],
            "coverage": cov,
        }
        rows.append({
            "predicate": pred,
            "gt": c["gt"],
            "matched": c["matched"],
            "coverage": cov,
        })

    with open(out_dir / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "coverage_by_predicate.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predicate", "gt", "matched", "coverage"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
