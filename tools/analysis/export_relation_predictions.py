#!/usr/bin/env python3
"""Export relation-level predictions for Motifs PredCLS.

Produces one JSONL row per GT relation with full predicate score vectors,
top-k predictions, and GT metadata.

Usage:
    # Dry-run: generate synthetic valid JSONL (no checkpoint needed)
    python tools/analysis/export_relation_predictions.py --dry-run

    # Real export with a model checkpoint (requires full project env)
    python tools/analysis/export_relation_predictions.py \
        --ckpt_path outputs/pretrained/motifs/motifs_predcls.ckpt \
        --task PredCLS --model Motifs

Outputs:
    outputs/analysis/fine_to_coarse/<model>/<task>/relation_predictions.jsonl
    outputs/analysis/fine_to_coarse/<model>/<task>/export_summary.json
    outputs/analysis/fine_to_coarse/<model>/<task>/export_schema.md
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Predicate names from canonical VG150 source
_PRED_FREQ_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "predicate_frequencies.json"
_REL_JSON_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "rel.json"

# Object class names from VG150 COCO-format JSON
_TRAIN_JSON_PATH = _PROJECT_ROOT / "data" / "VisualGenome" / "train.json"


def load_predicate_names():
    """Return list of 51 predicate names (index = VG predicate ID, 0 = background)."""
    with open(_REL_JSON_PATH, "r") as f:
        data = json.load(f)
    return data["rel_categories"]


def load_object_class_names():
    """Return dict mapping category_id (1-indexed) -> class name."""
    with open(_TRAIN_JSON_PATH, "r") as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}


# ---- Score extraction (mirrors src.core.metrics._extract_relation_scores) ----

def extract_relation_scores(
    rel_logits,
    rel_nums=50,
    predicate_bg_index="first",
    softmax_scope="all",
):
    """Convert relation logits into [num_pairs, 50] foreground score array.

    Mirrors src.core.metrics._extract_relation_scores without torch dependency
    for dry-run / validation use. When torch is available, delegates to the
    original function for bit-identical results.
    """
    try:
        import torch
        from src.core.metrics import _extract_relation_scores as _orig

        if isinstance(rel_logits, np.ndarray):
            rel_logits = torch.from_numpy(rel_logits)
        return _orig(rel_logits, rel_nums, predicate_bg_index, softmax_scope)
    except ImportError:
        pass

    # Pure-numpy fallback for dry-run / testing
    arr = np.asarray(rel_logits, dtype=np.float64)
    dim = arr.shape[-1]

    def softmax(x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    if dim == rel_nums + 2:
        scores = softmax(arr[:, 1:-1], axis=-1)
    elif dim == rel_nums + 1:
        if predicate_bg_index in ("first", 0):
            if softmax_scope == "all":
                scores = softmax(arr, axis=-1)[:, 1 : rel_nums + 1]
            else:
                scores = softmax(arr[:, 1:], axis=-1)
        else:
            if softmax_scope == "all":
                scores = softmax(arr, axis=-1)[:, :rel_nums]
            else:
                scores = softmax(arr[:, :-1], axis=-1)
    elif dim == rel_nums:
        scores = softmax(arr, axis=-1)
    else:
        raise ValueError(
            f"rel_logits dim mismatch: got {dim}, expected {rel_nums}, "
            f"{rel_nums + 1}, or {rel_nums + 2}"
        )

    return scores


# ---- Synthetic data generation (for --dry-run) ----

def generate_synthetic_predictions(predicate_names, object_names, seed=42):
    """Generate a small set of synthetic relation predictions for dry-run validation.

    Returns a list of dicts, each representing one GT relation with model scores.
    """
    rng = np.random.default_rng(seed)

    # Pick a subset of predicates that form the on-family
    pred_name_to_id = {name: i for i, name in enumerate(predicate_names)}
    on_family = ["sitting on", "standing on", "lying on", "walking on",
                 "parked on", "mounted on"]
    other_preds = [n for n in predicate_names
                   if n not in on_family and n != "__background__"]

    # Pick a few object classes
    obj_ids = sorted(object_names.keys())
    obj_pairs = [(1, 38), (1, 8), (3, 5), (2, 10), (1, 4)]

    rows = []
    num_preds = 50  # foreground only

    for img_id in range(100, 105):
        for rel_idx, (subj_cls, obj_cls) in enumerate(obj_pairs):
            gt_pred_name = on_family[rel_idx % len(on_family)]
            gt_pred_id = pred_name_to_id.get(gt_pred_name, 40)

            # Generate logits biased toward the parent "on" (VG ID 31, score idx 30)
            parent_id = 31  # "on"
            parent_score_idx = parent_id - 1  # 30

            logits = rng.normal(0, 0.5, size=51).astype(np.float64)
            # Bias: parent "on" gets higher logit
            logits[parent_id] = rng.uniform(2.0, 4.0)
            # Fine predicate also gets some signal
            logits[gt_pred_id] = rng.uniform(0.5, 2.0)
            # Add some noise to other on-family members
            for fp_name in on_family:
                fp_id = pred_name_to_id.get(fp_name)
                if fp_id and fp_id != gt_pred_id:
                    logits[fp_id] = rng.uniform(-1.0, 1.0)

            # Extract 50 foreground scores (Motifs: bg at index 0, softmax over all 51)
            scores = extract_relation_scores(
                logits.reshape(1, -1), rel_nums=50,
                predicate_bg_index="first", softmax_scope="all",
            ).flatten()

            # Sort top-k
            topk_indices = np.argsort(-scores)[:10]
            topk_score_indices = topk_indices.tolist()
            topk_vg_ids = [int(i) + 1 for i in topk_score_indices]
            topk_names = [predicate_names[vg_id] for vg_id in topk_vg_ids]
            topk_scores = [float(scores[i]) for i in topk_score_indices]

            # Exclude background from top-k
            topk_vg_ids_filtered = [vg_id for vg_id in topk_vg_ids if vg_id != 0]
            topk_names_filtered = [predicate_names[vg_id] for vg_id in topk_vg_ids_filtered]
            topk_scores_filtered = [
                float(scores[vg_id - 1]) for vg_id in topk_vg_ids_filtered
            ]

            if topk_vg_ids_filtered:
                pred_vg_id = topk_vg_ids_filtered[0]
                pred_name = predicate_names[pred_vg_id]
            else:
                pred_vg_id = 31  # fallback "on"
                pred_name = "on"

            rows.append({
                "image_id": img_id,
                "subject_idx": rel_idx,
                "object_idx": rel_idx + 1,
                "subject_class_id": subj_cls,
                "object_class_id": obj_cls,
                "subject_class_name": object_names.get(subj_cls, f"class_{subj_cls}"),
                "object_class_name": object_names.get(obj_cls, f"class_{obj_cls}"),
                "gt_predicate_id": gt_pred_id,
                "gt_predicate_name": gt_pred_name,
                "pred_predicate_id": pred_vg_id,
                "pred_predicate_name": pred_name,
                "predicate_scores_all": [float(s) for s in scores],
                "topk_predicate_ids": topk_vg_ids_filtered[:5],
                "topk_predicate_names": topk_names_filtered[:5],
                "topk_predicate_scores": [
                    s for s in topk_scores_filtered[:5]
                ],
                "matched_pair_found": True,
                "model": "Motifs",
                "task": "PredCLS",
            })

    return rows


def generate_synthetic_unmatched(predicate_names, object_names):
    """Generate a few rows with matched_pair_found=False for validation coverage."""
    return [
        {
            "image_id": 999,
            "subject_idx": 0,
            "object_idx": 99,
            "subject_class_id": 1,
            "object_class_id": 2,
            "subject_class_name": object_names.get(1, "airplane"),
            "object_class_name": object_names.get(2, "animal"),
            "gt_predicate_id": 40,
            "gt_predicate_name": predicate_names[40],
            "pred_predicate_id": -1,
            "pred_predicate_name": None,
            "predicate_scores_all": [0.0] * 50,
            "topk_predicate_ids": [],
            "topk_predicate_names": [],
            "topk_predicate_scores": [],
            "matched_pair_found": False,
            "model": "Motifs",
            "task": "PredCLS",
        }
    ]


# ---- JSONL validation ----

REQUIRED_FIELDS = [
    "image_id", "subject_idx", "object_idx",
    "subject_class_id", "object_class_id",
    "subject_class_name", "object_class_name",
    "gt_predicate_id", "gt_predicate_name",
    "pred_predicate_id", "pred_predicate_name",
    "predicate_scores_all", "topk_predicate_ids",
    "topk_predicate_names", "topk_predicate_scores",
    "matched_pair_found", "model", "task",
]


def validate_jsonl(rows, predicate_names):
    """Validate JSONL rows against acceptance criteria.

    Returns (errors, warnings, summary).
    """
    errors = []
    warnings = []
    pred_name_set = set(predicate_names)
    num_preds = 50  # foreground only

    total = len(rows)
    matched = sum(1 for r in rows if r.get("matched_pair_found"))
    unmatched = total - matched

    for i, row in enumerate(rows):
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"Row {i}: missing required field '{field}'")

        if row.get("matched_pair_found"):
            # Score array length
            scores = row.get("predicate_scores_all", [])
            if len(scores) != num_preds:
                errors.append(
                    f"Row {i} (image {row.get('image_id')}): "
                    f"predicate_scores_all length is {len(scores)}, expected {num_preds}"
                )

            # Top-k sorted descending
            topk_scores = row.get("topk_predicate_scores", [])
            for j in range(1, len(topk_scores)):
                if topk_scores[j] > topk_scores[j - 1]:
                    errors.append(
                        f"Row {i}: topk_predicate_scores not sorted descending "
                        f"(idx {j}: {topk_scores[j]} > idx {j-1}: {topk_scores[j-1]})"
                    )
                    break

            # Top-k IDs consistent with scores
            topk_ids = row.get("topk_predicate_ids", [])
            topk_scores_check = row.get("topk_predicate_scores", [])
            if len(topk_ids) != len(topk_scores_check):
                errors.append(
                    f"Row {i}: topk_predicate_ids length ({len(topk_ids)}) != "
                    f"topk_predicate_scores length ({len(topk_scores_check)})"
                )

            # No background in top-k
            for pid in topk_ids:
                if pid == 0:
                    errors.append(
                        f"Row {i}: background predicate (id=0) found in topk_predicate_ids"
                    )

            # Predicate IDs are 1..50
            gt_id = row.get("gt_predicate_id")
            if gt_id is not None and (gt_id < 1 or gt_id > 50):
                errors.append(
                    f"Row {i}: gt_predicate_id {gt_id} out of range [1, 50]"
                )

            pred_id = row.get("pred_predicate_id")
            if pred_id is not None and pred_id > 0 and (pred_id < 1 or pred_id > 50):
                errors.append(
                    f"Row {i}: pred_predicate_id {pred_id} out of range [1, 50]"
                )

            # GT/pred names exist in VG150
            gt_name = row.get("gt_predicate_name")
            if gt_name and gt_name not in pred_name_set:
                errors.append(f"Row {i}: gt_predicate_name '{gt_name}' not in VG150")

            # Check pred_predicate_name is None when matched_pair_found=False
        else:
            pred_name = row.get("pred_predicate_name")
            if pred_name is not None:
                errors.append(
                    f"Row {i}: matched_pair_found=false but pred_predicate_name is not None"
                )

    summary = {
        "total_gt_relations": total,
        "exported_rows": total,
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "unmatched_ratio": unmatched / total if total > 0 else 0.0,
        "score_dimension": num_preds,
        "errors": len(errors),
        "warnings": len(warnings),
    }

    return errors, warnings, summary


# ---- Output writers ----

def write_jsonl(rows, output_path):
    """Write rows as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_export_summary(summary, output_path):
    """Write export summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


SCHEMA_MD_TEMPLATE = """# Relation Prediction Export Schema

## File format
One JSON object per line (JSONL). Each line represents one ground-truth relation.

## Fields

| Field | Type | Description |
|---|---|---|
| `image_id` | int | COCO image ID |
| `subject_idx` | int | Subject object index within the image |
| `object_idx` | int | Object object index within the image |
| `subject_class_id` | int | Subject class ID (1..150) |
| `object_class_id` | int | Object class ID (1..150) |
| `subject_class_name` | str | Subject class name |
| `object_class_name` | str | Object class name |
| `gt_predicate_id` | int | Ground-truth predicate ID (1..50) |
| `gt_predicate_name` | str | Ground-truth predicate name |
| `pred_predicate_id` | int | Top-1 predicted predicate ID (1..50), or -1 if unmatched |
| `pred_predicate_name` | str or null | Top-1 predicted predicate name, or null if unmatched |
| `predicate_scores_all` | float[50] | Full 50-way foreground predicate scores |
| `topk_predicate_ids` | int[K] | Top-K predicted predicate IDs (1..50) |
| `topk_predicate_names` | str[K] | Top-K predicted predicate names |
| `topk_predicate_scores` | float[K] | Top-K predicate scores |
| `matched_pair_found` | bool | Whether GT (subj, obj) pair matched a model pair |
| `model` | str | Model name (e.g., "Motifs") |
| `task` | str | Task type (e.g., "PredCLS") |

## Predicate ID scheme
- VG annotation IDs: 1..50
- Background / no-relation: ID 0 (excluded from scores)
- Score index: `vg_predicate_id - 1`
- `predicate_scores_all[i]` corresponds to VG predicate `i + 1`

## Score extraction (Motifs)
- Model output: 51 logits (index 0 = background, indices 1..50 = predicates)
- Softmax over all 51 dimensions
- Foreground scores: slice `[:, 1:51]` (50 values)

## Validation rules
1. `predicate_scores_all` length must be 50 for matched rows.
2. `topk_predicate_ids` must be sorted by descending score.
3. Predicate IDs must be in [1, 50].
4. Background (ID 0) must not appear in top-k.
5. `matched_pair_found=false` rows have empty scores and null pred name.
"""


def write_schema_md(output_path):
    """Write export schema markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(SCHEMA_MD_TEMPLATE)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Export relation-level predictions for Motifs PredCLS"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic valid JSONL and validate (no checkpoint needed)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to model checkpoint (required for real export)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="PredCLS",
        help="Task type (default: PredCLS)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Motifs",
        help="Model name (default: Motifs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: outputs/analysis/fine_to_coarse/<model>/<task>/)",
    )
    parser.add_argument(
        "--test_dataset_size",
        type=int,
        default=0,
        help="Limit test dataset size (0 = full dataset, >0 for small-sample validation)",
    )
    args = parser.parse_args()

    model = args.model
    task = args.task

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _PROJECT_ROOT / "outputs" / "analysis" / "fine_to_coarse" / model / task

    jsonl_path = output_dir / "relation_predictions.jsonl"
    summary_path = output_dir / "export_summary.json"
    schema_path = output_dir / "export_schema.md"

    print("=" * 70)
    print("Relation Prediction Export")
    print("=" * 70)
    print(f"  Model:      {model}")
    print(f"  Task:       {task}")
    print(f"  Output dir: {output_dir}")
    print(f"  Dry-run:    {args.dry_run}")
    print()

    # Load predicate and object names
    print("[1/5] Loading VG150 names...")
    predicate_names = load_predicate_names()
    object_names = load_object_class_names()
    print(f"      Predicates: {len(predicate_names)} (including background)")
    print(f"      Object classes: {len(object_names)}")
    print()

    # Generate or load predictions
    if args.dry_run:
        print("[2/5] Generating synthetic predictions (dry-run)...")
        rows = generate_synthetic_predictions(predicate_names, object_names)
        rows += generate_synthetic_unmatched(predicate_names, object_names)
        print(f"      Generated {len(rows)} synthetic rows")
    elif args.ckpt_path:
        print("[2/5] Running inference with checkpoint...")
        print(f"      ckpt_path: {args.ckpt_path}")
        rows = _export_from_checkpoint(
            args.ckpt_path, predicate_names, object_names,
            task, model, args.test_dataset_size,
        )
    else:
        print("[2/5] ERROR: --ckpt_path is required for real export (or use --dry-run)")
        sys.exit(1)

    print()

    # Validate
    print("[3/5] Validating JSONL...")
    errors, warnings, summary = validate_jsonl(rows, predicate_names)
    print(f"      Total rows:    {summary['total_gt_relations']}")
    print(f"      Matched:       {summary['matched_rows']}")
    print(f"      Unmatched:     {summary['unmatched_rows']}")
    print(f"      Score dim:     {summary['score_dimension']}")

    if errors:
        print(f"      ERRORS: {len(errors)}")
        for err in errors[:10]:
            print(f"        - {err}")
        if len(errors) > 10:
            print(f"        ... and {len(errors) - 10} more")
    else:
        print(f"      PASS: No validation errors")

    if warnings:
        print(f"      Warnings: {len(warnings)}")
        for w in warnings[:5]:
            print(f"        - {w}")
    print()

    # Write outputs
    print("[4/5] Writing output files...")
    write_jsonl(rows, jsonl_path)
    print(f"      Wrote: {jsonl_path}")

    write_export_summary(summary, summary_path)
    print(f"      Wrote: {summary_path}")

    write_schema_md(schema_path)
    print(f"      Wrote: {schema_path}")
    print()

    # Summary
    print("[5/5] Export Summary")
    print("-" * 70)
    print(f"  Status:           {'FAILED' if errors else 'PASSED'}")
    print(f"  GT relations:     {summary['total_gt_relations']}")
    print(f"  Exported rows:    {summary['exported_rows']}")
    print(f"  Matched pairs:    {summary['matched_rows']}")
    print(f"  Unmatched pairs:  {summary['unmatched_rows']}")
    print(f"  Score dimension:  {summary['score_dimension']}")
    print()

    if errors:
        print("Validation FAILED. Fix errors before using exported data.")
        sys.exit(1)

    print("Done.")
    sys.exit(0)


def _export_from_checkpoint(
    ckpt_path, predicate_names, object_names, task, model, test_size
):
    """Real export using a model checkpoint. Requires full project environment."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required for real inference export")
        sys.exit(1)

    # This path requires the full training infrastructure.
    # For the initial PR, the dry-run mode is the primary validation path.
    # Real export will be exercised in Task 07 when a checkpoint is confirmed.
    print("      WARNING: Real inference export requires full project environment.")
    print("      Use --dry-run for format validation without a checkpoint.")
    print("      Real export will be tested in Task 07 (baseline reproduction).")

    # Placeholder: real inference will be implemented when checkpoint is available
    rows = []
    return rows


if __name__ == "__main__":
    main()
