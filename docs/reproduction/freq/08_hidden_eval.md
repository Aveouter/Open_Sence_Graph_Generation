# FREQ Hidden-Positive Evaluation

Adaptation:

- `tools/analysis/export_relation_predictions.py` now supports `--method FREQ`
  without `--ckpt_path`, because FREQ is a deterministic pair prior.
- The exported JSONL uses the existing relation prediction schema:
  50 foreground predicate scores, top-k predicate IDs/names/scores, GT metadata,
  and `matched_pair_found`.
- `tools/analysis/compute_predicate_recall_from_jsonl.py` consumes this JSONL
  directly for GT-aligned predicate recall.

Command:

```bash
python tools/analysis/export_relation_predictions.py --method FREQ --model FREQ --task PredCLS --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/freq/hidden_export
python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/freq/hidden_export/relation_predictions.jsonl --output_dir outputs/reproduction/freq/predicate_recall --ks 1 5 10
```

Result:

- JSONL validation: PASS.
- Rows: 2 total, 2 matched, 0 unmatched.
- Score dimension: 50.
- Predicate checksum prefix: `8dc025ea105b`.
- Predicate recall: R@1 0.0, R@5 1.0, R@10 1.0.
- Mean predicate recall: mR@1 0.0, mR@5 1.0, mR@10 1.0.

Interpretation:

- Hidden/GT-aligned evaluation is adapted at the schema and execution level.
- The reported numbers are tiny-slice sanity results, not benchmark claims.
