# TDE Hidden-Positive Evaluation

Export command:

```bash
conda run -n hsg python tools/analysis/export_relation_predictions.py --method TDE --model TDE --task PredCLS --allow_random_init --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/tde/hidden_export_random
```

Export result:

- Mode: `random_init_fallback`.
- JSONL validation: PASS.
- Total GT relation rows: 2.
- Matched rows: 2.
- Unmatched rows: 0.
- Score dimension: 50.
- Predicate checksum prefix: `8dc025ea105b`.

Predicate recall command:

```bash
conda run -n hsg python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/tde/hidden_export_random/relation_predictions.jsonl --output_dir outputs/reproduction/tde/predicate_recall_random --ks 1 5 10
```

Predicate recall result:

- R@1: 0.0
- R@5: 0.0
- R@10: 0.0
- mR@1: 0.0
- mR@5: 0.0
- mR@10: 0.0

Interpretation:

- Hidden/GT-aligned evaluation schema is adapted for TDE outputs.
- Values are random-init fallback sanity numbers only.
