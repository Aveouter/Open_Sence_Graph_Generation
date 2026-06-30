# RA-SGG Hidden-Positive Evaluation

Export command:

```bash
conda run -n hsg python tools/analysis/export_relation_predictions.py --method RA_SGG --model RA-SGG --task PredCLS --allow_random_init --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/ra_sgg/hidden_export_random
```

Result:

- Mode: `random_init_fallback`.
- JSONL validation: PASS.
- Rows: 2 total, 2 matched, 0 unmatched.
- Score dimension: 50.

Predicate recall:

- R@1/R@5/R@10: 0.0 / 0.0 / 0.0.
- mR@1/mR@5/mR@10: 0.0 / 0.0 / 0.0.
