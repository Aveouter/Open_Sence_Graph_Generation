# RelTR Hidden-Positive Evaluation

Export command:

```bash
conda run -n hsg python tools/analysis/export_reltr_predictions.py --ckpt_path outputs/pretrained/reltr/reltr_vg.pth --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --top_k 10 --output_dir outputs/reproduction/reltr/hidden_export_ckpt
```

Result:

- JSONL validation: PASS.
- Rows: 2 total, 2 matched, 0 unmatched.

Predicate recall:

- R@1/R@5/R@10: 0.0 / 0.0 / 0.0.
- mR@1/mR@5/mR@10: 0.0 / 0.0 / 0.0.
