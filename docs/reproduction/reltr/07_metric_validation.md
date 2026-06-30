# RelTR Standard Metric Validation

Command:

```bash
conda run -n hsg python tools/analysis/official_metric_eval.py --ckpt_path outputs/pretrained/reltr/reltr_vg.pth --method_name RelTR --output_dir outputs/reproduction/reltr/official_metric_ckpt --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --device cpu --max_batches 1
```

Result:

- Batches processed: 1.
- `predcls_R@10`: 0.0
- `predcls_R@20`: 0.0
- `predcls_R@50`: 0.0
- `predcls_R@100`: 0.0
- `predcls_mR@10`: 0.0
- `predcls_mR@20`: 0.0
- `predcls_mR@50`: 0.0
- `predcls_mR@100`: 0.0

Interpretation:

- RelTR checkpoint outputs are accepted by the standard OpenSGG PredCls metric
  path.
