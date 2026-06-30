# TDE Standard Metric Validation

Command summary:

```bash
conda run -n hsg bash -lc 'python - <<PY
# Build TDE random-init fallback, run one test batch,
# call src.core.metrics.metric for PredCls R@K and mR@K.
PY'
```

Concrete setup:

- Checkpoint: none.
- Fallback: random init, explicitly recorded.
- Device: CPU.
- Test dataset cap: 2 images.
- Batch cap: first test batch.
- Task: PredCls metric path.
- No training.

Result on the tiny slice:

- `predcls_R@20`: 0.0
- `predcls_R@50`: 0.0
- `predcls_R@100`: 0.0
- `predcls_mR@20`: 0.0
- `predcls_mR@50`: 0.0
- `predcls_mR@100`: 0.0

Interpretation:

- This validates that OpenSGG TDE outputs are accepted by the standard PredCls
  metric path.
- It does not validate official checkpoint performance or paper alignment.
