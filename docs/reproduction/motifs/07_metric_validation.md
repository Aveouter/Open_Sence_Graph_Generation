# Motifs Standard Metric Validation

Command summary:

```bash
conda run -n hsg bash -lc 'python - <<PY
# Build Motifs in OpenSGG, load local checkpoint, run one test batch,
# call src.core.metrics.metric for PredCls R@K and mR@K.
PY'
```

Concrete setup:

- Checkpoint: `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth`
- Device: CPU
- Test dataset cap: 2 images
- Batch cap: first test batch
- Task: PredCls metric path

Checkpoint mapping:

- Remapped external checkpoint keys: 627 -> 48.
- Missing keys: `input_visual_proj.weight`, `input_visual_proj.bias`.

Result on the tiny slice:

- `predcls_R@20`: 0.0
- `predcls_R@50`: 0.0
- `predcls_R@100`: 0.0
- `predcls_mR@20`: 0.0
- `predcls_mR@50`: 0.0
- `predcls_mR@100`: 0.0

Interpretation:

- This validates that checkpoint-backed Motifs outputs are accepted by the
  standard OpenSGG PredCls metric path.
- It does not establish paper-scale Motifs accuracy.
