# RA-SGG Standard Metric Validation

Setup:

- Environment: `conda hsg`.
- Checkpoint: none.
- Memory bank: none.
- Fallback: explicit random init / no memory.
- Device: CPU.
- Test dataset cap: 2 images.
- Task: PredCls metric path.
- No training.

Result:

- `predcls_R@20`: 0.0
- `predcls_R@50`: 0.0
- `predcls_R@100`: 0.0
- `predcls_mR@20`: 0.0
- `predcls_mR@50`: 0.0
- `predcls_mR@100`: 0.0

Interpretation:

- RA-SGG adapter outputs are accepted by the standard OpenSGG PredCls metric
  path.
- Values are random-init/no-memory tiny-slice sanity numbers only.
