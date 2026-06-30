# PENet Standard Metric Validation

Setup:

- Environment: `conda hsg`.
- Checkpoint: none.
- Fallback: explicit random init.
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

- PENet outputs are accepted by the standard OpenSGG PredCls metric path.
- Values are random-init tiny-slice sanity numbers only.
- The 0.0 result is not working reproduction evidence. It only shows the local
  metric path can consume the adapter output without crashing.
- Official PENet metric validation is blocked by missing PENET-format VG inputs,
  missing pretrained detector checkpoint, missing official PENet checkpoints,
  and unverified `rel_nms` evaluator parity.
