# RelTR Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods reltr --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic loss observed: `1587.5037`.

No training was run.
