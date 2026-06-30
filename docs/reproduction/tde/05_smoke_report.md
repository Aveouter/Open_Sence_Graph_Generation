# TDE Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods tde --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `3.9619`.

Interpretation:

- The OpenSGG TDE method can be imported, instantiated, and called under
  `conda hsg`.
- This is not checkpoint or paper-number evidence.
- No training was run.
