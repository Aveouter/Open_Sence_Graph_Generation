# SHA-GCL Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods shagcl --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `3.7902`.

No training was run.
