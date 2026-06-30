# RA-SGG Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods ra_sgg --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `4.7866`.

No training was run.
