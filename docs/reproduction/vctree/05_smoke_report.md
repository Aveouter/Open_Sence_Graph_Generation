# VCTree Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods vctree --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `4.9152`.

No training was run.
