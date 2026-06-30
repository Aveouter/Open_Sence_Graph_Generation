# EGTR Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods egtr --skip-train
```

Result:

- PASS.
- EGTR loaded its custom Deformable DETR components.
- The CPU run emitted expected custom-op messages noting that multi-scale deformable attention is not implemented on CPU, but the smoke still completed.
- No model training was run.

Scope:

- Synthetic/no-training forward smoke only.
- This does not establish paper-level benchmark performance.
