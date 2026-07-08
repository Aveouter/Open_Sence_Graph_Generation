# RA-SGG Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods ra_sgg --skip-train
```

Previous result from the earlier draft branch:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `4.7866`.

No training was run.

Clean integration branch command:

```bash
python tools/ci_smoke_test.py --methods ra_sgg --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss was finite. Local reruns observed `4.1928` and
  `4.3971`; this value is stochastic because the smoke uses random synthetic
  tensors.

Passing the smoke only demonstrates adapter plumbing; it is not benchmark
reproduction.
