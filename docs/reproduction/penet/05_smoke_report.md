# PENet Smoke Report

Command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods penet --skip-train
```

Previous result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `4.4942`.

No training was run.

Clean integration branch rerun:

```bash
python tools/ci_smoke_test.py --methods penet --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss was finite. The exact value is stochastic because the
  smoke uses random synthetic tensors.

The synthetic loss is stochastic and only demonstrates adapter plumbing. It is
not benchmark or reproduction evidence.
