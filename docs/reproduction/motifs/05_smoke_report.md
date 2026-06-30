# Motifs Smoke Report

Accepted no-training smoke command:

```bash
conda run -n hsg python tools/ci_smoke_test.py --methods motifs --skip-train
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss observed in `conda hsg`: `3.6934`.

Additional diagnostic context:

- A stronger `tools/ci_smoke_test.py --methods motifs` run was used during
  diagnosis before the latest no-training constraint. It exposed shared CLI and
  scheduler-default issues that were fixed, but it is not counted as Motifs
  reproduction evidence because the active objective forbids training.

No training was used for the final Motifs closure.
