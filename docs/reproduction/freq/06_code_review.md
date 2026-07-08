# FREQ Code Review

Review status: PASS for minimal reproduction.

Checks:

- The implementation reuses existing prior/evaluator code rather than adding a
  parallel metric path.
- Output keys match Motifs-style downstream consumers.
- `PairFrequencyBias` remains frozen; the dummy scalar exists only for optimizer
  compatibility and does not alter logits.
- No dataset labels, ground truth, or evaluator semantics were changed.
- Exporter no-checkpoint behavior is limited to `method.lower() == "freq"`.

Residual risks:

- The FREQ prior depends on the local `train.json`/`rel.json` schema used by
  `PairFrequencyBias._build_vg_prior`.
- The tiny smoke/metric slices validate plumbing, not paper-scale performance.
