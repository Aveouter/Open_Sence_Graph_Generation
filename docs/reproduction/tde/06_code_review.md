# TDE Code Review

Review status: PASS for random-init no-training fallback reproduction.

Checks:

- `TDE_Method` reuses Motifs data/feature handling and only swaps in
  `build_tde`.
- `TDEModel` exposes official causal controls and branch structure:
  visual, context, frequency, `calculate_logits`, untreated buffers, and
  effect-type switch.
- `--allow_random_init` is explicit and records fallback mode; it does not
  silently hide checkpoint absence.
- Export and metric paths reuse existing evaluator/schema code.
- No evaluator semantics, labels, or ground truth were changed.

Residual risks:

- No verified official TDE checkpoint is available.
- Random-init numbers do not represent TDE paper behavior.
- Official evaluator equivalence remains unverified.
