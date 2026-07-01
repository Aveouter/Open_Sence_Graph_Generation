# Contributing

OpenSGG accepts changes that improve code correctness, reproducibility,
documentation, or maintainability.

## Pull Request Rules

- Use a focused branch and keep unrelated changes out of the same PR.
- Do not commit datasets, checkpoints, compiled extensions, experiment outputs,
  local reports, or generated caches.
- Run the lightweight validation checks before opening a PR:

```bash
python tools/reproduction/run_reproduction_guardrails.py
python tools/ci_validate.py
python -m compileall -q tools/reproduction tools/analysis src data utils train.py
```

For reproduction-related changes, also run:

```bash
python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction docs/reproduction tools/reproduction README.md
```

## Reproduction Standard

OpenSGG distinguishes runnable adapters from paper-aligned reproduction.
Do not claim a baseline is reproduced unless the implementation, checkpoint,
configuration, preprocessing, inference path, evaluator, and metric semantics
are aligned with the original paper or official repository.

Use the labels in `reproduction/README.md` and `AGENTS.md` for partial work,
including `pipeline_smoke_only`, `implementation_audit`,
`checkpoint_unavailable`, `protocol_mismatch`, and `deferred_reproduction`.

## Code Style

- Prefer the existing module structure and naming conventions.
- Keep analysis scripts reusable and documented; one-off experiment scripts
  belong outside the main branch unless they are part of a tracked report.
- Add tests or guardrails when changing method registration, metrics,
  checkpoint loading, or reproduction evidence handling.
