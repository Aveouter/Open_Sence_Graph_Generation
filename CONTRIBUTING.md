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
  AGENTS.md reproduction tools/reproduction README.md
```

## Where Does This File Belong

`main` holds the stable SGG system. Exploratory analysis, protocol-specific
research runners, and the artifacts they generate do not belong here. Two
questions decide it:

1. Is this stable SGG functionality, or research-only experimentation?
2. Which supported component will import or reuse this file?

If no supported component imports it and it serves a single experiment or PR, it
belongs in the research repository. That covers one-off diagnostics
(`diagnose_*.py`), phase drivers (`run_phase*.py`, `summarise_phase*.py`),
versioned research trees, generated reports and figures, and the decision
records for a research protocol. If a research result later becomes a supported
feature, reintroduce a distilled implementation with tests rather than moving
the experiment harness across.

`tools/reproduction/check_repository_boundary.py` enforces this mechanically. It
is a ratchet: the research surface currently in `main` is frozen by path and
blob hash, and the check fails when that surface changes -- not only when
something new is added, but when a frozen file is edited, staged or not. A
deliberate change is recorded in
`reproduction/migration/relational_emergence_migration.json` and re-frozen with
`--print-baseline`.

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
