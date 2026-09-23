## Change Summary

Describe the core code, documentation, configuration, or CI change.

## Motivation

Explain the bug, experiment need, reproduction gap, or maintenance reason.

## Main Changes

- 

## Validation

Commands run and results:

```bash

```

## Reproduction / Benchmark Impact

Does this affect any of the following?

- Dataset:
- Evaluation protocol:
- Metric definition:
- Baseline implementation:
- Result reproducibility:

If yes, explain the reason, before/after behavior, and risk. Do not claim
successful reproduction unless the evidence satisfies `AGENTS.md` and
`reproduction/README.md`.

## Repository Boundary

Is this stable SGG functionality, or research-only experimentation?

- Stable or research-only:
- Supported component that imports or reuses this file:
- Generated reports, figures, or artifacts added to git (or "none"):
- Change to the frozen research surface, or "none":

Research-only work belongs in the research repository. A file that serves one
experiment and is imported by no supported component defaults there. See
`AGENTS.md` and `tools/reproduction/check_repository_boundary.py`.

## Risk And Limitations

List known risks, compatibility concerns, skipped checks, or follow-up work.
