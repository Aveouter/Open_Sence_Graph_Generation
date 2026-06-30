# ADR 0001: Reproduction Claim Standard

## Status

Accepted

## Context

AI agents can easily make a baseline run without proving that the run matches
the original method. This creates misleading reports, especially when random
initialization, tiny slices, or all-zero metrics are treated as success.

## Decision

A baseline may be called reproduced only when method implementation,
checkpoint, configuration, inference flow, and evaluator semantics align with
the original paper or official repository.

## Consequences

- Random-init and tiny-slice runs are labeled `pipeline_smoke_only`.
- Missing or untrusted checkpoints produce `checkpoint_unavailable` or
  `deferred_reproduction`.
- PRs must avoid success language unless alignment evidence is present.
