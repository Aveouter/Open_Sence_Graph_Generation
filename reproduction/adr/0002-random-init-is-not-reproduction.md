# ADR 0002: Random Initialization Is Not Reproduction

## Status

Accepted

## Context

Random-init runs can be useful for checking tensor shapes, dataloaders, and
export schemas. They do not represent a trained baseline.

## Decision

Random-init output must never be used as reproduction evidence or baseline
performance evidence.

## Consequences

- Random-init runs may appear only in smoke/audit/gap reports.
- Reports must explicitly state that random-init metrics are not comparable to
  paper numbers.
- PRs containing only random-init evidence should be titled as audit, smoke, or
  deferred work, not reproduction.
