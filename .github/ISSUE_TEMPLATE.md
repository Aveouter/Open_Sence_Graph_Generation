## Issue Type

Choose the closest template from the issue template picker when possible:

- Bug report: runtime, correctness, or CI failures.
- Documentation: unclear setup, usage, or reproduction notes.
- Baseline reproduction audit: evidence tracking before any reproduction claim.

## Summary

Describe the request or problem.

## Context

Include the relevant commit, command, dataset/config, checkpoint, or document.

## Expected Outcome

State what should change and what would count as resolved.

## Mathematical Expressions

Use `$...$` for inline math. For display math, use a fenced `math` block:

````markdown
```math
P(R \mid C_s, C_o)
```
````

Standalone `[` and `]` lines do not delimit GitHub math. If an issue body is
generated through JSON, preserve LaTeX backslashes and check the rendered issue
preview before submitting it.

## Reproduction Claim Boundary

If this issue involves a baseline, checkpoint, metric, or paper result, use the
project status labels from `AGENTS.md` and `reproduction/README.md`. Do not
describe random-init, tiny-slice, all-zero, or partial-checkpoint outputs as a
successful reproduction.
