# OpenSGG Agent Instructions

These instructions apply to all AI agents working in this repository. Codex is
the primary supported agent. Claude Code may reuse the same workflow through
`reproduction/agents/claude_code.md`, but if the two systems conflict, follow
this file and the files under `reproduction/`.

## Reproduction Standard

The project standard is baseline reproduction with evidence, not pipeline
demonstration.

An agent may call a baseline `reproduced` only when all of the following are
aligned with the original paper or official repository:

- method implementation
- checkpoint identity and provenance
- configuration and preprocessing
- inference flow
- evaluation protocol and metric semantics

OpenSGG implementations with the same method name are candidates only. They are
not automatically official reproductions.

## Forbidden Claims

Do not describe any of the following as reproduction success:

- random initialization runs
- tiny-slice evaluation that only proves the loop executes
- all-zero metrics
- partial checkpoint remapping
- approximate OpenSGG adapters without official-code parity
- changed labels, ground truth, evaluator semantics, or filtered failures
- no-checkpoint results; these must not be presented as baseline results

Use these labels instead:

- `pipeline_smoke_only`
- `implementation_audit`
- `checkpoint_unavailable`
- `protocol_mismatch`
- `deferred_reproduction`
- `not_reproduction_ready`

## Required Workflow Before Running a Baseline

1. Create or update a baseline issue from
   `reproduction/issues/baseline_reproduction_issue.md`.
2. Record decisions in `reproduction/adr/` when evidence is ambiguous or a
   method is deferred.
3. Fill the relevant templates in `reproduction/templates/`.
4. Verify checkpoint/config/evaluator compatibility before running metrics.
5. If alignment is missing, stop at audit/gap analysis. Do not manufacture a
   reproduction result from random init or a smoke run.

## PR Rules

A PR is appropriate only when it does one of the following:

- fixes code to match the paper or official repository
- adds checkpoint-backed evaluation with recorded provenance
- documents protocol alignment or a precise mismatch
- adds an audit/deferred status report with explicit evidence
- adds guardrails that prevent random-init/tiny-slice outputs from being
  mistaken for reproduction evidence
- moves code, reports, or decision records across the stable/research boundary
  and records the move under `reproduction/migration/`

PR titles and bodies must not imply successful reproduction unless the evidence
meets the full standard above.

Before opening a PR, run:

```bash
python tools/reproduction/run_reproduction_guardrails.py

python tools/reproduction/check_reproduction_docs.py

python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction <pr-body-or-release-note-files>
```

The documentation check verifies that the target baseline reports are present
and tracked. The claim script scans text files only. Include changed
Markdown/text docs, copied PR body text, release notes, and any other
claim-bearing files. It does not inspect GitHub metadata automatically. Passing
these checks does not prove reproduction. The aggregate guardrail script runs
the lightweight syntax, documentation, and claim-boundary checks without running
model evaluation or rewriting evidence JSON files.

## Repository Boundary

`main` holds the stable SGG system: training, evaluation, registered methods,
and the reproduction contracts under `reproduction/`. Exploratory analysis,
one-off diagnostics, protocol-specific research runners, generated reports, and
experimental artifacts belong in the research repository, not here.

The practical rule:

> If a file serves only one experiment or PR and no supported main component
> imports it, default it to the research repository.

Before adding a file to `main`, answer:

- Is this stable SGG functionality, or research-only experimentation?
- Which supported component will import or reuse this file?
- Does this change add generated reports, figures, or artifacts to git?

If a research result later becomes a supported SGG feature, reintroduce a
distilled implementation with its own tests, not the experiment harness.

`tools/reproduction/check_repository_boundary.py` enforces this and runs both in
`run_reproduction_guardrails.py` and in the CI `validate` job. It is a ratchet
rather than a clean-slate rule: the research surface currently present in `main`
is frozen by path and blob SHA-256, and the check fails when that surface
changes -- a new research path, an edit to a frozen file, or a frozen entry that
has gone stale. A deliberate change is recorded in
`reproduction/migration/relational_emergence_migration.json` and re-frozen with
`--print-baseline`; nothing else authorizes an edit to the frozen surface.
