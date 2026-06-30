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

PR titles and bodies must not imply successful reproduction unless the evidence
meets the full standard above.

Before opening a PR, run:

```bash
python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction <pr-body-or-release-note-files>
```

The script scans text files only. Include changed Markdown/text docs, copied PR
body text, release notes, and any other claim-bearing files. It does not inspect
GitHub metadata automatically. Passing it does not prove reproduction.
