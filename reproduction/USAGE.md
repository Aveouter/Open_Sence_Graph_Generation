# Reproduction Workflow Usage Guide

This guide explains how to use the OpenSGG reproduction workflow. It is written
for both human maintainers and AI agents.

## What This Part Is For

Use this workflow when a task claims to reproduce, audit, evaluate, or defer an
SGG baseline. The purpose is to keep evidence and claims aligned:

- official paper/repository first
- checkpoint provenance before metrics
- configuration, inference, and evaluator semantics recorded before results
- smoke/audit/gap work clearly separated from reproduction claims

Do not use this workflow to make an approximate implementation look complete.

## Required Reading

Before touching code for a baseline:

1. Read `AGENTS.md`.
2. Read `reproduction/README.md`.
3. Read `reproduction/glossary.md`.
4. Read `reproduction/issues/baseline_reproduction_issue.md`.
5. Read `reproduction/templates/pr_checklist.md`.

Codex-specific notes are in `reproduction/agents/codex.md`. Claude Code notes
are in `reproduction/agents/claude_code.md`, but Codex rules are authoritative
when the two differ.

## Baseline Workflow

For each baseline, create an auditable trail in this order:

1. Create an issue from `reproduction/issues/baseline_reproduction_issue.md`.
2. Record official paper and official repository links.
3. Record checkpoint source, checksum if available, and license/access notes.
4. Compare OpenSGG code against the official implementation.
5. Compare config, preprocessing, inference flow, and evaluator semantics.
6. Choose the correct outcome label from `reproduction/README.md`.
7. Run metrics only after the required alignment gates pass.
8. Write deviations and blockers before opening a PR.

If any required evidence is missing, stop at `implementation_audit`,
`checkpoint_unavailable`, `protocol_mismatch`, or `deferred_reproduction`.

## Evidence Gates

Use these gates before calling a result reproduction evidence:

| Gate | Required evidence |
|---|---|
| Method | local implementation matches the paper or official repo, or deviations are documented |
| Checkpoint | checkpoint identity, source, and compatibility are verified |
| Config | dataset, splits, preprocessing, and hyperparameters match the official protocol |
| Inference | prediction path and postprocessing match the official protocol |
| Evaluation | metric names, modes, and evaluator semantics match the official protocol |

Passing a smoke test does not pass these gates.

## What To Do When Evidence Is Missing

Use the precise label rather than stretching the claim:

- missing official checkpoint: `checkpoint_unavailable`
- official evaluator differs from OpenSGG: `protocol_mismatch`
- local code is only approximate: `implementation_audit`
- only import/forward/eval loop ran: `pipeline_smoke_only`
- work should wait for official inputs: `deferred_reproduction`

Write the missing evidence in the issue and in the relevant report. Do not
replace missing evidence with random initialization, tiny slices, partial
checkpoint remaps, or all-zero metrics.

## Guardrail Check

Before opening or updating a PR, run:

```bash
python tools/reproduction/run_reproduction_guardrails.py
```

This aggregate guardrail compiles reproduction tools, checks tracked report
completeness, runs reproduction guardrail unit tests, and scans claim-bearing
text. It does not run model evaluation or rewrite evidence JSON files.

Individual claim guardrail:

```bash
python tools/reproduction/check_reproduction_claims.py \
  --changed-from origin/main \
  AGENTS.md reproduction <pr-body-or-release-note-files>
```

Also pass any Markdown/text files outside `reproduction/` that contain claims,
such as report drafts or copied PR body text. The check scans text only. Passing
it does not prove reproduction; it only catches common misleading phrasing.

## PR Checklist

Use `reproduction/templates/pr_checklist.md` in every reproduction-related PR.
A valid PR should be one of these:

- code alignment fix
- checkpoint-backed evaluation
- protocol alignment documentation
- implementation audit
- deferred reproduction report
- guardrail against misleading claims

The PR title and body must match the evidence. For example:

- Good: `audit(tde): record checkpoint-unavailable reproduction blocker`
- Good: `fix(freq): align prior computation with official statistics file`
- Not allowed: `reproduce tde baseline` when no verified checkpoint exists

## Agent Operating Rules

Agents must not start by trying to make a command pass. Start with source,
checkpoint, config, inference, and evaluation alignment. If alignment cannot be
verified, produce an audit or deferred report and stop.

Agents must not:

- modify labels or ground truth
- change evaluator semantics to improve numbers
- delete failed samples
- hide zero or failed metrics
- present random-init or tiny-slice output as reproduction
- call an OpenSGG adapter official without comparing it to the official repo

## Typical Commands

Claim guardrail:

```bash
python tools/reproduction/check_reproduction_claims.py \
  AGENTS.md reproduction
```

Documentation completeness guardrail for the target baseline suite:

```bash
python tools/reproduction/check_reproduction_docs.py
```

This check verifies that FREQ, TDE, VCTree, PENet, SHA-GCL, and RA-SGG have
tracked phase reports, evidence-gate audits, official-input check JSON files,
baseline matrix rows, method gate sections, the suite evidence runner, and the
tracked suite summary. It does not prove reproduction.

Python syntax check for reproduction tools:

```bash
python -m py_compile tools/reproduction/*.py
```

FREQ/TDE/VCTree input audits, when those files are present:

```bash
python tools/reproduction/check_sgb_freq_inputs.py \
  --output docs/reproduction/freq/sgb_freq_input_check.json

python tools/reproduction/check_tde_official_inputs.py \
  --output docs/reproduction/tde/tde_official_input_check.json

python tools/reproduction/check_vctree_official_inputs.py \
  --output docs/reproduction/vctree/vctree_official_input_check.json
```

Full target-suite evidence gate audit:

```bash
python tools/reproduction/run_evidence_gate_checks.py
```

Exit code `2` means at least one baseline is blocked by missing official inputs
or checkpoints. Treat that as a deferred reproduction state, not as a successful
metric result.

The audit commands report blockers. A blocker is not a failure to hide; it is
the evidence needed to keep the reproduction claim honest.
