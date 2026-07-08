# Reproduction Workflow

This directory defines the research-grade baseline reproduction workflow for
OpenSGG. It exists to make AI agents obey scientific standards instead of
optimizing for a runnable demo.

## Goal

Produce baseline reproduction with evidence.

The highest-priority sources are:

1. official paper
2. official repository
3. official or provenance-verified checkpoint
4. paper-compatible configuration and preprocessing
5. paper-compatible inference and evaluator semantics

OpenSGG code with a matching method name is only a candidate implementation.

## Outcomes

Use one of these outcomes for every baseline:

| Outcome | Meaning |
|---|---|
| `reproduction_ready` | All required alignment evidence is present before full evaluation. |
| `checkpoint_backed_evaluation` | A verified checkpoint was evaluated under an aligned protocol. |
| `implementation_audit` | Code was compared to official sources, but reproduction evidence is incomplete. |
| `pipeline_smoke_only` | Import/forward/eval loop runs, but method reproduction is not established. |
| `checkpoint_unavailable` | No trustworthy checkpoint is available. |
| `protocol_mismatch` | Evaluator/config/inference semantics do not match the official protocol. |
| `deferred_reproduction` | Work stops until missing alignment evidence is available. |

## Directory Map

- `glossary.md`: definitions and claim boundaries.
- `USAGE.md`: step-by-step instructions for humans and agents.
- `adr/`: decision records for non-obvious choices.
- `issues/`: issue templates for baseline work.
- `templates/`: reusable evidence and report templates.
- `agents/`: agent-specific operating notes.
- `baselines/`: per-baseline status files created from templates.

## Documentation Boundary

The tracked reproduction source of truth is this `reproduction/` directory.
Historical notes that may exist under local `docs/` directories are evidence
archives only and must not override the workflow, templates, ADRs, or baseline
status files here.

## Minimal Baseline Flow

1. Open an issue using `issues/baseline_reproduction_issue.md`.
2. Fill source, checkpoint, config, inference, and evaluator evidence.
3. Decide whether the method is reproduction-ready.
4. If not ready, write a gap analysis and mark it deferred.
5. If ready, run checkpoint-backed evaluation under the aligned protocol.
6. Open a PR only for code fixes, protocol documentation, checkpoint-backed
   evaluation, or deferred audit reports.

## Non-Goals

This workflow does not reward making a command run at any cost. It explicitly
rejects random-init or all-zero tiny-slice outputs as reproduction evidence.

For a complete operator guide, read `USAGE.md` before editing code or running
baseline commands.
