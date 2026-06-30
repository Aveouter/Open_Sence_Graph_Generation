# OpenSGG Baseline Reproduction Tracker

This directory tracks baseline reproduction work for the SGG suite requested in
the Codex goal. The active execution environment is `conda hsg`. The current
objective forbids model training: validation must use checkpoints when available
and otherwise use official-code logic checks plus inference/smoke/test paths.
Each method is closed independently and records source audit, code inventory,
integration work, smoke tests, metric validation, hidden-positive adaptation,
deviations, and final claims.

Stricter reproduction audit note:

- The active standard is now original-paper/original-repository alignment.
- Tiny-slice zero metrics, random initialization, partial checkpoint remapping,
  and minimal OpenSGG-compatible adapters are pipeline evidence, not successful
  reproduction evidence.
- The previous draft PRs have been closed as deferred:
  FREQ #59, Motifs #60, TDE #61, VCTree #62, PENet #63, SHA-GCL #64,
  RA-SGG #65, RelTR #66, and EGTR #67.
- Reopen or replace a PR only after official checkpoint/protocol/evaluator
  compatibility is verified without training.
- Before reopening any deferred baseline, satisfy the global and method-specific
  gates in `docs/reproduction/evidence_gates.md`.

Completion states used here:

- `HIDDEN_EVAL_ADAPTED`: OpenSGG integration plus standard metric path and
  hidden/GT-aligned export path validated.
- `METRIC_VALIDATED`: OpenSGG integration plus standard metric path validated.
- `SMOKE_ONLY_WITH_DEVIATION_REPORT`: smoke passes but metric validation is not
  available or not meaningful in the current environment.
- `FAILED_WITH_REPORT`: failure diagnosed and documented after retries.
- `DEFERRED_NOT_REPRODUCED`: the method is not counted as reproduced under the
  stricter original-alignment requirement.
- `PR_CLOSED_DEFERRED`: the previous draft PR was closed to avoid presenting
  pipeline evidence as reproduction evidence.
- `NOT_STARTED`: scaffold exists but no phase has run.

The tracker preserves evaluator semantics and dataset labels. Resource,
checkpoint, and protocol gaps are recorded as deviations.

Primary control document:

- `evidence_gates.md`: required evidence before a deferred baseline may be
  upgraded from audit/smoke/gap analysis to reproduction.
