# OpenSGG Baseline Reproduction Tracker

This directory tracks baseline reproduction work for the SGG suite requested in
the Codex goal. The active execution environment is `conda hsg`. The current
objective forbids model training: validation must use checkpoints when available
and otherwise use official-code logic checks plus inference/smoke/test paths.
Each method is closed independently and records source audit, code inventory,
integration work, smoke tests, metric validation, hidden-positive adaptation,
deviations, and final claims.

PR submission note:

- The active objective asks for an independent GitHub PR after each method.
- Draft PRs have been submitted for all tracked methods, targeting
  `feat/analysis-tools`:
  FREQ #59, Motifs #60, TDE #61, VCTree #62, PENet #63, SHA-GCL #64,
  RA-SGG #65, RelTR #66, and EGTR #67.

Completion states used here:

- `HIDDEN_EVAL_ADAPTED`: OpenSGG integration plus standard metric path and
  hidden/GT-aligned export path validated.
- `METRIC_VALIDATED`: OpenSGG integration plus standard metric path validated.
- `SMOKE_ONLY_WITH_DEVIATION_REPORT`: smoke passes but metric validation is not
  available or not meaningful in the current environment.
- `FAILED_WITH_REPORT`: failure diagnosed and documented after retries.
- `PR_SUBMITTED_DRAFT`: an independent draft GitHub PR was created for the
  method.
- `NOT_STARTED`: scaffold exists but no phase has run.

The tracker preserves evaluator semantics and dataset labels. Resource,
checkpoint, and protocol gaps are recorded as deviations.
