# EGTR Deviations

Deviations from paper-level reproduction:

- Tiny two-image CPU slice instead of full Visual Genome evaluation.
- No multi-seed full benchmark run.
- No model training.
- Hidden-positive evaluation uses OpenSGG's GT-aligned JSONL adapter, not the paper's complete evaluation protocol.
- Direct Motifs-style PredCls metric validation is not claimed because EGTR is query/SGDet based and requires EGTR target formatting for loss/matcher paths.
- Independent draft GitHub PR submitted: #67.
  - Branch: `features-codex/repro-egtr`.
  - Base: `feat/analysis-tools`.

Non-deviations:

- No data labels changed.
- No ground truth changed.
- No failed samples deleted.
- No evaluator semantics changed to inflate numbers.
