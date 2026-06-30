# EGTR Code Review

Review stance:

- The existing EGTR code path is transformer/query based and should not be forced into the Motifs-family pair-index API without a protocol change.
- `tools/analysis/export_egtr_predictions.py` correctly labels its output as GT-aligned relation export rather than PredCls.
- `_egtr_to_compact()` is the key adapter for converting dense query outputs to compact relation rows.

Findings:

- No data-label, ground-truth, or evaluator-semantics modifications were made.
- Direct standard PredCls probing can fail if raw OpenSGG targets are passed through EGTR's loss-building path without conversion to EGTR target fields such as `class_labels`.
- The checkpoint-backed GT-aligned export path avoids that mismatch by using EGTR inference and compact conversion explicitly.

Residual risk:

- The hidden-positive numbers are from a two-image CPU slice and should be treated as pipeline validation only.
