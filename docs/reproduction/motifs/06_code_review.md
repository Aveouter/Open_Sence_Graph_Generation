# Motifs Code Review

Review status: PASS for no-training checkpoint/test reproduction.

Checks:

- Existing Motifs implementation is reused rather than replaced.
- Output contract matches downstream metric/export consumers:
  `rel_logits`, `pair_indices`, subject/object boxes, object labels,
  `predicate_bg_index`, and `relation_softmax_scope`.
- Checkpoint loading uses the project checkpoint adapter instead of ad hoc key
  mutation in evaluator code.
- JSONL export preserves evaluator/label semantics and only serializes model
  outputs plus GT metadata.
- Shared CLI/default fixes are baseline-test plumbing and do not change dataset
  labels, ground truth, or metric definitions.

Residual risks:

- Local checkpoint is only partially mapped into the OpenSGG Motifs module.
- The tiny CPU test slice verifies plumbing, not benchmark-scale accuracy.
- Full R-101-FPN official architecture parity is not established.
