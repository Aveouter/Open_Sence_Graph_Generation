# VCTree Code Integration

No VCTree-specific code changes were required.

Shared infrastructure used:

- Lowercase method alias support in `utils/parser.py` and `train.py`.
- Explicit random-init export mode in
  `tools/analysis/export_relation_predictions.py`.

No dataset labels, ground truth, or evaluator semantics were changed.
