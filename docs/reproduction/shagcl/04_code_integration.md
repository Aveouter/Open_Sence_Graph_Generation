# SHA-GCL Code Integration

SHA-GCL-specific model code already existed and was reused.

Shared infrastructure used:

- Lowercase method alias support in `utils/parser.py` and `train.py`.
- Exporter config alias `shagcl -> SHA_GCL`.
- Explicit random-init export mode in
  `tools/analysis/export_relation_predictions.py`.

No dataset labels, ground truth, or evaluator semantics were changed.
