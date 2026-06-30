# PENet Code Integration

PENet-specific model code already existed and was reused.

Supporting integration change:

- `tools/analysis/export_relation_predictions.py`
  - Added config filename aliases including `penet -> PE_NET`.
  - This resolved the exporter failure where it tried to load
    `configs/VisualGenome/PENet.py`.

Shared infrastructure used:

- Lowercase method alias support in `utils/parser.py` and `train.py`.
- Explicit random-init export mode in
  `tools/analysis/export_relation_predictions.py`.

No dataset labels, ground truth, or evaluator semantics were changed.
