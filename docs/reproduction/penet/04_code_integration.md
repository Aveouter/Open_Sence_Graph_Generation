# PENet Code Integration

PENet-specific model code already existed and was reused.

Clean integration regression coverage:

- `tests/reproduction/test_penet_adapter.py`
  - Verifies `penet` is registered in `method_maps`.
  - Verifies `build_penet` consumes the OpenSGG config fields.
  - Verifies the adapter returns the Motifs-compatible output schema.
  - Verifies the one-object/no-pair path returns empty relation tensors.

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

This remains implementation-audit coverage only. It does not establish
official `PrototypeEmbeddingNetwork` parity.
