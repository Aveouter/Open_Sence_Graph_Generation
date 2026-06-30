# Motifs Code Inventory

OpenSGG implementation files:

- `configs/VisualGenome/Motifs.py`
  - PredCls-oriented VisualGenome Motifs config.
  - Uses background-inclusive label convention: `entity_nums=151`,
    `rel_nums=51`.
- `src/methods/motifs_method.py`
  - Lightning wrapper and `MotifsCriterion`.
  - Converts OpenSGG batches into per-image object features, boxes, labels, and
    relation targets.
- `src/models/motifs.py`
  - SGB/Kaihua-style Motifs components:
    `PairFrequencyBias`, object/edge context layers, relation compression, and
    TDE-related causal modules.
- `src/methods/__init__.py`
  - Registers method key `motifs`.
- `tools/analysis/export_relation_predictions.py`
  - Motifs-style relation JSONL export and validation.
- `tools/analysis/compute_predicate_recall_from_jsonl.py`
  - GT-aligned predicate recall and mean recall.
- `src/core/metrics.py`
  - Standard PredCls metric path.

Local checkpoint/source artifacts:

- `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth`
- `outputs/pretrained/motifs/coldmanck/motifs_full_predcls_test.log`
- `outputs/pretrained/motifs/coldmanck/output.zip`

Observed checkpoint compatibility:

- `BaseExperiment._adapt_state_dict` remapped 627 external checkpoint keys to
  48 OpenSGG model keys.
- Missing OpenSGG keys: `input_visual_proj.weight`, `input_visual_proj.bias`.
