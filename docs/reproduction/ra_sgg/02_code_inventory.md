# RA-SGG Code Inventory

Official audit anchors:

- `configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml`
  - `RASGG.NUM_RETRIEVALS: 5`
  - relation predictor: `RA-PENetCorrectProtoBetaEnhanceBG`
- `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
  - `ReTAGPENet`
  - RA-PENet predictor variants
  - retrieval top-k, retrieved predicate distribution, mixup/prototype losses.
- `README.md`
  - pretrained PE-Net and memory-bank requirement.

OpenSGG files added:

- `src/models/ra_sgg.py`
  - Minimal `RASGGModel` adapter.
  - Reuses `PENetContext` and optional memory-distribution logit fusion.
- `src/methods/ra_sgg_method.py`
  - Motifs-compatible Lightning method wrapper.
- `configs/VisualGenome/RA_SGG.py`
  - VisualGenome config with memory-bank and retrieval fields.

Registry/config updates:

- `src/methods/__init__.py`
- `src/models/__init__.py`
- `utils/parser.py`

No evaluator, export schema, label, or ground-truth semantics were changed by
the clean RA-SGG integration PR.

Checkpoint/memory inventory:

- No local RA-SGG checkpoint found.
- No local RA-SGG memory bank found.
