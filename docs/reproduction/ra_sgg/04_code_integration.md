# RA-SGG Code Integration

Files added:

- `src/models/ra_sgg.py`
- `src/methods/ra_sgg_method.py`
- `configs/VisualGenome/RA_SGG.py`

Files updated:

- `src/methods/__init__.py`
- `src/models/__init__.py`
- `utils/parser.py`
- `train.py`
- `tools/ci_smoke_test.py`
- `tools/analysis/export_relation_predictions.py`

Integration notes:

- The adapter uses `PENetContext` as the base relation predictor.
- Optional memory distribution fusion is present but inactive unless
  `ra_sgg_memory_bank_path` and a positive `ra_sgg_retrieval_logit_coef` are
  supplied.
- No evaluator semantics, labels, or ground truth were changed.
