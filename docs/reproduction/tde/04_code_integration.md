# TDE Code Integration

TDE-specific model code already existed in OpenSGG and was reused.

New/supporting integration change:

- `tools/analysis/export_relation_predictions.py`
  - Added `--allow_random_init`.
  - Allows no-checkpoint inference export only when the flag is explicitly set.
  - Marks export summary mode as `random_init_fallback`.

Shared test-plumbing changes also support TDE:

- `utils/parser.py` accepts lowercase `tde`.
- `train.py` maps lowercase `tde` to `configs/VisualGenome/TDE.py`.
- `train.py` fills missing default parser values after `--overwrite`.

No dataset labels, ground truth, or evaluator semantics were changed.
