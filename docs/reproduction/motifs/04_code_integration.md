# Motifs Code Integration

Motifs-specific model code already existed in OpenSGG and was reused.

Shared plumbing fixes made while validating baseline tests:

- `utils/parser.py`
  - Accepted lowercase method keys such as `motifs`, `vctree`, `tde`,
    `penet`, and `shagcl`, matching `method_maps` and the smoke harness.
- `train.py`
  - Added config filename aliases for lowercase method keys.
  - Fills missing default parser values after `--overwrite` config merge.
    This prevents evaluation/test startup failures when a method config does
    not explicitly define scheduler fields.
- `tools/ci_smoke_test.py`
  - Hides CUDA only for CPU smoke subprocesses and passes
    `--no_display_method_info` to avoid unrelated CUDA-driver display failures.
- `utils/main_utils.py`
  - Records CUDA device-query errors instead of crashing environment collection.

Motifs model/evaluator semantics, labels, and ground truth were not changed.
