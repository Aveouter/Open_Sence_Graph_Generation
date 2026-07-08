# FREQ Code Integration

Files changed:

- `src/methods/freq_method.py`
  - Adds `FREQModel` and `FREQ_Method`.
  - Uses frozen `PairFrequencyBias`.
  - Adds a scalar dummy parameter so optimizer construction remains valid.
- `configs/VisualGenome/FREQ.py`
  - Adds PredCls-oriented VisualGenome config.
- `src/methods/__init__.py`
  - Registers `freq`.
- `utils/parser.py`
  - Adds `FREQ` and `freq` as accepted CLI method choices.
- `train.py`
  - Adds lowercase `freq` to config filename aliases.
- `tools/ci_smoke_test.py`
  - Adds `--no_display_method_info` to CPU train smoke.
  - Hides CUDA from the CPU train subprocess to avoid broken-driver CUDA RNG
    initialization in Lightning.
- `utils/main_utils.py`
  - Makes CUDA device-name collection robust when CUDA is visible but the driver
    cannot initialize.
- `tools/analysis/export_relation_predictions.py`
  - Allows `--method FREQ` without a checkpoint and records mode
    `prior_no_checkpoint`.

Evaluator semantics, labels, and ground truth were not changed.
