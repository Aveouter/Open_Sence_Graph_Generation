# EGTR Code Integration

Integration status: existing OpenSGG implementation reused.

Code touched for EGTR in this goal:

- No EGTR model or config code was changed.
- Documentation was added under `docs/reproduction/egtr/`.

Relevant existing paths:

- `src/methods/egtr_method.py`
- `src/modules/egtr/`
- `configs/VisualGenome/EGTR.py`
- `tools/analysis/export_egtr_predictions.py`

Integration claim:

- EGTR can be instantiated and used for no-training smoke and checkpoint-backed relation export under `conda hsg`.
