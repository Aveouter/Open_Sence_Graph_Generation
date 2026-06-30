# EGTR Code Inventory

Existing implementation:

- `src/methods/egtr_method.py` wraps EGTR as an OpenSGG method.
- `src/modules/egtr/egtr.py` provides the scene graph generation model and relation heads.
- `src/modules/egtr/deformable_detr.py` provides the Deformable DETR backbone/configuration pieces.
- `configs/VisualGenome/EGTR.py` stores the Visual Genome config.

Evaluation/export utilities:

- `tools/ci_smoke_test.py --methods egtr --skip-train` exercises the method without training.
- `tools/analysis/export_egtr_predictions.py` exports checkpoint-backed GT-aligned relation JSONL.
- `tools/analysis/compute_predicate_recall_from_jsonl.py` computes hidden-positive predicate recall from the exported rows.
- `src/core/metrics.py` contains EGTR-specific adapters, but the reliable checkpoint-backed path for this goal is the SGDet query export plus post-hoc GT alignment.

Data/evaluator safety:

- No data labels were edited.
- No ground truth was edited.
- No evaluator semantics were changed for EGTR.
