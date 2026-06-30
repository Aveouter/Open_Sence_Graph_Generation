# EGTR Official Sources

Primary source:

- `EGTR: Extracting Graph from Transformer for Scene Graph Generation`, CVPR 2024.

OpenSGG-relevant local sources:

- `src/methods/egtr_method.py`
- `src/modules/egtr/egtr.py`
- `src/modules/egtr/deformable_detr.py`
- `configs/VisualGenome/EGTR.py`
- `tools/analysis/export_egtr_predictions.py`

Checkpoint source used in this reproduction:

- Local archive `outputs/pretrained/egtr/egtr_vg.tar.gz`.
- Extracted Lightning checkpoint under `outputs/pretrained/egtr/.../checkpoints/epoch=03-validation_loss=1.71.ckpt`.

Source audit conclusion:

- EGTR is already integrated as a transformer/DETR-family SGG method in OpenSGG.
- The available local checkpoint is usable for inference/export.
- The evaluation protocol is not Motifs-style PredCls. EGTR runs query-based SGDet inference, then OpenSGG's helper can post-hoc align predictions to GT boxes/relations for compact JSONL hidden evaluation.
