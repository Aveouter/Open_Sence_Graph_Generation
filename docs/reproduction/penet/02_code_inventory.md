# PENet Code Inventory

OpenSGG files:

- `configs/VisualGenome/PE_NET.py`
  - PENet VisualGenome config.
- `src/methods/penet_method.py`
  - Extends `Motifs_Method` and builds `build_penet`.
- `src/models/penet.py`
  - Prototype/semantic embedding components, visual-semantic gates, predicate
    classifier, and frequency bias.
- `src/methods/__init__.py`
  - Registers method key `penet`.
- `tools/analysis/export_relation_predictions.py`
  - Used with explicit `--allow_random_init` fallback.

Checkpoint inventory:

- Search of `outputs/pretrained/penet_official` found no PredCls, SGCls, or
  SGDet official checkpoint.

Official-code delta:

- Official `PrototypeEmbeddingNetwork` uses `mlp_dim=2048`, `embed_dim=300`,
  `dropout_p=0.2`, `W_pred`, `project_head`, normalized cosine similarity, and
  `logit_scale`.
- Official training adds prototype losses `l21_loss`, `dist_loss2`, and
  `loss_dis`.
- OpenSGG `PENetContext` uses `hidden_dim=512`, `penet_embed_dim=200`, a
  simpler subject/object fusion, and a direct `pred_classifier`.
- Evaluator parity for official `rel_nms` is not verified.
