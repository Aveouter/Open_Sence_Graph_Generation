# PENet Official Sources

Primary source:

- `Prototype-based Embedding Network for Scene Graph Generation`, Zheng et al.,
  CVPR 2023.
- Official repository: `https://github.com/VL-Group/PENET`
- Local checkout: `/workspace/external/penet_official/PENET`
- Commit inspected: `9c9f50777c66647799cb7a17dfb855aa98aecd1e`

Implementation interpretation:

- Official predictor class: `PrototypeEmbeddingNetwork`.
- Official code path:
  `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`.
- Official PredCls test command uses
  `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`,
  `USE_GT_BOX True`, `USE_GT_OBJECT_LABEL True`,
  `PREDICTOR PrototypeEmbeddingNetwork`, `DTYPE float32`, and
  `TEST.ALLOW_LOAD_FROM_CACHE False`.
- README notes that PredCls and SGCls evaluation use `rel_nms` from
  RU-Net/HL-Net.

Checkpoint status:

- Official README lists Google Drive model IDs:
  - PredCls `1rjsLs3N33iiOB5xYO7zetNhR7ebi385W`
  - SGCls `1uRl-O-yXmpCs__l_V-WTdYtbWPl57M1B`
  - SGDet `1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C`
- No trusted local PENet checkpoint was found under
  `outputs/pretrained/penet_official`.
- The official pretrained detector checkpoint is also absent locally.
- No paper-number alignment is claimed.
