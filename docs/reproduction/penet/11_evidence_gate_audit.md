# PENet Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`

This audit supersedes the earlier random-init smoke evidence. PENet cannot be
reported as reproduced until official source, checkpoint, config, inference,
and evaluator evidence are aligned.

## Official Source Anchor

- Paper: `Prototype-based Embedding Network for Scene Graph Generation`,
  Zheng et al., CVPR 2023.
- Official repository: `https://github.com/VL-Group/PENET`
- Local official checkout: `/workspace/external/penet_official/PENET`
- Commit inspected: `9c9f50777c66647799cb7a17dfb855aa98aecd1e`
- Official predictor: `PrototypeEmbeddingNetwork` in
  `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`

## Official Protocol Evidence

The official README and scripts define the PredCls command with:

- config `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True`
- `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
- `MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork`
- `DTYPE float32`
- `GLOVE_DIR ./datasets/vg/`
- `MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth`
- `MODEL.WEIGHT ./checkpoints/PE-NET_PredCls/model_final.pth`
- `TEST.ALLOW_LOAD_FROM_CACHE False`

The README provides Google Drive IDs for PredCls, SGCls, and SGDet weights:

- PredCls: `1rjsLs3N33iiOB5xYO7zetNhR7ebi385W`
- SGCls: `1uRl-O-yXmpCs__l_V-WTdYtbWPl57M1B`
- SGDet: `1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C`

The README also states that PredCls and SGCls use a `rel_nms` operation from
RU-Net/HL-Net during evaluation.

## OpenSGG Implementation Gap

OpenSGG has a local PENet adapter in `src/models/penet.py`, but it is not
official-code parity:

- local class: `PENetContext`
- official class: `PrototypeEmbeddingNetwork`
- local config uses `hidden_dim=512`, `penet_embed_dim=200`, `dropout=0.1`
- official source hard-codes `mlp_dim=2048`, `embed_dim=300`, `dropout_p=0.2`
- official relation scoring uses predicate prototypes from `rel_embed`,
  `W_pred`, `project_head`, normalized cosine similarity, and `logit_scale`
- official training adds prototype regularization losses `l21_loss`,
  `dist_loss2`, and `loss_dis`
- local OpenSGG path uses a simpler fused subject/object representation and
  `pred_classifier`; it does not implement the official predicate prototype
  scoring/loss stack
- evaluator parity for official `rel_nms` has not been established

Therefore the local adapter is useful for smoke/audit work only.

## Automated Input Check

Command:

```bash
python tools/reproduction/check_penet_official_inputs.py \
  --output docs/reproduction/penet/penet_official_input_check.json
```

Result: `BLOCKED`

Blockers:

- `missing_penet_vg_inputs`
- `missing_pretrained_detector_checkpoint`
- `missing_official_penet_checkpoints`

Missing official inputs:

- `/workspace/external/penet_official/PENET/datasets/vg/VG-SGG-with-attri.h5`
- `/workspace/external/penet_official/PENET/datasets/vg/VG-SGG-dicts-with-attri.json`
- `/workspace/external/penet_official/PENET/datasets/vg/image_data.json`
- `/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth`
- official PENet PredCls/SGCls/SGDet model files under
  `outputs/pretrained/penet_official`

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| PENET-1 official source pinned | PASS | official repo and commit recorded |
| PENET-2 checkpoint provenance | FAIL | official Google Drive IDs known, no local trusted checkpoint |
| PENET-3 config/inference parity | FAIL | official command recorded, but required detector/data/checkpoint unavailable |
| PENET-4 evaluator parity | FAIL | official `rel_nms` detail identified, OpenSGG parity not verified |
| PENET-5 benchmark-scale metrics | FAIL | no checkpoint-backed official evaluation |

## Current Trusted Claim

PENet has an OpenSGG-compatible smoke adapter and an official-source audit. The
official PENet source is pinned locally, but reproduction is deferred because
required official inputs and evaluator parity evidence are missing.

## Forbidden Claim

Do not claim PENet reproduction, paper-number alignment, or checkpoint-backed
performance from the prior random-init/tiny-slice evidence.
