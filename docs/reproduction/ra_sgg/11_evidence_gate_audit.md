# RA-SGG Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`

This audit supersedes the earlier no-memory random-init smoke evidence. RA-SGG
cannot be reported as reproduced until official source, pretrained PE-Net,
ReTAG checkpoint, memory-bank features, config, inference, and evaluator
evidence are aligned.

## Official Source Anchor

- Method name in official repo: ReTAG / RA-SGG
- Official repository: `https://github.com/KanghoonYoon/torch-rasgg`
- Local official checkout: `/workspace/external/ra_sgg_official/torch-rasgg`
- Commit inspected: `e8be01b9fde5c694243606e73931a4c8a8b1bf41`

## Official Protocol Evidence

The official README says ReTAG requires:

- a pretrained PE-Net model
- a memory bank populated with relation embeddings from the training dataset
- official pretrained model folders
- official memory-bank feature folders

The official PredCls script `scripts/predcls_train_retag.sh` uses:

- config `configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml`
- `TYPE "retag"`
- predictor `ReTAGPENet`
- pretrained PE-Net `checkpoints/PE-NET_PredCls/model_final.pth`
- `RASGG.MEMORY_SIZE 8`
- `RASGG.NUM_RETRIEVALS 20`
- `RASGG.THRESHOLD 0.3`
- `RASGG.MIXUP True`
- `RASGG.MIXUP_ALPHA 20`
- `RASGG.MIXUP_BETA 5`
- `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True`
- `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
- `MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False`
- `DTYPE "float32"`

The official code loads feature-bank files matching:

```text
featurebank/<mode>_bg_processed_fb_train_<MEMORY_SIZE>.npy
```

The README also states that PredCls and SGCls use `rel_nms` from RU-Net/HL-Net.

## OpenSGG Implementation Gap

OpenSGG has a local `RASGGModel`, but it is not official ReTAG parity:

- local code wraps the simplified `PENetContext`
- local retrieval is an optional predicate-distribution logit prior
- official ReTAG retrieves top-k relation embeddings from a feature bank
- official code uses retrieved subject/object/predicate values, retrieved
  predicate distributions, frequency reweighting, reliable selection, BG
  correction, mixup, and prototype losses
- local no-memory fallback cannot validate retrieval behavior

Therefore the local adapter is implementation-audit/smoke evidence only.

## Automated Input Check

Command:

```bash
python tools/reproduction/check_rasgg_official_inputs.py \
  --output docs/reproduction/ra_sgg/rasgg_official_input_check.json
```

Result: `BLOCKED`

Blockers:

- `missing_rasgg_vg_inputs`
- `missing_official_retag_checkpoints`
- `missing_pretrained_penet_checkpoints`
- `missing_official_memory_bank_features`

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| RASGG-1 official source pinned | PASS | official repo and commit recorded |
| RASGG-2 checkpoint provenance | FAIL | official model folder known, no trusted local ReTAG checkpoint |
| RASGG-3 memory-bank artifacts | FAIL | no local feature-bank files |
| RASGG-4 augmentation/retrieval parity | FAIL | local adapter lacks official top-k feature-bank retrieval and mixup/reliable-selection behavior |
| RASGG-5 evaluator semantics | FAIL | official `rel_nms`/SGB evaluator path not run or compared |
| RASGG-6 benchmark-scale metrics | FAIL | no checkpoint-backed official evaluation |

## Current Trusted Claim

RA-SGG has an OpenSGG-compatible no-memory smoke adapter and an official-source
audit. Reproduction is deferred because required official inputs, checkpoints,
memory-bank features, and evaluator parity evidence are missing.

## Forbidden Claim

Do not claim RA-SGG/ReTAG reproduction, official retrieval behavior,
paper-number alignment, or checkpoint-backed performance from the local
random-init/no-memory evidence.
