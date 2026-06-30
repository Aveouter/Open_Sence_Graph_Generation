# SHA-GCL Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`

This audit supersedes the earlier random-init smoke evidence. SHA-GCL cannot be
reported as reproduced until official source, checkpoint, config, inference,
and evaluator evidence are aligned.

## Official Source Anchor

- Paper: `Stacked Hybrid-Attention and Group Collaborative Learning for
  Unbiased Scene Graph Generation`, Dong et al., CVPR 2022.
- Official repository: `https://github.com/dongxingning/SHA-GCL-for-SGG`
- Local official checkout:
  `/workspace/external/shagcl_official/SHA-GCL-for-SGG`
- Commit inspected: `8acfb818a0b2a88f9dd4a8ed64591ef856e66bf5`

## Official Protocol Evidence

The official README/config define the VG PredCls path with:

- config `configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml`
- `GLOBAL_SETTING.DATASET_CHOICE 'VG'`
- `GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL'`
- `GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention'`
- `GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4'`
- `GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown'`
- `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True`
- `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
- `DTYPE "float16"`

The README lists a public OneDrive link for `SHA_GCL_VG_PredCls`. It says other
trained models from the paper require contacting the authors.

## OpenSGG Implementation Gap

OpenSGG has a local `SHAGCLModel`, but it is not proven official-code parity:

- official predictor: `TransLike_GCL`
- official GCL uses grouped auxiliary classifiers, `FrequencyBias_GCL`, fixed
  predicate group splits, and KL-logit knowledge transfer modes
- official defaults include `divide4`, `KL_logit_TopDown`,
  `NO_RELATION_RESTRAIN`, `ZERO_LABEL_PADDING_MODE`, and
  `NO_RELATION_PENALTY`
- local OpenSGG implementation uses a simplified pair-level co-attention stack,
  learned group prototypes, and an MSE-style `gcl_loss`

Therefore the local adapter is implementation-audit/smoke evidence only.

## Automated Input Check

Command:

```bash
python tools/reproduction/check_shagcl_official_inputs.py \
  --output docs/reproduction/shagcl/shagcl_official_input_check.json
```

Result: `BLOCKED`

Blockers:

- `missing_shagcl_vg_inputs`
- `missing_pretrained_detector_checkpoint`
- `missing_official_shagcl_checkpoint`

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| SHAGCL-1 official source pinned | PASS | official repo and commit recorded |
| SHAGCL-2 checkpoint provenance | FAIL | README link known, no trusted local checkpoint |
| SHAGCL-3 semantic hierarchy/GCL parity | FAIL | local implementation differs from official `TransLike_GCL` GCL stack |
| SHAGCL-4 evaluator semantics | FAIL | official SGB evaluator path not run or compared |
| SHAGCL-5 benchmark-scale metrics | FAIL | no checkpoint-backed official evaluation |

## Current Trusted Claim

SHA-GCL has an OpenSGG-compatible smoke adapter and an official-source audit.
Reproduction is deferred because required official inputs, checkpoint, and
protocol parity evidence are missing.

## Forbidden Claim

Do not claim SHA-GCL reproduction, paper-number alignment, or
checkpoint-backed performance from random-init/tiny-slice evidence.
