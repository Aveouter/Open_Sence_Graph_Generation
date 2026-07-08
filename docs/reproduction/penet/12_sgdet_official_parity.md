# PENet SGDet Official Parity Report

Status: `protocol_mismatch` / `not_reproduction_ready`

No training was run. This report only records local code and official GitHub
alignment work for the SGDet target.

## Official Anchor

- Official repository: `https://github.com/VL-Group/PENET`
- Local official checkout: `/workspace/external/penet_official/PENET`
- Reference commit: `9c9f50777c66647799cb7a17dfb855aa98aecd1e`
- Official predictor: `PrototypeEmbeddingNetwork` in
  `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
- Official config: `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`

Official README SGDet table row:

- R@20/R@50/R@100: `23.36 / 30.41 / 34.84`
- mR@20/mR@50/mR@100: `9.16 / 12.25 / 14.34`

OpenSGG metric fractions should therefore align to approximately:

- `sgdet_R@50 = 0.3041`
- `sgdet_mR@50 = 0.1225`

These numbers are targets from the official README, not local results.

## Local Alignment Changes

- `configs/VisualGenome/PE_NET.py` now defaults to `eval_mode = "sgdet"` for
  the PE-NET SGDet reproduction target.
- `src/methods/penet_method.py` treats both SGCls and SGDet as object-prediction
  modes instead of letting SGDet fall back to GT labels.
- `src/methods/penet_method.py` explicitly requires SGDet detector proposal
  fields: `boxes_per_cls` plus `predict_logits`/`scores_all`.
- `src/models/penet.py` can remap official checkpoint keys from
  `roi_heads.relation.predictor.*` into local `PENetContext`.
- `tools/reproduction/check_penet_official_inputs.py` now checks checkpoint
  directories per protocol and no longer counts an SGDet checkpoint as a
  PredCls/SGCls checkpoint.

## Official Checkpoint

Downloaded official SGDet checkpoint:

- Path: `outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth`
- Size: `2627548389` bytes
- SHA256:
  `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183`
- Official Google Drive ID: `1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C`

Key inspection:

- Top-level checkpoint keys: `model`, `optimizer`, `iteration`
- Model tensor count: `642`
- Official relation predictor tensors:
  `63` keys under `roi_heads.relation.predictor.*`
- Sample matched keys include `post_emb`, `obj_embed`, `rel_embed`, `W_sub`,
  `W_obj`, `W_pred`, gates, `vis2sem`, `project_head`, and `logit_scale`.
- Dry-loading those remapped keys into local `PENetContext` gives:
  `remapped_keys=63`, `unexpected=0`, and two explained missing local-only
  fallback parameters: `union_fallback.0.weight` and `union_fallback.0.bias`.
  The fallback is not part of the official predictor path when official union
  features are available.

## Remaining Blockers

Automated input check:

```bash
python tools/reproduction/check_penet_official_inputs.py \
  --output docs/reproduction/penet/penet_official_input_check.json
```

Current result: `BLOCKED`

Blocking inputs:

- Missing PENET-format VG root:
  `/workspace/external/penet_official/PENET/datasets/vg`
- Missing `VG-SGG-with-attri.h5`
- Missing `VG-SGG-dicts-with-attri.json`
- Missing `image_data.json`
- Missing official detector checkpoint:
  `/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth`

The checker also reports missing PredCls/SGCls checkpoints, but those are
non-target protocols for this SGDet task.

## Evaluation Boundary

No checkpoint-backed SGDet metric run has been performed yet. Running the
current local code without official VG inputs and detector proposals would be a
protocol mismatch, not an official PE-NET SGDet reproduction.
