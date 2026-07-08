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
- `tools/reproduction/check_penet_official_inputs.py` supports
  `--protocol sgdet` for target-specific gates and checks the `VG_100K` image
  directory required for full VG evaluation.

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
  --protocol sgdet \
  --output docs/reproduction/penet/penet_official_input_check.json
```

Current result: `BLOCKED`

Blocking inputs:

- Missing PENET-format VG root:
  `/workspace/external/penet_official/PENET/datasets/vg`
- Missing `VG-SGG-with-attri.h5`
- Missing `VG-SGG-dicts-with-attri.json`
- Missing `image_data.json`
- Missing VG image directory:
  `/workspace/external/penet_official/PENET/datasets/vg/VG_100K`
- Missing official detector checkpoint:
  `/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth`

The SGDet-targeted checker no longer reports missing PredCls/SGCls checkpoints
as blockers for this task.

Official/manual download links recorded from PENET and Scene-Graph-Benchmark:

- VG images part 1:
  `https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip`
- VG images part 2:
  `https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip`
- Official scene-graph files:
  `https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed`
- Official pretrained Faster R-CNN:
  `https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw`
- Backup Baidu link:
  `https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA`, extraction code `1234`
- Backup Weiyun link:
  `https://share.weiyun.com/ViTWrFxG`

Network note:

- Current environment resolves the official OneDrive resources but `curl`
  exits with TLS `unexpected eof while reading` before a downloadable payload is
  received.
- A non-official Hugging Face mirror candidate for `VG-SGG-with-attri.h5` was
  downloaded only for structure inspection:
  `outputs/pretrained/penet_official/mirrors/VG-SGG-with-attri.h5`
- Mirror candidate SHA256:
  `ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1`
- The mirror H5 contains SGB/PENET-style keys including `boxes_1024`,
  `boxes_512`, `labels`, `relationships`, `predicates`, `attributes`, and
  `split`; it is not treated as official provenance by this report.

## Evaluation Boundary

No checkpoint-backed SGDet metric run has been performed yet. Running the
current local code without official VG inputs and detector proposals would be a
protocol mismatch, not an official PE-NET SGDet reproduction.
