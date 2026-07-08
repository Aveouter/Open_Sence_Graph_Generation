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

## Official Backup Inputs

Automated input check:

```bash
python tools/reproduction/check_penet_official_inputs.py \
  --protocol sgdet \
  --output docs/reproduction/penet/penet_official_input_check.json
```

Current result: `PASS`

Official backup inputs now present:

- PENET-format VG root:
  `/workspace/external/penet_official/PENET/datasets/vg`
- Official-backup `VG-SGG-with-attri.h5`
  - Size: `150724042` bytes
  - SHA1 from Weiyun metadata:
    `345119e71d16e387a0154164379d918dd72cbb16`
  - SHA256:
    `a6370b4438991a4a866f02445a10d4123b8c8cea4a8e35828a0abb6aa7f739e8`
- `VG-SGG-dicts-with-attri.json`
  - SHA256:
    `1255873ae0250555f59d20f8d1e5a4bdf22b331119502c850f765173b6963ca1`
- `image_data.json`
  - SHA256:
    `5a0b63286b6ec81bcae17df1d4a50777bfc56befb70d65a8a7e9bcab0fcd9bc4`
- VG image directory:
  `/workspace/external/penet_official/PENET/datasets/vg/VG_100K`
  - Linked to the existing local VG image pack used for RELTR:
    `/workspace/Item_code/OpenSGG/data/VisualGenome/images`
  - Image count: `108249` jpg files
- Official-backup pretrained Faster R-CNN detector:
  `/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth`
  - Size: `1293168527` bytes
  - SHA256:
    `b2cf9b2b4771a340c5dd02a30a9e4d08d14d032de38704545fec643801feba9e`

The detector archive was retrieved from the official Weiyun backup:

- Archive:
  `/workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/downloads/pretrained_faster_rcnn.zip`
- Archive size: `1201052079` bytes
- SHA1 from Weiyun metadata:
  `a8329c1a656af4972c1a5fa07a2d829fea3921d6`
- Archive SHA256:
  `8039ddf7fa0414250349bdeeca5cc2268407db8626498a40efbcea2b1d003fae`

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

- Current environment can reach the official Weiyun backup through the
  container proxy at `127.0.0.1:17891`.
- Current environment resolves the official OneDrive resources but `curl`
  exits with TLS `unexpected eof while reading` before a downloadable payload is
  received.
- The same OneDrive failure was reproduced with `wget`, Python `urllib`,
  `curl --http1.1 --tlsv1.2`, and a small range request against
  `onedrive.live.com/download?resid=22376FFAD72C4B64%21779870`; no detector
  bytes were retrieved from OneDrive.
- Stanford VG image links respond with HTTP 200 and byte ranges, but the
  existing local raw VG image pack is already the same `VG_100K_2` image source
  required by PENET/SGB and was reused instead of redownloading.
- A non-official Hugging Face mirror candidate for `VG-SGG-with-attri.h5` was
  downloaded only for structure inspection:
  `outputs/pretrained/penet_official/mirrors/VG-SGG-with-attri.h5`
- Mirror candidate SHA256:
  `ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1`
- The mirror H5 contains SGB/PENET-style keys including `boxes_1024`,
  `boxes_512`, `labels`, `relationships`, `predicates`, `attributes`, and
  `split`; it is not treated as official provenance by this report.

## Eval-Only Probes

Local OpenSGG eval-only probe:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n hsg python train.py \
  --test \
  --method PENet \
  --config_file configs/VisualGenome/PE_NET.py \
  --ckpt_path outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth \
  --test_dataset_size 1 \
  --val_batch_size 1 \
  --num_workers 0 \
  --ex_name PENet_official_sgdet_probe \
  --no_display_method_info \
  --gpus 0
```

Observed result:

- Official relation checkpoint loaded through the remapping path:
  `Remapped external checkpoint keys: 642 -> 63`.
- Missing local-only fallback tensors were limited to:
  `union_fallback.0.weight` and `union_fallback.0.bias`.
- The run stopped before metric reporting because the local VisualGenome eval
  target does not provide official SGDet detector proposal fields:
  `boxes_per_cls` and `obj_dists`/`scores_all`.

This is the intended guard behavior. Producing R@50/mR@50 without those
proposal fields would be a protocol mismatch.

Official PENET runtime probe in `conda hsg`:

```bash
cd /workspace/external/penet_official/PENET
conda run --no-capture-output -n hsg python setup.py build_ext --inplace
```

Observed result:

- `apex` is not installed in `hsg`; official `relation_test_net.py` imports
  `apex.amp` at module import time.
- `maskrcnn_benchmark._C` is not importable before building.
- Building the official extension fails under the current Python 3.10 /
  modern PyTorch/CUDA environment with legacy maskrcnn-benchmark C++/CUDA
  compilation errors, including missing legacy THC headers and a final
  `RuntimeError: Error compiling objects for extension`.

This is an official-code runtime environment blocker, not a missing-input
blocker. The official repository's legacy stack needs a compatible
maskrcnn-benchmark/APEX environment before the official evaluator can be used
as the parity oracle.

## Evaluation Boundary

No checkpoint-backed SGDet R@50/mR@50 result has been produced from local
OpenSGG. The current status is:

- official input provenance: `PASS`
- local OpenSGG SGDet evaluation: `protocol_mismatch` until detector proposal
  fields are provided
- official PENET evaluator: `runtime_blocked` in the current `hsg` environment
- paper table targets only:
  `R@50 = 30.41` and `mR@50 = 12.25`
