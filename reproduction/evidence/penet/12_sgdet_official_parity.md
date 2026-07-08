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
- `utils/penet_weights.py` now maps the official detector checkpoint's real
  key names into local PE-NET detector feature modules:
  `backbone.body.*` to the local ResNeXt backbone,
  `backbone.fpn.*` to `FPNNeck`, and
  `roi_heads.box.feature_extractor.*` to `PENetBoxFeatureExtractor`,
  plus `rpn.head.*` and `roi_heads.box.predictor.*` to the local SGDet
  proposal generator.
- `src/models/penet_detector.py` adds a PE-NET-only eval detector proposal
  path for SGDet: official-style RPN anchors, box coder, RPN postprocessing,
  box predictor, and relation proposal fields (`boxes_per_cls`,
  `predict_logits`/`obj_dists`).
- `utils/parser.py` exposes `--penet_detector_ckpt` so eval-only runs can
  point at the official detector checkpoint without hardcoding local absolute
  paths in the committed config.
- `data/dataloaders/coco.py` keeps the default Visual Genome eval resize at
  `800/1333`, but allows a config to opt into explicit eval resize settings.
- `data/dataloaders/vg_official_h5.py` adds a PE-NET eval-only official
  PENET/SGB H5 dataset path that preserves duplicate test relations and
  H5-derived float boxes.
- `configs/VisualGenome/PE_NET.py` now sets official PENET eval preprocessing:
  `eval_min_size = 600` and `eval_max_size = 1000`.
- `configs/VisualGenome/PE_NET.py` enables
  `penet_use_official_vg_h5_eval = True`; the official root is supplied at
  runtime with `--penet_official_vg_root`.
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
  - Local PE-NET detector-feature weight-load probe:
    `backbone=520`, `fpn=16`, `box_extractor=4`,
    `rpn_head=6`, `box_predictor=4`

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

## Data Gap Audit

Local OpenSGG Visual Genome files were compared against the official
PENET/SGB H5 and metadata files:

- Official H5 image rows: `108073`
- Official H5 split counts: `split=0 -> 75651`, `split=2 -> 32422`
- Local OpenSGG JSON image counts:
  `train=57723`, `val=5000`, `test=26446`
- Local JSON image filenames are all present in official `image_data.json`.
- Local JSON does not cover `18904` official H5 image rows:
  `12928` from official `split=0` and `5976` from official `split=2`.
- The missing official rows are zero-relation rows:
  local train/val/test keep only images with at least one relation.
- For the local test image set, official H5 object count is `325570`; local
  `test.json` object annotation count is also `325570`.
- Object class labels match exactly for the sampled full-test alignment check.
- Local image width/height fields match official `image_data.json`.
- Object vocabulary matches official `label_to_idx` exactly for 151 entries
  including background index 0.
- Predicate foreground vocabulary order matches official `predicate_to_idx`
  for ids `1..50`; local `rel_categories[0]` explicitly stores
  `__background__`.

Relation annotation gap:

- For local test images, official H5 contains `183642` relationship rows.
- Local `rel.json` contains `152226` test relationship rows.
- The local count equals the number of unique official
  `(subject_box_idx, object_box_idx, predicate)` triples on those same images.
- The missing `31416` official rows are exact duplicate relationship triples,
  not missing predicate classes.
- Official PENET keeps duplicate `relation_tuple` rows for test/evaluation;
  duplicate filtering is only enabled for training in the official
  `VGDataset`.

Box-coordinate gap:

- Local boxes and official boxes have the same per-image object counts and
  labels, but the coordinate representation is not byte-identical.
- Official evaluator reconstructs float `xyxy` boxes from `boxes_1024`.
- Local OpenSGG uses integer COCO-style `xywh` boxes in JSON.
- On all `325570` local test boxes, the maximum absolute coordinate
  difference from official H5-derived `xywh` is `1.0` pixel and the mean
  absolute coordinate difference is approximately `0.681` pixels.

Data-level conclusion:

- The local OpenSGG VG files are derived from the official SGB/PENET data, but
  they are not an exact evaluator-input clone.
- A PE-NET-only official H5 eval dataset path has now been added to evaluate
  directly from the official H5 loader semantics.
- Strict official parity still requires full-VG metrics from this new H5 eval
  path, plus evaluator/proposal runtime parity.

## Eval-Only Probes

Local OpenSGG full-VG eval-only run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n hsg python train.py \
  --test \
  --method PENet \
  --config_file configs/VisualGenome/PE_NET.py \
  --ckpt_path outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth \
  --penet_detector_ckpt /workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth \
  --penet_official_vg_root /workspace/external/penet_official/PENET/datasets/vg \
  --val_batch_size 1 \
  --num_workers 4 \
  --ex_name PENet_official_h5_sgdet_full_vg_union_roi \
  --no_display_method_info \
  --gpus 0
```

Observed full-test result:

- Run directory:
  `outputs/runs/penet/2026-07-08_PENet_official_h5_sgdet_full_vg_union_roi`
- Test split size: `26446/26446` images
- Runtime shown by Lightning: `0:40:24`
- Metrics file:
  `outputs/runs/penet/2026-07-08_PENet_official_h5_sgdet_full_vg_union_roi/eval/sgdet/metrics.json`
- `sgdet_R@20 = 0.0`
- `sgdet_R@50 = 0.0`
- `sgdet_R@100 = 0.0`
- `sgdet_mR@20 = 0.0`
- `sgdet_mR@50 = 0.0`
- `sgdet_mR@100 = 0.0`
- The run loaded the official SGDet relation checkpoint, official detector
  checkpoint, and official relation union feature extractor weights:
  `Loaded PE-NET union extractor weights: 20`.
- The run used official H5 eval inputs via
  `--penet_official_vg_root /workspace/external/penet_official/PENET/datasets/vg`.
- The run used PE-NET official eval preprocessing:
  `eval_min_size = 600`, `eval_max_size = 1000`.

Comparison with the official README SGDet row:

- Official target `R@50 = 0.3041`; local full-VG result:
  `sgdet_R@50 = 0.0`.
- Official target `mR@50 = 0.1225`; local full-VG result:
  `sgdet_mR@50 = 0.0`.

This is checkpoint-backed full-VG evaluation evidence, but it is
`protocol_mismatch` / `not_reproduction_ready`, not a successful reproduction.
The stricter PE-NET SGDet compact evaluator now consumes detector boxes,
detector object scores/classes, relation softmax scores, and official-style
mean-recall semantics. Under that stricter evaluator the local adapter still
does not match the official runtime.

Smaller probes after the evaluator and feature-alignment changes:

- `PENet_official_sgdet_compact_64probe`: 64 images, all R/mR values `0.0`.
- `PENet_official_sgdet_union_64probe`: loaded 18 union extractor weights,
  64 images, all R/mR values `0.0`.
- `PENet_official_sgdet_union_all_levels_64probe_v2`: loaded 20 union
  extractor weights after enabling official all-level union pooling,
  64 images, all R/mR values `0.0`.
- `PENet_official_sgdet_roi_align_64probe`: switched local ROIAlign to
  maskrcnn-benchmark-compatible `aligned=False`, 64 images, all R/mR values
  `0.0`.

These probes are `pipeline_smoke_only`; they are not reproduction evidence.

Detector checkpoint load probe:

```bash
conda run --no-capture-output -n hsg python - <<'PY'
from src.models.backbone import ResNetBackbone, FPNNeck, PENetBoxFeatureExtractor
from utils.penet_weights import load_detector_checkpoint
ckpt = "/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth"
backbone = ResNetBackbone("resnext101_32x8d", pretrained=False, frozen=True)
fpn = FPNNeck((256, 512, 1024, 2048), 256)
box = PENetBoxFeatureExtractor()
print(load_detector_checkpoint(backbone, fpn, box, ckpt))
PY
```

Observed result:
`{'backbone': 520, 'fpn': 16, 'box_extractor': 4, 'rpn_head': 6, 'box_predictor': 4}`.
This confirms local detector feature modules can consume the official detector
weights needed for SGDet proposal generation.

Union feature extractor checkpoint load:

- Official SGDet checkpoint contains 20 tensors under
  `roi_heads.relation.union_feature_extractor.*`.
- Local PE-NET now maps all 20 tensors into `PENetUnionFeatureExtractor`,
  including `rect_conv`, `fc6`, `fc7`, and
  `feature_extractor.pooler.reduce_channel.0.*`.
- Local PE-NET union pooling now uses official relation-head semantics:
  P2-P5 all-level ROI pooling, channel concatenation, `reduce_channel`, and
  legacy maskrcnn-benchmark-style ROIAlign (`aligned=False`).

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

The full-VG checkpoint-backed SGDet run completed, but it does not align with
the official table. The current status is:

- official input provenance: `PASS`
- local OpenSGG SGDet evaluation: full VG completed on official H5 inputs and
  official `600/1000` preprocessing, but local `sgdet_R@50 = 0.0` and
  `sgdet_mR@50 = 0.0` do not align with official targets `0.3041` and
  `0.1225`
- official PENET evaluator: `runtime_blocked` in the current `hsg` environment
- paper table targets only:
  `R@50 = 30.41` and `mR@50 = 12.25`

Known parity gaps to resolve before this can be counted as reproduction:

- Detector proposal path mismatch risk: local SGDet proposals are generated by
  a PE-NET-only adapter that loads official detector weights, but it is not the
  official maskrcnn-benchmark `GeneralizedRCNN` runtime.
- Feature/runtime parity remains unproven at the full detector stack level:
  even after loading official detector, relation predictor, and union extractor
  weights, local proposals/features produce zero strict SGDet recall.
- Official PENET evaluator remains blocked in `conda hsg` by the legacy
  maskrcnn-benchmark/APEX stack; a compatible official runtime is still needed
  as the parity oracle.
