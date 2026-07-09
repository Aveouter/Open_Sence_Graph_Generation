# PENet SGDet PR84 Parity Report

## Scope

This report covers the SGDet adapter work layered on PR84
(`fix(penet): align PredCls/SGCls evaluation with official PENet`). It does not
claim full reproduction of the PE-NET SGDet table.

## Official Target

- Official SGDet R@20/50/100: `23.36 / 30.41 / 34.84`
- Official SGDet mR@20/50/100: `9.16 / 12.25 / 14.34`
- Fractional targets used for comparison: R@50 `0.3041`, mR@50 `0.1225`

## Implemented Alignment

- Split detector and relation ROI box extractors in the SGDet path.
- Load `roi_heads.relation.box_feature_extractor.*` into the relation extractor.
- Load `roi_heads.box.feature_extractor.*`, `rpn.head.*`, and
  `roi_heads.box.predictor.*` into the frozen detector proposal path.
- Re-extract relation ROI features with the relation extractor after detector
  proposals.
- Remap official `roi_heads.relation.predictor.*` tensors into the local
  `PENetContext` predictor.
- Add PE-NET SGDet compact outputs and a model-family-specific evaluator path.
- Handle relation score tensors with background column even when epoch-end passes
  foreground-only `rel_nums=50`.
- Match the official X-101 SGDet test config by using
  `TEST.RELATION.REQUIRE_OVERLAP=False`.
- Convert OpenSGG RGB/ImageNet-normalized eval tensors back to the official
  maskrcnn-benchmark BGR255 pixel-mean space before PE-NET detector inference,
  then pad to the official size-divisible-by-32 image list shape.
- Match the official FPN top-down interpolation call and union feature
  `POOLING_ALL_LEVELS` reduce-channel `Conv2d + ReLU` block.

## Tensor Parity

`tests/reproduction/test_penet_adapter.py` verifies exact tensor equality between
local loaded extractors and official checkpoint keys:

- `roi_heads.relation.box_feature_extractor.fc6/fc7.{weight,bias}`
- `roi_heads.box.feature_extractor.fc6/fc7.{weight,bias}`

The test also asserts the relation and detector `fc7.weight` tensors are not
identical, guarding against the original extractor-mixing bug.

The same test module verifies that official `roi_heads.relation.predictor.*`
keys remap into local `PENetContext` keys and that representative predictor
tensors, including `post_emb.weight`, `vis2sem.0.weight`, `out_obj.weight`, and
`lin_obj_cyx.weight`, exactly match the official SGDet checkpoint after loading.

The tests also guard official config/preprocessing details:

- `--penet_sgdet_require_overlap False` parses to boolean `False`.
- The PE-NET X-101 fallback for `sgdet_require_overlap` is `False`.
- The union feature pooler's `reduce_channel` block includes the official ReLU.

## Smoke Evidence

64-image SGDet smoke on 5 RTX 2080 Ti GPUs after aligning official no-overlap
config, size-divisible padding, and union reduce-channel ReLU:

- Command: `CUDA_VISIBLE_DEVICES=2,3,4,5,6 conda run --no-capture-output -n hsg python train.py --test --method PENet --config_file configs/VisualGenome/PE_NET.py --ckpt_path /workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth --test_dataset_size 64 --val_batch_size 1 --num_workers 1 --penet_sgdet_eval_topk 100 --penet_sgdet_require_overlap False --ex_name PENet_pr84base_official_sgdet_smoke64_union_reduce_relu --output_dir /workspace/Item_code/OpenSGG/outputs --no_display_method_info --gpus 0 1 2 3 4`
- Metrics path: `/workspace/Item_code/OpenSGG/outputs/runs/penet/2026-07-09_PENet_pr84base_official_sgdet_smoke64_union_reduce_relu/eval/sgdet/metrics.json`
- `sgdet_R@20 = 0.20146863162517548`
- `sgdet_R@50 = 0.24727560579776764`
- `sgdet_R@100 = 0.276100218296051`
- `sgdet_mR@20 = 0.05804852023720741`
- `sgdet_mR@50 = 0.0665256679058075`
- `sgdet_mR@100 = 0.08660943806171417`

The previous 64-image smoke from the earlier SGDet branch had
`sgdet_R@50 = 0.007558847311884165` and
`sgdet_mR@50 = 0.011564625427126884`. The new smoke is diagnostic only.

Earlier smoke/full runs with overlap filtering or without the official union
reduce ReLU are superseded diagnostics and are not used as the current
comparison row.

## Full Visual Genome Result

Full SGDet evaluation on the complete OpenSGG Visual Genome test split completed
on 5 RTX 2080 Ti GPUs with official X-101 no-overlap config, official BGR255
pad32 preprocessing, and union reduce-channel ReLU:

- Command: `CUDA_VISIBLE_DEVICES=2,3,4,5,6 conda run --no-capture-output -n hsg python train.py --test --method PENet --config_file configs/VisualGenome/PE_NET.py --ckpt_path /workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth --val_batch_size 1 --num_workers 1 --penet_sgdet_eval_topk 100 --penet_sgdet_require_overlap False --ex_name PENet_pr84base_official_sgdet_full_vg_no_overlap_union_relu --output_dir /workspace/Item_code/OpenSGG/outputs --no_display_method_info --gpus 0 1 2 3 4`
- Metrics path: `/workspace/Item_code/OpenSGG/outputs/runs/penet/2026-07-09_PENet_pr84base_official_sgdet_full_vg_no_overlap_union_relu/eval/sgdet/metrics.json`

| Metric | Local full VG | Official table | Gap, percentage points |
|---|---:|---:|---:|
| R@20 | 21.96 | 23.36 | -1.40 |
| R@50 | 28.82 | 30.41 | -1.59 |
| R@100 | 33.08 | 34.84 | -1.76 |
| mR@20 | 8.85 | 9.16 | -0.31 |
| mR@50 | 11.94 | 12.25 | -0.31 |
| mR@100 | 13.95 | 14.34 | -0.39 |

Raw fractional metrics:

- `sgdet_R@20 = 0.21955223381519318`
- `sgdet_R@50 = 0.2882086932659149`
- `sgdet_R@100 = 0.33078208565711975`
- `sgdet_mR@20 = 0.08854109793901443`
- `sgdet_mR@50 = 0.11937950551509857`
- `sgdet_mR@100 = 0.1394738107919693`

## Current Gap

- The visible gap is now concentrated in R@K. mR@50 is close to the official
  table, but R@50 remains `1.59` percentage points below the reported row.
- VG h5/test split parity was checked separately: the local test split and
  official h5 test split both contain `26446` relation-bearing test images.
- Detector anchors, detector config, box NMS, and relation postprocessor
  semantics have been checked against the official repository.
- The next likely sources are lower-level ROIAlign/FPN numeric parity or a
  side-by-side run of the original PENET/SGB process from the same local inputs.
- The local run still uses an OpenSGG runtime wrapper rather than the original
  PENET/SGB process.
- Smoke metrics must not be reported as PE-NET baseline table results.
