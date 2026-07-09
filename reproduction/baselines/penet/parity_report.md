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
- Match official SGDet test pair sampling by keeping only overlapping detected
  box pairs when `TEST.RELATION.REQUIRE_OVERLAP=True`.

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

## Smoke Evidence

64-image SGDet smoke on 5 RTX 2080 Ti GPUs after enabling official overlap pair
filtering:

- Command: `CUDA_VISIBLE_DEVICES=2,3,4,5,6 conda run --no-capture-output -n hsg python train.py --test --method PENet --config_file configs/VisualGenome/PE_NET.py --ckpt_path /workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth --test_dataset_size 64 --val_batch_size 1 --num_workers 1 --penet_sgdet_eval_topk 100 --ex_name PENet_pr84base_official_sgdet_smoke64_overlap_pairs --output_dir /workspace/Item_code/OpenSGG/outputs --no_display_method_info --gpus 0 1 2 3 4`
- Metrics path: `/workspace/Item_code/OpenSGG/outputs/runs/penet/2026-07-09_PENet_pr84base_official_sgdet_smoke64_overlap_pairs_004/eval/sgdet/metrics.json`
- `sgdet_R@20 = 0.17721953988075256`
- `sgdet_R@50 = 0.21474899351596832`
- `sgdet_R@100 = 0.257851779460907`
- `sgdet_mR@20 = 0.052009839564561844`
- `sgdet_mR@50 = 0.061864517629146576`
- `sgdet_mR@100 = 0.0743805319070816`

The previous 64-image smoke from the earlier SGDet branch had
`sgdet_R@50 = 0.007558847311884165` and
`sgdet_mR@50 = 0.011564625427126884`. The new smoke is diagnostic only.

An earlier full run was interrupted at 5210/5290 images after discovering the
missing official overlap-pair filter. That run is discarded and is not used as
evidence.

## Full Visual Genome Result

Full SGDet evaluation on the complete OpenSGG Visual Genome test split completed
on 5 RTX 2080 Ti GPUs with official overlap-pair filtering enabled:

- Command: `CUDA_VISIBLE_DEVICES=2,3,4,5,6 conda run --no-capture-output -n hsg python train.py --test --method PENet --config_file configs/VisualGenome/PE_NET.py --ckpt_path /workspace/Item_code/OpenSGG/outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth --val_batch_size 1 --num_workers 1 --penet_sgdet_eval_topk 100 --ex_name PENet_pr84base_official_sgdet_full_vg_overlap_pairs --output_dir /workspace/Item_code/OpenSGG/outputs --no_display_method_info --gpus 0 1 2 3 4`
- Metrics path: `/workspace/Item_code/OpenSGG/outputs/runs/penet/2026-07-09_PENet_pr84base_official_sgdet_full_vg_overlap_pairs_001/eval/sgdet/metrics.json`

| Metric | Local full VG | Official table | Gap, percentage points |
|---|---:|---:|---:|
| R@20 | 21.60 | 23.36 | -1.76 |
| R@50 | 28.40 | 30.41 | -2.01 |
| R@100 | 32.58 | 34.84 | -2.26 |
| mR@20 | 8.86 | 9.16 | -0.30 |
| mR@50 | 11.50 | 12.25 | -0.75 |
| mR@100 | 13.28 | 14.34 | -1.06 |

Raw fractional metrics:

- `sgdet_R@20 = 0.21597830951213837`
- `sgdet_R@50 = 0.2839694917201996`
- `sgdet_R@100 = 0.3257637023925781`
- `sgdet_mR@20 = 0.088646799325943`
- `sgdet_mR@50 = 0.11496535688638687`
- `sgdet_mR@100 = 0.13284049928188324`

## Current Gap

- The remaining full-run gap is now small but still visible, especially R@50
  and R@100.
- The next likely sources are detector proposal parity, evaluator parity, or
  data-protocol parity.
- The local run still uses an OpenSGG runtime wrapper rather than the original
  PENET/SGB process.
- Smoke metrics must not be reported as PE-NET baseline table results.
