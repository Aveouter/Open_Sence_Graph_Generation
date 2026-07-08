# Baseline Reproduction Issue: EGTR

## Scope

- Baseline: EGTR
- Official repository: `https://github.com/naver-ai/egtr`
- Official local checkout: `/workspace/Item_code/egtr`
- Target dataset/split: Visual Genome `test`
- Target task: SGDet
- Target metrics: constrained single-predicate SGDet `R@50` and `mR@50`
- Training: not allowed; evaluation only

## Official Inputs

- Official archive: `outputs/pretrained/egtr/egtr_vg.tar.gz`
- Archive SHA256: `39e9f9ee9ff755ad4233f68a30333f8da756db3f2d95a236e3e1b5e142d5d2fe`
- Checkpoint symlink: `checkpoints/egtr/egtr_vg.ckpt`
- Checkpoint SHA256: `8926640facecd2784160c221c048dd4209df7d623ceb3c9eda289a11cb1f84d5`
- Official config member:
  `egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/config.json`
- Official metric member:
  `egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71__test__26446__V100-PCIE-32GB.json`

## Alignment Checklist

- [x] Official checkpoint is local and checksum-recorded.
- [x] Official checkpoint loads with `missing=0, unexpected=0`.
- [x] Official config values are available from the archive.
- [x] Visual Genome EGTR dataset/preprocessing wrapper is byte-identical to the official local checkout for `data/visual_genome.py`.
- [x] SGDet object scoring follows official `pred_logits.softmax(-1)[:, :num_labels]`.
- [x] SGDet relation scoring follows official `pred_rel * pred_connectivity`.
- [x] SGDet single-predicate top-k follows official pair ranking by max predicate score times subject/object scores.
- [x] Mean recall per-class evaluation filters GT relations only and keeps the full prediction list, matching official evaluator behavior.
- [x] Full VG test evaluation completed under `conda hsg` using GPU DDP.

## Completed Evaluation

Command:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 conda run --no-capture-output -n hsg \
  python train.py --test --method EGTR \
  --config_file configs/VisualGenome/EGTR.py \
  --ckpt_path checkpoints/egtr/egtr_vg.ckpt \
  --val_batch_size 1 --num_workers 4 --device cuda \
  --gpus 0 1 2 3 4 5 \
  --ex_name EGTR_official_parity_full_vg_ddp6 \
  --no_display_method_info
```

Run directory:

`outputs/runs/egtr/2026-07-08_EGTR_official_parity_full_vg_ddp6`

## Result Boundary

This is a checkpoint-backed full VG SGDet evaluation aligned to the official
EGTR archive reference. It is not a training reproduction and does not modify
labels, ground truth, evaluator thresholds, or failed samples.
