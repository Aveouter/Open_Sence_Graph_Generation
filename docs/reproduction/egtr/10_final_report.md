# EGTR Final Reproduction Report

## 1. 方法简介

EGTR is a transformer-based scene graph generation method built on Deformable
DETR-style object queries and dense relation prediction.

## 2. 官方来源

Primary source: `EGTR: Extracting Graph from Transformer for Scene Graph
Generation`, CVPR 2024.

## 3. OpenSGG 集成状态

OpenSGG already contains EGTR in `src/methods/egtr_method.py`,
`src/modules/egtr/`, and `configs/VisualGenome/EGTR.py`. The local checkpoint
was loaded for inference/export.

## 4. Standard SGG Metrics

Direct Motifs-style PredCls metric validation is not claimed for EGTR. A direct
probe failed with target-schema mismatch (`class_labels`) because EGTR's
criterion expects EGTR-formatted targets, while the reliable checkpoint-backed
path runs SGDet query inference and compact GT alignment.

## 5. Paper-Number Alignment

Not claimed from this tiny validation run.

## 6. Hidden-Positive Metrics

Checkpoint-backed GT-aligned JSONL export passed validation on 2 rows:

- R@1/R@5/R@10: 0.0 / 0.5 / 1.0
- mR@1/mR@5/mR@10: 0.0 / 0.5 / 1.0

## 7. Deviations

Tiny CPU slice, GT-aligned hidden-eval adapter, no training, no full benchmark,
no paper-number alignment, direct Motifs-style PredCls not claimed, PR blocked.

## 8. 当前可信 Claim

EGTR checkpoint-backed OpenSGG inference works under `conda hsg` for no-training
GT-aligned relation export, and the hidden-positive JSONL/recall pipeline is
validated on a two-image slice.

## 9. 不能声称的内容

Do not claim full EGTR benchmark reproduction, paper-level alignment, or
Motifs-style PredCls equivalence.

## 10. 对本项目的作用

EGTR is documented as a P2 checkpoint-backed transformer baseline with a clear
evaluation-scope boundary.

## 11. 下一步建议

Run the full EGTR official protocol on GPU resources if paper-number alignment
is required, and create an independent PR once GitHub transport/tooling is
available.
