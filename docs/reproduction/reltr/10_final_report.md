# RelTR Final Reproduction Report

## 1. 方法简介

RelTR is a transformer-based scene graph generation model using entity and
triplet queries.

## 2. 官方来源

Primary source: `RelTR: Relation Transformer for Scene Graph Generation`, ECCV
2022.

## 3. OpenSGG 集成状态

OpenSGG contains RelTR in `src/models/reltr.py`,
`src/methods/reltr_method.py`, and `configs/VisualGenome/RelTR.py`.

## 4. Standard SGG Metrics

Checkpoint-backed PredCls metric slice with one batch:

- R@10/R@20/R@50/R@100: all 0.0
- mR@10/mR@20/mR@50/mR@100: all 0.0

## 5. Paper-Number Alignment

Not claimed from this tiny validation run.

## 6. Hidden-Positive Metrics

Checkpoint-backed JSONL export passed validation on 2 rows. R@1/R@5/R@10 and
mR@1/mR@5/mR@10 were all 0.0.

## 7. Deviations

Tiny CPU slice, local hidden-eval tooling, PR blocked.

## 8. 当前可信 Claim

RelTR checkpoint-backed OpenSGG inference works under `conda hsg` and feeds both
standard metric and hidden/GT-aligned JSONL paths.

## 9. 不能声称的内容

Do not claim full RelTR benchmark reproduction or paper-level alignment.

## 10. 对本项目的作用

RelTR is documented as a P2 checkpoint-backed baseline with no-training test
evidence.

## 11. 下一步建议

Proceed to P2 EGTR.
