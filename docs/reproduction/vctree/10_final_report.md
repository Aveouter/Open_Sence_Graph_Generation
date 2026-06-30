# VCTree Final Reproduction Report

## 1. 方法简介

VCTree composes dynamic tree structures over visual objects and uses tree-based
context to predict scene graph predicates.

## 2. 官方来源

Primary source: `Learning to Compose Dynamic Tree Structures for Visual
Contexts`, Tang et al., CVPR 2019.

## 3. OpenSGG 集成状态

OpenSGG contains VCTree in `src/models/vctree.py`,
`src/methods/vctree_method.py`, and `configs/VisualGenome/VCTree.py`.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. No verified checkpoint was available.

## 6. Hidden-Positive Metrics

Random-init fallback JSONL export passed validation on 2 rows. R@1/R@5/R@10
and mR@1/mR@5/mR@10 were all 0.0.

## 7. Deviations

No checkpoint; random-init fallback; tiny CPU slice; local hidden-eval tooling.

## 8. 当前可信 Claim

VCTree can be instantiated and tested under `conda hsg`; its outputs work with
the standard metric and hidden/GT-aligned JSONL paths.

## 9. 不能声称的内容

Do not claim paper-level VCTree reproduction numbers or checkpoint-backed
performance.

## 10. 对本项目的作用

VCTree is available as a documented P0 baseline with no-training fallback test
evidence.

## 11. 下一步建议

Proceed to P1 PENet using checkpoint-first, test-only policy.
