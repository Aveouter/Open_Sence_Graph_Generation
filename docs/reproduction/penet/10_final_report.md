# PENet Final Reproduction Report

## 1. 方法简介

PENet is a prototype-based embedding network for SGG that combines semantic
object prototypes with visual features for predicate prediction.

## 2. 官方来源

Primary source: `Prototype-based Embedding Network for Scene Graph Generation`,
Zheng et al., CVPR 2023.

## 3. OpenSGG 集成状态

OpenSGG contains PENet in `src/models/penet.py`,
`src/methods/penet_method.py`, and `configs/VisualGenome/PE_NET.py`.

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

No checkpoint; random-init fallback; tiny CPU slice; local hidden-eval tooling;
GitHub PR blocked by remote/proxy tooling.

## 8. 当前可信 Claim

PENet can be instantiated and tested under `conda hsg`; its outputs work with
the standard metric and hidden/GT-aligned JSONL paths.

## 9. 不能声称的内容

Do not claim paper-level PENet reproduction numbers or checkpoint-backed
performance.

## 10. 对本项目的作用

PENet is available as a documented P1 baseline with no-training fallback test
evidence.

## 11. 下一步建议

Proceed to P1 SHA-GCL using checkpoint-first, test-only policy.
