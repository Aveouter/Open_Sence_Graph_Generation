# VCTree Final Reproduction Report

**Superseded by stricter alignment audit:** this report documents random-init
pipeline evidence only. VCTree is currently `DEFERRED_NOT_REPRODUCED` because no
verified official checkpoint/protocol parity was established.

## 1. 方法简介

VCTree composes dynamic tree structures over visual objects and uses tree-based
context to predict scene graph predicates.

## 2. 官方来源

Primary source: `Learning to Compose Dynamic Tree Structures for Visual
Contexts`, Tang et al., CVPR 2019.

The current gate audit also pins the SGB reference implementation at
`/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
commit `ceb71fa88461c2a97a6258a80f47669d89207296`.

## 3. OpenSGG 集成状态

OpenSGG contains VCTree in `src/models/vctree.py`,
`src/methods/vctree_method.py`, and `configs/VisualGenome/VCTree.py`.

Source audit shows this is a simplified local VCTree-style implementation, not
SGB `VCTreePredictor` parity. It does not use SGB `VCTreeLSTMContext`,
`generate_forest`, `arbForest_to_biForest`, or SGB statistics-backed
`FrequencyBias`.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. No verified checkpoint was available, and SGB-format VG roidb
input is missing.

## 6. Hidden-Positive Metrics

Random-init fallback JSONL export passed validation on 2 rows. R@1/R@5/R@10
and mR@1/mR@5/mR@10 were all 0.0.

## 7. Deviations

No checkpoint; missing SGB-format roidb input; local simplified implementation;
random-init fallback; tiny CPU slice; local hidden-eval tooling.

## 8. 当前可信 Claim

The local VCTree-named implementation can be instantiated and tested under
`conda hsg`; its outputs work with the standard metric and hidden/GT-aligned
JSONL paths. This is pipeline/audit evidence only.

## 9. 不能声称的内容

Do not claim paper-level VCTree reproduction numbers, SGB `VCTreePredictor`
parity, or checkpoint-backed performance.

## 10. 对本项目的作用

VCTree is available as a documented P0 baseline with no-training fallback test
evidence.

## 11. 下一步建议

Provide SGB-format VG inputs plus a trusted VCTree checkpoint, run official SGB
VCTree evaluation, then decide whether OpenSGG should import/adapt the official
VCTree structure or keep the current implementation as audit/smoke only.
