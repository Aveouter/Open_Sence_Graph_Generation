# PENet Final Reproduction Report

**Superseded by stricter alignment audit:** this report documents
audit/smoke evidence only. PENet is currently `DEFERRED_NOT_REPRODUCED`
because official inputs and protocol parity are missing.

## 1. 方法简介

PENet is a prototype-based embedding network for SGG that combines semantic
object prototypes with visual features for predicate prediction.

## 2. 官方来源

Primary source: `Prototype-based Embedding Network for Scene Graph Generation`,
Zheng et al., CVPR 2023.

Official repository: `https://github.com/VL-Group/PENET`, inspected locally at
commit `9c9f50777c66647799cb7a17dfb855aa98aecd1e`.

Official predictor: `PrototypeEmbeddingNetwork`.

## 3. OpenSGG 集成状态

OpenSGG contains PENet in `src/models/penet.py`,
`src/methods/penet_method.py`, and `configs/VisualGenome/PE_NET.py`.

This local adapter is not official-code parity. The official implementation
uses 2048-dim prototype scoring, 300d object/predicate GloVe embeddings,
`W_pred`, `project_head`, normalized cosine similarity with `logit_scale`, and
prototype losses. The OpenSGG adapter uses a simplified fusion/classifier path.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. The official README lists checkpoint links, but no trusted local
PENet checkpoint, pretrained detector checkpoint, or PENET-format VG inputs are
available in this workspace.

## 6. Hidden-Positive Metrics

Random-init fallback JSONL export passed validation on 2 rows. R@1/R@5/R@10
and mR@1/mR@5/mR@10 were all 0.0.

## 7. Deviations

No official checkpoint; no official pretrained detector; missing official
PENET-format VG inputs; local implementation is not official
`PrototypeEmbeddingNetwork` parity; official `rel_nms` evaluator behavior is
not aligned; random-init fallback; tiny CPU slice; local hidden-eval tooling.

## 8. 当前可信 Claim

PENet can be instantiated and tested under `conda hsg`; its outputs work with
the standard metric and hidden/GT-aligned JSONL paths. The official PENet repo
and commit are pinned, and missing reproduction inputs are recorded in
`penet_official_input_check.json`.

## 9. 不能声称的内容

Do not claim paper-level PENet reproduction numbers, official checkpoint-backed
performance, or official implementation parity.

## 10. 对本项目的作用

PENet is available as a documented P1 baseline with an implementation audit and
no-training fallback smoke evidence.

## 11. 下一步建议

Provide official PENET-format VG inputs, the official pretrained detector, and
trusted PENet PredCls/SGCls/SGDet checkpoints. Then run official PENet
evaluation first, compare evaluator semantics including `rel_nms`, and only
then decide whether OpenSGG should import/adapt the official predictor.
