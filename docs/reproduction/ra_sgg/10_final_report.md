# RA-SGG Final Reproduction Report

**Superseded by stricter alignment audit:** this report documents a minimal
OpenSGG adapter and random-init/no-memory pipeline evidence only. RA-SGG is
currently `DEFERRED_NOT_REPRODUCED` until official ReTAG checkpoint,
pretrained PE-Net, memory bank, and protocol parity are established.

## 1. 方法简介

RA-SGG/ReTAG is a retrieval-augmented scene graph generation framework using a
PENet-style multi-prototype predictor and a memory bank of relation embeddings.

## 2. 官方来源

Official repository: https://github.com/KanghoonYoon/torch-rasgg. Local audited
commit: `e8be01b9fde5c694243606e73931a4c8a8b1bf41`.

Official PredCls script uses `ReTAGPENet`, pretrained PE-Net, memory-bank
feature files, `NUM_RETRIEVALS=20`, `MEMORY_SIZE=8`, reliable selection, and
mixup.

## 3. OpenSGG 集成状态

OpenSGG includes a minimal adapter after the RA-SGG integration PR:

- `src/models/ra_sgg.py`
- `src/methods/ra_sgg_method.py`
- `configs/VisualGenome/RA_SGG.py`

This adapter is not official ReTAG parity. Official ReTAG retrieves top-k
relation embeddings from a feature bank and uses retrieved predicate
distributions, frequency reweighting, BG correction, reliable selection,
mixup, and prototype losses.

## 4. Standard SGG Metrics

Random-init/no-memory fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. No official ReTAG checkpoint, pretrained PE-Net checkpoint, memory
bank, or official VG inputs were available.

## 6. Hidden-Positive Metrics

Random-init/no-memory fallback JSONL export passed validation on 2 rows. All
R@1/R@5/R@10 and mR@1/mR@5/mR@10 values were 0.0.

## 7. Deviations

No checkpoint; no pretrained PE-Net; no memory bank; missing official VG
inputs; minimal adapter rather than strict official ReTAG parity; tiny CPU
slice; local hidden-eval tooling. The previous PR was closed as deferred
because this is not original-aligned reproduction evidence.

## 8. 当前可信 Claim

RA-SGG is now a registered OpenSGG method with a tested no-training adapter.
Its outputs work with standard metric and hidden/GT-aligned JSONL paths.

The official repo/commit and missing input classes are recorded in
`rasgg_official_input_check.json`.

## 9. 不能声称的内容

Do not claim official RA-SGG reproduction, memory-bank retrieval behavior,
checkpoint-backed performance, or paper-level results.

## 10. 对本项目的作用

RA-SGG is represented as a documented P1 baseline with source-audited,
OpenSGG-compatible minimal test coverage.

## 11. 下一步建议

Provide official VG inputs, pretrained PE-Net checkpoints, ReTAG checkpoints,
and memory-bank feature files. Then run official evaluation first and replace
or extend the adapter only after retrieval/evaluator parity is established.
