# RA-SGG Final Reproduction Report

## 1. 方法简介

RA-SGG/ReTAG is a retrieval-augmented scene graph generation framework using a
PENet-style multi-prototype predictor and a memory bank of relation embeddings.

## 2. 官方来源

Official repository: https://github.com/KanghoonYoon/torch-rasgg. Local audited
commit: `e8be01b9fde5c694243606e73931a4c8a8b1bf41`.

## 3. OpenSGG 集成状态

OpenSGG now includes a minimal adapter:

- `src/models/ra_sgg.py`
- `src/methods/ra_sgg_method.py`
- `configs/VisualGenome/RA_SGG.py`

## 4. Standard SGG Metrics

Random-init/no-memory fallback PredCls metric slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

## 5. Paper-Number Alignment

Not claimed. No official checkpoint or memory bank was available.

## 6. Hidden-Positive Metrics

Random-init/no-memory fallback JSONL export passed validation on 2 rows. All
R@1/R@5/R@10 and mR@1/mR@5/mR@10 values were 0.0.

## 7. Deviations

No checkpoint; no memory bank; minimal adapter rather than strict official
ReTAG parity; tiny CPU slice; local hidden-eval tooling; GitHub PR blocked.

## 8. 当前可信 Claim

RA-SGG is now a registered OpenSGG method with a tested no-training adapter.
Its outputs work with standard metric and hidden/GT-aligned JSONL paths.

## 9. 不能声称的内容

Do not claim official RA-SGG reproduction, memory-bank retrieval behavior,
checkpoint-backed performance, or paper-level results.

## 10. 对本项目的作用

RA-SGG is represented as a documented P1 baseline with source-audited,
OpenSGG-compatible minimal test coverage.

## 11. 下一步建议

Proceed to P2 RelTR and EGTR. If GitHub transport is restored, split method
changes into independent PRs.
