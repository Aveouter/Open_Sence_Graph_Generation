# TDE Final Reproduction Report

## 1. 方法简介

TDE, Total Direct Effect, is the causal debiasing inference method from
`Unbiased Scene Graph Generation from Biased Training`. It subtracts causal
counterfactual effects from biased relation predictions, usually on top of a
Motifs relation head.

## 2. 官方来源

Primary source: Tang et al., CVPR 2020, `Unbiased Scene Graph Generation from
Biased Training`. Official repository:
https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.

## 3. OpenSGG 集成状态

OpenSGG contains an official-style TDE implementation:

- `src/models/motifs.py:TDEModel`
- `src/methods/tde_method.py:TDE_Method`
- `configs/VisualGenome/TDE.py`

Official-source mapping is documented in `05_mapping.md`.

## 4. Standard SGG Metrics

Random-init fallback PredCls metric slice ran through the standard OpenSGG
metric path. On a 2-image cap / first-batch CPU test:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

These validate metric-path compatibility only.

## 5. Paper-Number Alignment

Not claimed. No verified official TDE checkpoint was available, and official
runtime parity was not established in `conda hsg`.

## 6. Hidden-Positive Metrics

Random-init fallback relation JSONL export passed validation on 2 GT relations.
GT-aligned predicate recall:

- R@1/R@5/R@10: 0.0 / 0.0 / 0.0
- mR@1/mR@5/mR@10: 0.0 / 0.0 / 0.0

## 7. Deviations

See `09_deviations.md`. Main deviations are missing verified official
checkpoint, random-init fallback, tiny CPU slice, and no official evaluator
parity.

## 8. 当前可信 Claim

OpenSGG TDE can be instantiated and tested under `conda hsg`; its outputs work
with the standard PredCls metric path and local hidden/GT-aligned relation JSONL
evaluation. Official TDE logic has been mapped to OpenSGG source locations.

## 9. 不能声称的内容

Do not claim paper-level TDE reproduction, checkpoint-backed TDE performance,
official evaluator equivalence, or full VG results.

## 10. 对本项目的作用

TDE is now documented as a P0 causal-debiasing baseline with clear source
mapping and no-training fallback test evidence.

## 11. 下一步建议

Continue to P0 VCTree. If an official VCTree checkpoint is absent, follow the
same explicit no-checkpoint fallback procedure and record deviations.
