# Motifs Final Reproduction Report

## 1. 方法简介

Neural Motifs is a two-stage scene graph relation baseline that uses object
context, edge context, visual relation features, and frequency bias to predict
predicates from detected or ground-truth objects.

## 2. 官方来源

Primary source: `Neural Motifs: Scene Graph Parsing with Global Context`,
Zellers et al., CVPR 2018. Project page: https://rowanzellers.com/neuralmotifs/.

## 3. OpenSGG 集成状态

OpenSGG already contains a Motifs implementation:

- `src/models/motifs.py`
- `src/methods/motifs_method.py`
- `configs/VisualGenome/Motifs.py`

The method is registered as `motifs` and can be instantiated/tested under
`conda hsg`.

## 4. Standard SGG Metrics

Checkpoint-backed PredCls metric slice ran through `src.core.metrics.metric`.
On a 2-image cap / first-batch CPU test:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

These are metric-path validation numbers, not benchmark claims.

## 5. Paper-Number Alignment

Not claimed. Full VG evaluation and strict official checkpoint parity were not
established.

## 6. Hidden-Positive Metrics

Checkpoint-backed relation JSONL export passed validation on 2 GT relations.
GT-aligned predicate recall:

- R@1/R@5/R@10: 0.0 / 1.0 / 1.0
- mR@1/mR@5/mR@10: 0.0 / 1.0 / 1.0

## 7. Deviations

See `09_deviations.md`. Main deviations are partial checkpoint mapping,
tiny-slice CPU evaluation, and no paper-scale benchmark claim.

## 8. 当前可信 Claim

OpenSGG Motifs can be instantiated and tested under `conda hsg`; a local Motifs
checkpoint can be partially adapted and used to drive standard PredCls metric
plumbing plus hidden/GT-aligned JSONL predicate-recall evaluation.

## 9. 不能声称的内容

Do not claim paper-level Motifs reproduction numbers, full VG results, strict
official checkpoint parity, or CUDA validation from this phase.

## 10. 对本项目的作用

Motifs is now documented as a checkpoint-backed P0 baseline with working local
test/export evidence and clear checkpoint deviations.

## 11. 下一步建议

Continue to P0 TDE. Prefer official/local TDE checkpoint evidence; if no strict
TDE checkpoint is available, validate the OpenSGG TDE logic against official
causal predictor structure and record checkpoint deviations.
