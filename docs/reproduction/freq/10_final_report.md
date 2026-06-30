# FREQ Final Reproduction Report

## 1. 方法简介

FREQ is the pair-frequency baseline for PredCls scene graph generation. It ranks
predicates using `P(predicate | subject class, object class)` without visual
features.

## 2. 官方来源

The baseline follows the FREQ prior reported with Neural Motifs:
`Scene Graph Parsing with Global Context`, Zellers et al., CVPR 2018. The local
implementation reuses OpenSGG's existing `PairFrequencyBias`, which mirrors the
frequency-prior branch used by Motifs/TDE-style relation heads.

## 3. OpenSGG 集成状态

Integrated as:

- `src/methods/freq_method.py`
- `configs/VisualGenome/FREQ.py`
- `src/methods/__init__.py` registry entry `freq`
- CLI/config alias support in `utils/parser.py` and `train.py`
- FREQ no-checkpoint export support in `tools/analysis/export_relation_predictions.py`

## 4. Standard SGG Metrics

The standard PredCls metric path accepts FREQ outputs. On a tiny 2-image local
slice:

- R@20/R@50/R@100: 0.0 / 0.0 / 0.0
- mR@20/mR@50/mR@100: 0.0 / 0.0 / 0.0

These are plumbing-validation numbers only.

## 5. Paper-Number Alignment

Not claimed. Full VG benchmark evaluation was not run.

## 6. Hidden-Positive Metrics

The relation JSONL export path is adapted for FREQ without checkpoints. On a
1-batch tiny slice:

- Rows: 2 total, 2 matched.
- R@1/R@5/R@10: 0.0 / 1.0 / 1.0.
- mR@1/mR@5/mR@10: 0.0 / 1.0 / 1.0.

## 7. Deviations

See `09_deviations.md`. Main deviations are no full VG paper alignment, local
OpenSGG-compatible prior implementation, and CPU-only validation due to CUDA
driver mismatch.

## 8. 当前可信 Claim

OpenSGG now has a registered, smoke-tested FREQ baseline whose outputs work with
the standard PredCls metric path and the local GT-aligned hidden/JSONL predicate
recall path.

## 9. 不能声称的内容

Do not claim paper-level FREQ reproduction numbers, full VG multi-seed results,
or CUDA validation from this phase.

## 10. 对本项目的作用

FREQ provides a simple pair-prior control baseline and a reusable no-checkpoint
export path for prior-based diagnostics.

## 11. 下一步建议

Proceed to P0 Motifs with the same phase discipline: source audit, inventory,
smoke, metric slice, hidden export, deviations, and final claims.
