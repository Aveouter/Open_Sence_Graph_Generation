# FREQ Final Reproduction Report

**Superseded by stricter alignment audit:** this report documents pipeline
evidence only. FREQ is currently `DEFERRED_NOT_REPRODUCED` until original
paper/repository protocol alignment is verified.

## 1. 方法简介

FREQ is the pair-frequency baseline for PredCls scene graph generation. It ranks
predicates using `P(predicate | subject class, object class)` without visual
features.

## 2. 官方来源

The baseline follows the FREQ prior reported with Neural Motifs:
`Scene Graph Parsing with Global Context`, Zellers et al., CVPR 2018. The local
implementation reuses OpenSGG's existing `PairFrequencyBias`, which mirrors the
frequency-prior branch used by Motifs/TDE-style relation heads.

The local source audit now pins the SGB reference implementation at
`/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
commit `ceb71fa88461c2a97a6258a80f47669d89207296`. That implementation reads
`statistics['pred_dist']` into `FrequencyBias.obj_baseline`. OpenSGG currently
rebuilds a prior from local `train.json`/`rel.json`, so numerical prior parity
is still unverified.

An input check for exporting SGB `pred_dist` is blocked because the workspace is
missing `VG-SGG-with-attri.h5`; see `sgb_freq_input_check.json`.

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

See `09_deviations.md` and `11_evidence_gate_audit.md`. Main deviations are no
full VG paper alignment, missing SGB-format roidb input for official
`pred_dist` export, no numerical comparison against official SGB `pred_dist`,
local OpenSGG-compatible prior construction, and CPU-only validation due to CUDA
driver mismatch.

## 8. 当前可信 Claim

OpenSGG has a registered, smoke-tested FREQ-compatible pipeline whose outputs
work with the standard PredCls metric path and the local GT-aligned hidden/JSONL
predicate recall path. Its prior API is structurally similar to SGB
`FrequencyBias`.

## 9. 不能声称的内容

Do not claim paper-level FREQ reproduction numbers, official prior-table parity,
full VG multi-seed results, or CUDA validation from this phase.

## 10. 对本项目的作用

FREQ provides a simple pair-prior control baseline and a reusable no-checkpoint
export path for prior-based diagnostics.

## 11. 下一步建议

Before reopening FREQ, implement the prior-table comparison described in
`11_evidence_gate_audit.md`, then run benchmark-scale PredCls evaluation with
unchanged evaluator semantics.
