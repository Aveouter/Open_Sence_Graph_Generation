# TDE Reproduction: Deviations Log

Last updated: 2026-06-22.

This file records all known differences from the official TDE implementation.
Items start as open deviations and can be closed only with code, checkpoint, and
metric evidence.

## Code-Resolved, Validation-Open Items

These items were open before the 2026-06-22 implementation pass. They are now
resolved at the source-code level but still require official checkpoint and
metric validation before they can be treated as fully closed.

### R1: Official-style causal predictor

- Implemented: `src/models/motifs.py:TDEModel` now owns explicit visual,
  context, and frequency branches.
- Source evidence: `ctx_compress`, `vis_compress`,
  `PairFrequencyBias.index_with_probability`, and `calculate_logits`.
- Remaining validation: load official checkpoint and confirm required TDE keys
  map correctly.

### R2: Untreated feature handling

- Implemented: predictor and context untreated buffers are registered and
  updated during training; evaluation supports `ctx_average=True`.
- Source evidence: `untreated_spt`, `avg_post_ctx`, `untreated_feat`,
  `untreated_dcd_feat`, `untreated_obj_feat`, and `untreated_edg_feat`.
- Remaining validation: verify official checkpoint buffers load with expected
  shapes and nonzero learned averages.

### R3: Branch auxiliary losses

- Implemented: `TDECriterion` adds `auxiliary_ctx`, `auxiliary_vis`, and
  `auxiliary_frq` when relation annotations are available.
- Remaining validation: train-time reproduction is not the current priority,
  but the loss wiring now matches the official branch-loss structure.

### R4: Effect modes

- Implemented: config exposes `tde_effect_type` with `none`, `TDE`, `NIE`, and
  `TE`; `TDEModel.forward` implements the corresponding effect equations.
- Remaining validation: run each mode after official checkpoint loading.

## Open Deviations

### D5: Checkpoint compatibility is unverified

- Official checkpoints target `maskrcnn_benchmark` module/key layout.
- Current framework has different module names and likely different tensor key
  layout.
- Risk: Loading may silently skip or mis-map parameters if not audited.
- Required resolution: inspect official checkpoint keys, define exact mapping,
  and fail closed on missing/unexpected keys.

### D6: Evaluator equivalence is unverified

- Official metrics are produced by Scene Graph Benchmark's evaluator.
- Current metrics are produced by OpenSGG evaluation code.
- Risk: R/mR differences may come from evaluator protocol, not model behavior.
- Required resolution: first reproduce official metrics in official repo; then
  run current evaluator and compare with identical predictions/checkpoint.

## Closed Deviations

None yet.
