# TDE Reproduction: Official Code Analysis

Last updated: 2026-06-22.

Official repository inspected:
`KaihuaTang/Scene-Graph-Benchmark.pytorch@ceb71fa88461c2a97a6258a80f47669d89207296`.

## High-Level Architecture

The official implementation treats TDE as a causal inference mode inside the
relation head predictor, not as a standalone dataset, trainer, or evaluator.

Core components:

| Component | Official file/class | Role |
|---|---|---|
| Backbone/detector | `GeneralizedRCNN` with `R-101-FPN` | Frozen detector backbone and RoI features |
| Relation head predictor | `roi_relation_predictors.py:CausalAnalysisPredictor` | Owns context, visual, frequency branches and causal effect selection |
| Context encoder | `model_motifs.py:LSTMContext` when `CONTEXT_LAYER=motifs` | MOTIFS object/edge context with untreated moving averages |
| Frequency bias | `model_motifs.py:FrequencyBias` | Subject-object predicate prior branch |
| Visual branch | `CausalAnalysisPredictor.vis_compress` | Union feature predicate logits |
| Context branch | `CausalAnalysisPredictor.ctx_compress` | Context feature predicate logits |
| Fusion | `CausalAnalysisPredictor.calculate_logits` | `sum` or `gate` branch combination |
| Causal effect switch | `MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE` | `none`, `TDE`, `NIE`, or `TE` at inference |

## Key Config Controls

Official config path:

- `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`

Required causal MOTIFS-SUM settings:

```yaml
MODEL.ROI_RELATION_HEAD.PREDICTOR: CausalAnalysisPredictor
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE: none  # training
MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE: sum
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER: motifs
MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION: True
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS: True
```

During evaluation, only `EFFECT_TYPE` changes from `none` to `TDE`.

## CausalAnalysisPredictor Structure

Official file:

- `/tmp/tde_official/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`

Relevant lines in inspected checkout:

- Class declaration: lines 433-434.
- Config reads: lines 438-445.
- Context layer selection: lines 456-461.
- Branch heads:
  - `ctx_compress`: lines 472 or 478.
  - `vis_compress`: line 479.
  - `freq_bias`: line 495.
- Untreated buffers:
  - `untreated_spt`: line 513.
  - `untreated_conv_spt`: line 514.
  - `avg_post_ctx`: line 515.
  - `untreated_feat`: line 516.
- Pair feature construction: lines 519-556.
- Main forward: lines 560-641.
- Effect equations: lines 630-635.
- Fusion logits: lines 649-676.

## Feature Flow

The official predictor builds three predicate-logit branches for each
subject-object pair:

1. Context branch:
   - `context_layer(...)` produces object distributions, object predictions,
     edge context, and optional VCTree binary predictions.
   - `post_emb` splits edge context into subject and object edge features.
   - Pair context is subject/object concatenation for MOTIFS/VCTree or
     subtraction for VTransE.
   - `post_cat` maps pair context to the 4096-dimensional pooling space.
   - `ctx_compress(ctx_rep)` predicts predicate logits from context.

2. Visual branch:
   - `union_features` are relation union RoI features.
   - `vis_compress(union_features)` predicts predicate logits from visual
     union features.
   - If `SPATIAL_FOR_VISION=True`, pair-box spatial embedding modulates context
     representation before logit calculation.

3. Frequency branch:
   - `FrequencyBias.index_with_labels(pair_pred)` in normal label mode.
   - `FrequencyBias.index_with_probability(pair_obj_probs)` when using object
     label distributions for causal effect analysis.

For `FUSION_TYPE=sum`, the official logit fusion is:

```python
union_dists = vis_dists + ctx_dists + frq_dists
```

For `FUSION_TYPE=gate`, the official code applies a context-gated form:

```python
union_dists = ctx_dists * sigmoid(vis_dists + frq_dists + ctx_gate_dists)
```

The downloadable official checkpoints documented in the README use MOTIFS-SUM.

## TDE Calculation Location

Official TDE is computed in:

- File: `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
- Class: `CausalAnalysisPredictor`
- Function: `forward`
- Lines: 630-631 in the inspected checkout.

Official code:

```python
if self.effect_type == 'TDE':
    rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) \
        - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
```

Interpretation:

- Factual term: visual union features + factual context + factual object-label
  distribution.
- Counterfactual term: same visual union features + averaged/untreated context
  + same object-label distribution.
- The frequency branch is conditioned on the same object-label distribution in
  both terms, so the subtraction targets the context pathway for TDE.

## Other Effect Types

Official code also supports:

```python
NIE = calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) \
    - calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)

TE = calculate_logits(union_features, post_ctx_rep, pair_obj_probs) \
    - calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
```

These are useful diagnostic baselines, but the requested reproduction target is
TDE.

## Untreated Feature Buffers

Official TDE depends on moving averages accumulated during biased training.
These buffers are part of the checkpoint state and are required for faithful
TDE inference.

In `CausalAnalysisPredictor`:

- `untreated_spt`
- `untreated_conv_spt`
- `avg_post_ctx`
- `untreated_feat`

In `LSTMContext` for MOTIFS:

- `untreated_dcd_feat`
- `untreated_obj_feat`
- `untreated_edg_feat`

The moving average update uses:

```python
holder = holder * (1 - 0.0005) + 0.0005 * input.mean(0)
```

The paper states that mean training features are used for the wiped-out input;
zero vectors are mentioned only as an alternative with worse behavior. A strict
reproduction must therefore load and use official moving-average buffers from
the checkpoint, not recompute ad hoc zero or embedding averages unless the
deviation is explicitly recorded.

## Training Behavior

Training uses ordinary biased training with `EFFECT_TYPE=none`, but
`EFFECT_ANALYSIS=True` activates the additional branch losses and moving-average
buffer updates.

Training-only auxiliary losses in `CausalAnalysisPredictor.forward`:

- `auxiliary_ctx`: cross entropy on the context branch.
- `auxiliary_vis`: cross entropy on the visual branch.
- `auxiliary_frq`: cross entropy on the frequency branch.
- `binary_loss`: only when the selected context layer provides VCTree binary
  relation predictions.

This matters because a checkpoint trained without these branch losses and
moving-average buffers is not an official TDE checkpoint, even if inference
performs a logit subtraction.

## Current OpenSGG TDE Status

Existing local files:

- `configs/VisualGenome/TDE.py`
- `src/methods/tde_method.py`
- `src/models/motifs.py:TDEModel`

Local source evidence after the 2026-06-22 migration pass:

- `src/models/motifs.py:340`: `PairFrequencyBias.index_with_probability`
  mirrors the official probability-mode frequency branch.
- `src/models/motifs.py:627`: `SGBLSTMContext.forward` accepts
  `ctx_average`.
- `src/models/motifs.py:657`: decoder input can use
  `untreated_dcd_feat`.
- `src/models/motifs.py:670`: edge input can use `untreated_edg_feat`.
- `src/models/motifs.py:1123`: `TDEModel` is now an official-style
  `CausalAnalysisPredictor` port.
- `src/models/motifs.py:1164`: explicit `ctx_compress` branch.
- `src/models/motifs.py:1165`: explicit `vis_compress` branch.
- `src/models/motifs.py:1178`: predictor untreated buffers are registered.
- `src/models/motifs.py:1275`: frequency branch uses
  `index_with_probability` for causal effect calculations.
- `src/models/motifs.py:1284`: `calculate_logits` implements official
  branch fusion.
- `src/models/motifs.py:1415`: predictor untreated averages are updated
  during training.
- `src/models/motifs.py:1427`: evaluation counterfactual path calls
  `ctx_average=True`.
- `src/methods/tde_method.py:7`: `TDECriterion` adds official auxiliary
  branch losses.
- `configs/VisualGenome/TDE.py:48`: config exposes official
  `tde_effect_type`.

Current local behavior observed from source:

- `TDE_Method` still subclasses `Motifs_Method` to preserve OpenSGG's method
  interface.
- `build_tde` returns `TDEModel`.
- `TDEModel` owns separate visual, context, and frequency branches.
- `TDEModel.calculate_logits` supports official `sum` and `gate` fusion.
- `TDEModel.forward` supports `effect_type` values `none`, `TDE`, `NIE`, and
  `TE`.
- Training updates untreated moving-average buffers and emits
  `auxiliary_ctx`, `auxiliary_vis`, and `auxiliary_frq`.
- Evaluation computes TDE as
  `calculate_logits(visual, factual_context, pair_obj_probs) -
  calculate_logits(visual, average_context, pair_obj_probs)`.

Remaining alignment risks:

1. Official checkpoint key compatibility is not yet proven because the official
   checkpoint is not yet available in this workspace.
2. The current implementation preserves OpenSGG inputs and evaluator; evaluator
   equivalence against the official repo remains unverified.
3. The current runtime is `hsg` rather than the original official environment;
   this follows the current execution constraint but should remain documented
   as an environment deviation for official-paper reproduction.

The `hsg` environment can run current OpenSGG code, but its modern dependency
stack does not change this conclusion:

```text
Python 3.10.8
Torch 2.5.1+cu121
torchvision 0.20.1+cu121
GPU: NVIDIA A100-PCIE-40GB
```

## Implementation Requirement for Migration

The migration target should be shaped like the official predictor:

- Add a current-framework TDE module that owns three branch logits:
  `vis`, `ctx`, and `freq`.
- Preserve official `sum` fusion first; add `gate` only after SUM is validated.
- Preserve moving-average untreated buffers and load them from official
  checkpoints or deterministic converted checkpoints.
- Preserve effect modes: `none`, `TDE`, `NIE`, `TE`.
- Preserve protocol switches for PredCls, SGCls, SGDet.
- Do not alter dataset, trainer, or evaluator interfaces.
- Any unavoidable deviation must be recorded in
  [`06_deviations.md`](06_deviations.md).
