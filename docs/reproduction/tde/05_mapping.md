# TDE Reproduction: Official to Current Framework Mapping

Last updated: 2026-06-22.

This is a preliminary mapping based on source inspection. It is not a claim of
checkpoint compatibility.

| Official implementation | Current OpenSGG location | Status |
|---|---|---|
| `configs/e2e_relation_X_101_32_8_FPN_1x.yaml` | `configs/VisualGenome/TDE.py` | implemented as OpenSGG config; exact official CLI options are not one-to-one |
| `CausalAnalysisPredictor` | `src/models/motifs.py:TDEModel` | implemented; checkpoint compatibility still unverified |
| `LSTMContext` for `CONTEXT_LAYER=motifs` | `src/models/motifs.py:SGBLSTMContext` | implemented with `ctx_average` and untreated buffers |
| `FrequencyBias` | `src/models/motifs.py:PairFrequencyBias` | implemented with label and probability lookup |
| `vis_compress` branch | `src/models/motifs.py:TDEModel.vis_compress` | implemented |
| `ctx_compress` branch | `src/models/motifs.py:TDEModel.ctx_compress` | implemented |
| `calculate_logits(vis, ctx, frq)` | `src/models/motifs.py:TDEModel.calculate_logits` | implemented for `sum` and `gate` |
| `EFFECT_TYPE none/TDE/NIE/TE` | `configs/VisualGenome/TDE.py:tde_effect_type` | implemented |
| Untreated buffers in predictor | `TDEModel.untreated_spt`, `avg_post_ctx`, `untreated_feat` | implemented; load from official checkpoint still unverified |
| Untreated buffers in `LSTMContext` | `SGBLSTMContext.untreated_dcd_feat`, `untreated_obj_feat`, `untreated_edg_feat` | implemented |
| Branch auxiliary losses | `src/methods/tde_method.py:TDECriterion` | implemented |
| Official evaluator | current `train.py --test` metrics | unverified alignment |

## Current Local Source Anchors

| Local file anchor | Observation | Alignment implication |
|---|---|---|
| `src/models/motifs.py:340` | `index_with_probability` exists | official frequency probability path implemented |
| `src/models/motifs.py:1123` | `TDEModel` defines official-style causal predictor | implementation no longer uses mean-visual wrapper |
| `src/models/motifs.py:1164` | `ctx_compress` exists | context branch implemented |
| `src/models/motifs.py:1165` | `vis_compress` exists | visual branch implemented |
| `src/models/motifs.py:1178` | predictor untreated buffers are registered | official buffer names present |
| `src/models/motifs.py:1284` | `calculate_logits` exists | official fusion point implemented |
| `src/methods/tde_method.py:7` | `TDECriterion` exists | auxiliary losses are integrated |
| `configs/VisualGenome/TDE.py:48` | `tde_effect_type` exists | official effect modes exposed |

## Migration Direction

Strict migration now has a current-framework TDE module that mirrors the core
`CausalAnalysisPredictor` logic while preserving OpenSGG's public method
interface.

Minimum implementation units:

1. Download official checkpoints and inspect exact key names.
2. Validate that `remap_external_state_dict` maps official TDE keys into
   `TDEModel` without silent skips for required branches/buffers.
3. Run one-batch current-framework output inspection after checkpoint load.
4. Compare current-framework metrics with official-repo metrics once official
   checkpoint evaluation is available.

Initial implementation should be checkpoint-load strict: fail if an official
checkpoint key is missing or unexpectedly ignored, then add explicit conversion
rules. Silent `strict=False` loading is not acceptable for the final
reproduction.

## Do Not Modify

Per objective constraints, migration should avoid changing:

- Dataset interfaces
- Trainer interfaces
- Evaluator interfaces

If an interface mismatch prevents official checkpoint loading, document it in
`deviations.md` before adding an adapter.
