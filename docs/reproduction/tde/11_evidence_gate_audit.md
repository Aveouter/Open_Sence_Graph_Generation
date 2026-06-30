# TDE Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`.

This audit records the evidence required to reopen TDE as a checkpoint-backed
reproduction PR. It is not a reproduction result.

## Official Source Anchor

- Paper: `Unbiased Scene Graph Generation from Biased Training`, Tang et al.,
  CVPR 2020.
- Official repository: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
- Local checkout:
  `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
- Local commit: `ceb71fa88461c2a97a6258a80f47669d89207296`
- Primary config:
  `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- Official predictor:
  `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py:CausalAnalysisPredictor`

## Official TDE Behavior

The official config trains causal MOTIFS-SUM with:

- `PREDICTOR: CausalAnalysisPredictor`
- `CAUSAL.EFFECT_TYPE: none`
- `CAUSAL.FUSION_TYPE: sum`
- `CAUSAL.CONTEXT_LAYER: motifs`
- `CAUSAL.SPATIAL_FOR_VISION: True`
- `CAUSAL.EFFECT_ANALYSIS: True`

The official inference switches `CAUSAL.EFFECT_TYPE` to `TDE`. In source,
`CausalAnalysisPredictor.forward` computes:

- standard logits: `calculate_logits(union_features, post_ctx_rep, pair_obj_probs)`
- TDE logits:
  `calculate_logits(union_features, post_ctx_rep, pair_obj_probs) -
  calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)`

The average context and untreated buffers come from training-time moving
averages. Therefore random initialization or a Motifs-only checkpoint cannot
support official TDE inference.

## OpenSGG Current Behavior

OpenSGG has a TDE-like implementation:

- `src/models/motifs.py:TDEModel`
- `src/methods/tde_method.py:TDE_Method`
- `configs/VisualGenome/TDE.py`

It exposes effect modes `none`, `TDE`, `NIE`, and `TE`, and includes visual,
context, and frequency branches. However, checkpoint compatibility and official
runtime/evaluator parity are not established.

Important config caveat:

- Official training uses `EFFECT_TYPE none` and evaluation switches to `TDE`.
- OpenSGG `configs/VisualGenome/TDE.py` defaults to `tde_effect_type = 'TDE'`
  because the local config is evaluation-oriented.
- This is acceptable for an eval config only after official checkpoint and
  moving-average buffers are verified.

## Gate Status

| Gate | Status | Evidence | Gap |
|---|---|---|---|
| TDE-1 | Partial | SGB source and commit are pinned. | Need official runtime env or a documented official-code evaluation attempt. |
| TDE-2 | Failing | Official checkpoint URLs are documented in `04_checkpoint.md`. | No official checkpoint file exists locally for PredCls, SGCls, or SGDet. |
| TDE-3 | Partial | OpenSGG implements TDE-like causal subtraction and effect modes. | Need checkpoint key mapping and numerical/logit-level parity with `CausalAnalysisPredictor`. |
| TDE-4 | Failing | Config mapping is documented in `05_mapping.md`. | SGB-format `VG-SGG-with-attri.h5` is missing; evaluator parity cannot be checked. |
| TDE-5 | Failing | Random-init tiny smoke exists. | No benchmark-scale official checkpoint result exists. |

## Current Input Check

Command:

```bash
python tools/reproduction/check_tde_official_inputs.py \
  --output docs/reproduction/tde/tde_official_input_check.json
```

Observed result:

- Status: `BLOCKED`
- Blockers:
  - `missing_sgb_vg_inputs`
  - `missing_official_tde_checkpoints`
- Missing SGB input:
  - `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5`
- Missing checkpoint protocols:
  - `predcls`
  - `sgcls`
  - `sgdet`

## Required Verification Before Reopening

1. Provide SGB-format VG input:

```text
/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5
```

2. Download official TDE checkpoints into:

```text
outputs/pretrained/tde_official/predcls/
outputs/pretrained/tde_official/sgcls/
outputs/pretrained/tde_official/sgdet/
```

3. Record size, source URL, and SHA256 for each checkpoint.

4. Inspect checkpoint keys and verify `CausalAnalysisPredictor` buffers:

- `avg_post_ctx`
- `untreated_feat`
- `untreated_spt`
- context-layer untreated buffers
- `ctx_compress`
- `vis_compress`
- `freq_bias.obj_baseline.weight`

5. Run the official repository evaluation command for each protocol, with
training/eval effect type semantics preserved:

- training checkpoint: `EFFECT_TYPE none`
- evaluation: `EFFECT_TYPE TDE`

6. Only after official metrics are recorded should OpenSGG checkpoint mapping be
attempted. Mapping must fail loudly if required causal keys or untreated buffers
are missing.

## Current Trusted Claim

OpenSGG has a TDE-compatible implementation audit and random-init pipeline
smoke. It is not checkpoint-backed and not official-protocol aligned.

## Forbidden Claims

- Do not claim TDE reproduction.
- Do not claim official checkpoint-backed performance.
- Do not use a Motifs checkpoint as a TDE checkpoint.
- Do not use random-init TDE smoke as reproduction evidence.
- Do not claim evaluator parity before official SGB evaluation is run.
