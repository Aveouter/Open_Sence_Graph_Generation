# VCTree Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`.

This audit records the evidence required before VCTree can be reopened as a
checkpoint-backed reproduction PR. It is not a reproduction result.

## Official Source Anchor

- Paper: `Learning to Compose Dynamic Tree Structures for Visual Contexts`,
  Tang et al., CVPR 2019.
- Original paper code: https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation
- SGB reference used in this workspace:
  https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
- Local SGB checkout:
  `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
- Local SGB commit: `ceb71fa88461c2a97a6258a80f47669d89207296`

## Official/SGB Reference Behavior

SGB VCTree source anchors:

- `configs/e2e_relation_R_101_FPN_1x.yaml`
  - `PREDICTOR: "VCTreePredictor"`
  - `CONTEXT_HIDDEN_DIM: 512`
  - `POOLING_ALL_LEVELS: True`
- `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
  - `VCTreePredictor`
  - uses `VCTreeLSTMContext`
  - combines context logits with `FrequencyBias`
- `maskrcnn_benchmark/modeling/roi_heads/relation_head/model_vctree.py`
  - `VCTreeLSTMContext`
  - builds pair scores with `vctree_score_net`
  - calls `generate_forest` and `arbForest_to_biForest`
  - uses `MultiLayer_BTreeLSTM`, `DecoderTreeLSTM`, and edge TreeLSTM
- `maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_vctree.py`
  - official tree construction utilities.

The SGB README notes that most SGG checkpoints are not uploaded and that users
normally train their own SGG models. Under the current no-training,
checkpoint-first standard, this means VCTree must stay deferred unless a trusted
checkpoint is supplied.

## OpenSGG Current Behavior

OpenSGG has a VCTree-compatible pipeline:

- `src/models/vctree.py`
- `src/methods/vctree_method.py`
- `configs/VisualGenome/VCTree.py`

However, the current implementation is not the SGB `VCTreePredictor`:

- it uses a local `TreeConstructor` with greedy feature/box affinity;
- it does not use SGB `generate_forest` / `arbForest_to_biForest`;
- it does not use SGB `VCTreeLSTMContext` with `vctree_score_net`;
- it uses a simplified local `FrequencyBias`, not SGB `statistics['pred_dist']`
  backed `FrequencyBias`;
- checkpoint key compatibility with SGB VCTree is unverified and likely
  incompatible.

Therefore the current implementation can only support `implementation audit` or
`pipeline smoke`, not paper/official-code reproduction.

## Gate Status

| Gate | Status | Evidence | Gap |
|---|---|---|---|
| VCTREE-1 | Partial | SGB source and original paper repo are identified. | Need decide whether target is original VCTree repo or SGB reimplementation; current audit uses SGB. |
| VCTREE-2 | Failing | No trusted checkpoint candidate exists under `outputs/pretrained/vctree_official`. | Official/SGB README does not provide uploaded VCTree SGG checkpoints; training is out of scope for checkpoint-backed reproduction. |
| VCTREE-3 | Failing | OpenSGG local VCTree runs as pipeline smoke. | Local implementation does not match SGB `VCTreePredictor`/`VCTreeLSTMContext`. |
| VCTREE-4 | Failing | SGB evaluator source exists. | SGB-format `VG-SGG-with-attri.h5` is missing, so official evaluator parity cannot be run. |
| VCTREE-5 | Failing | Random-init tiny smoke exists. | No benchmark-scale checkpoint result exists. |

## Current Input Check

Command:

```bash
python tools/reproduction/check_vctree_official_inputs.py \
  --output docs/reproduction/vctree/vctree_official_input_check.json
```

Observed result:

- Status: `BLOCKED`
- Blockers:
  - `missing_sgb_vg_inputs`
  - `missing_trusted_vctree_checkpoint`
- Missing SGB input:
  - `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5`
- Checkpoint candidates under `outputs/pretrained/vctree_official`: none.

## Required Verification Before Reopening

1. Choose and pin the target implementation:
   - original VCTree repository, or
   - SGB `VCTreePredictor` reimplementation.
2. Provide SGB-format VG inputs, including:

```text
/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5
```

3. Provide a trusted VCTree checkpoint with source URL, size, SHA256, and
   expected architecture.
4. Inspect checkpoint keys for SGB VCTree modules:
   - `context_layer.bi_freq_prior`
   - `context_layer.obj_ctx_rnn`
   - `context_layer.decoder_rnn`
   - `context_layer.edge_ctx_rnn`
   - `ctx_compress`
   - `freq_bias.obj_baseline.weight`
5. Run official-repo evaluation with `PREDICTOR VCTreePredictor`.
6. Only after official metrics exist, decide whether OpenSGG should:
   - import/adapt the official VCTree structure, or
   - keep current VCTree as a separate approximate implementation.

## Current Trusted Claim

OpenSGG has a VCTree-named pipeline smoke implementation. It can exercise local
metric/export paths, but it is not official VCTree reproduction-ready.

## Forbidden Claims

- Do not claim VCTree reproduction.
- Do not claim SGB `VCTreePredictor` parity.
- Do not claim checkpoint-backed VCTree performance.
- Do not use the local simplified tree constructor as official VCTree evidence.
- Do not use random-init metrics as benchmark evidence.
