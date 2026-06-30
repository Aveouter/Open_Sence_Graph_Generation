# FREQ Evidence Gate Audit

Current status: `DEFERRED_NOT_REPRODUCED`.

This audit records the evidence required to reopen FREQ as a reproduction PR.
It is not a reproduction result.

## Official Sources

Primary paper/source:

- Paper: `Neural Motifs: Scene Graph Parsing with Global Context`, Zellers et
  al., CVPR 2018.
- Project page: https://rowanzellers.com/neuralmotifs/
- Official code: https://github.com/rowanz/neural-motifs

Reference implementation available locally:

- Repository: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
- Local path:
  `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
- Local commit: `ceb71fa88461c2a97a6258a80f47669d89207296`

The Neural Motifs paper describes FREQ as predicting the most frequent relation
between object-pair labels seen in training. The local SGB reference implements
this family of priors as `FrequencyBias`.

## Official Reference Behavior

Source locations in the local SGB reference:

- `maskrcnn_benchmark/modeling/roi_heads/relation_head/model_motifs.py`
  - `FrequencyBias.__init__` reads `statistics['pred_dist']`.
  - `index_with_labels(labels)` returns
    `obj_baseline[subject_label * num_objects + object_label]`.
  - `index_with_probability(pair_prob)` computes the subject/object joint
    probability and multiplies by the prior table.
- `maskrcnn_benchmark/data/datasets/visual_genome.py`
  - `get_statistics()` builds `fg_matrix`, adds background counts, and stores
    `pred_dist = log(fg_matrix / sum(fg_matrix) + eps)`.
  - `get_VG_statistics(..., must_overlap=True)` computes foreground relation
    counts from the training split and background counts from overlapping object
    pairs.

## OpenSGG Current Behavior

Source locations:

- `src/methods/freq_method.py`
  - `FREQModel` enumerates all directed non-self GT object pairs.
  - It scores each pair with `PairFrequencyBias.index_with_labels`.
  - It returns Motifs-style `rel_logits`, `pair_indices`, and
    `predicate_bg_index="first"`.
- `src/models/motifs.py`
  - `PairFrequencyBias` mirrors the SGB lookup API.
  - When `data_root` is available, it rebuilds a prior from
    `data/VisualGenome/train.json` and `data/VisualGenome/rel.json`.
  - If those files are absent or unreadable, it silently falls back to a uniform
    prior.
- `src/core/metrics.py`
  - `_evaluate_pair_indexed_predcls` matches GT relations by exact directed
    `(subject_idx, object_idx)`.
  - `_extract_relation_scores(..., predicate_bg_index="first",
    softmax_scope="all")` removes the background column after full softmax.

## Gate Status

| Gate | Status | Evidence | Gap |
|---|---|---|---|
| FREQ-1 | Partial | FREQ target is now tied to Neural Motifs/SGB frequency prior. | Need decide whether the accepted target is original `rowanz/neural-motifs` or SGB-style `FrequencyBias` parity. |
| FREQ-2 | Failing | OpenSGG can rebuild a prior from local `train.json`/`rel.json`. | Need prove the local train split and relation filtering match the official VG split/statistics. |
| FREQ-3 | Partial | Both implementations use `subject * num_objects + object` lookup over log prior table. | Need numerical comparison between OpenSGG `PairFrequencyBias.obj_baseline.weight` and SGB `statistics['pred_dist']` for the same VG statistics. |
| FREQ-4 | Failing | Tiny CPU smoke/export exists. | Need benchmark-scale PredCls evaluation with unchanged evaluator semantics. |

## Current Export Blocker

Input check command:

```bash
python tools/reproduction/check_sgb_freq_inputs.py \
  --output docs/reproduction/freq/sgb_freq_input_check.json
```

Observed result:

- Status: `BLOCKED_MISSING_SGB_VG_INPUTS`.
- Present:
  - SGB checkout:
    `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
  - SGB config:
    `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
  - SGB dictionary:
    `datasets/vg/VG-SGG-dicts-with-attri.json`
  - SGB image metadata:
    `datasets/vg/image_data.json`
- Missing:
  - `datasets/vg/VG-SGG-with-attri.h5`

Because the SGB reference computes `statistics['pred_dist']` from the roidb h5
file, the official prior cannot be exported in the current workspace. The local
OpenSGG `train.json`/`rel.json` files must not be treated as equivalent without
a documented conversion/parity check.

## Required Verification Before Reopening

1. Provide or locate the SGB-format VG roidb file:

```text
/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5
```

2. Build or locate the official SGB statistics cache for the same VG split:

```bash
python - <<'PY'
from maskrcnn_benchmark.data import get_dataset_statistics
# Run inside the SGB environment with the exact VG config used for PredCls.
PY
```

3. Export the official `pred_dist` and compare it against OpenSGG:

```bash
python tools/reproduction/compare_freq_prior.py \
  --sgb-statistics <path-to-sgb-statistics-cache-or-export> \
  --opensgg-data-root data/VisualGenome \
  --output docs/reproduction/freq/freq_prior_comparison.json
```

The comparison output must include:

- object class count and predicate count,
- background index convention,
- maximum absolute difference over the prior table,
- number of differing entries above tolerance,
- sampled disagreements with subject/object/predicate names.

4. Run full PredCls evaluation with the aligned prior:

```bash
conda run -n hsg python train.py \
  --test \
  --method FREQ \
  --dataname VisualGenome \
  --eval_mode predcls \
  --gpus 0
```

If GPU resources are unavailable, record `RESOURCE_BLOCKED` and keep FREQ
deferred. Do not replace this with a tiny-slice reproduction claim.

## Current Trusted Claim

OpenSGG has a FREQ-compatible pipeline smoke path. The implemented prior API is
structurally similar to SGB `FrequencyBias`.

## Forbidden Claims

- Do not claim paper-aligned FREQ reproduction.
- Do not claim the local prior table matches official SGB statistics.
- Do not claim benchmark FREQ results from the tiny slice.
- Do not use the uniform-prior fallback as FREQ reproduction evidence.
