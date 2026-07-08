# EGTR Official-Parity Full VG SGDet Report

## Summary

Official EGTR SGDet full Visual Genome evaluation was run with the locally
downloaded official checkpoint and no training. The local OpenSGG EGTR adapter
was checked against `/workspace/Item_code/egtr` and aligned to the official
single-predicate SGDet inference/evaluation path before the full run.

## Commands

Syntax/import check:

```bash
conda run --no-capture-output -n hsg \
  python -m py_compile src/methods/egtr_method.py src/core/metrics.py
```

Full evaluation:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 conda run --no-capture-output -n hsg \
  python train.py --test --method EGTR \
  --config_file configs/VisualGenome/EGTR.py \
  --ckpt_path checkpoints/egtr/egtr_vg.ckpt \
  --val_batch_size 1 --num_workers 4 --device cuda \
  --gpus 0 1 2 3 4 5 \
  --ex_name EGTR_official_parity_full_vg_ddp6 \
  --no_display_method_info
```

Reproduction guardrails:

```bash
python tools/reproduction/run_reproduction_guardrails.py
```

## Official Reference

Official metric JSON inside `outputs/pretrained/egtr/egtr_vg.tar.gz`:

```text
epoch=03-validation_loss=1.71__test__26446__V100-PCIE-32GB.json
```

Official reference values:

| Metric | Official value |
|---|---:|
| `(single)R@20` | 0.2353710987 |
| `(single)R@50` | 0.3021079916 |
| `(single)R@100` | 0.3434980943 |
| `(single)mR@20` | 0.0546044564 |
| `(single)mR@50` | 0.0793714169 |
| `(single)mR@100` | 0.1005019707 |
| `AP50` | 0.3082679029 |

## Local Full VG Results

Run directory:

```text
outputs/runs/egtr/2026-07-08_EGTR_official_parity_full_vg_ddp6
```

Persisted metrics:

| Metric | Local value |
|---|---:|
| `sgdet_R@10` | 0.1809246838 |
| `sgdet_R@20` | 0.2346869558 |
| `sgdet_R@50` | 0.3008347452 |
| `sgdet_mR@10` | 0.0390994065 |
| `sgdet_mR@20` | 0.0549287722 |
| `sgdet_mR@50` | 0.0792875737 |

Stdout also reported:

| Metric | Local stdout value |
|---|---:|
| `sgdet_R@100` | 0.3435 |
| `sgdet_mR@100` | 0.0996 |

## Paper/Table Alignment

The official archive uses `(single)R@*` and `(single)mR@*`; local constrained
single-predicate SGDet metrics are reported as `sgdet_R@*` and `sgdet_mR@*`.

| Metric | Local full VG | Official archive | Absolute delta |
|---|---:|---:|---:|
| R@20 | 0.2346869558 | 0.2353710987 | 0.0006841429 |
| R@50 | 0.3008347452 | 0.3021079916 | 0.0012732464 |
| mR@20 | 0.0549287722 | 0.0546044564 | 0.0003243158 |
| mR@50 | 0.0792875737 | 0.0793714169 | 0.0000838432 |

## Parity Notes

- Config: local `configs/VisualGenome/EGTR.py` matches the official archive
  config for the evaluated fields: `num_queries=200`, `auxiliary_loss=false`,
  `num_labels=150`, `num_rel_labels=50`, `use_freq_bias=true`,
  `rel_loss_coefficient=15`, `connectivity_loss_coefficient=30`,
  `min_size=800`, `max_size=1333`.
- Checkpoint: official Lightning checkpoint loads with `missing=0` and
  `unexpected=0`.
- Preprocessing: local EGTR Visual Genome dataset wrapper is byte-identical to
  the official local checkout at `/workspace/Item_code/egtr/data/visual_genome.py`.
- Inference: local SGDet compact adapter now follows official object softmax
  scoring, relation-connectivity multiplication, self-pair exclusion, and
  top-100 single-predicate query-pair ranking.
- Evaluator: local evaluator is not a byte-for-byte copy of official
  `BasicSceneGraphEvaluator`, but the SGDet entry semantics are aligned. Mean
  recall now filters GT by predicate while preserving the full prediction list,
  matching the official per-predicate evaluator flow.

## Verification

- `conda run --no-capture-output -n hsg python -m py_compile ...`: passed.
- `python tools/reproduction/run_reproduction_guardrails.py`: passed.
- Full VG DDP evaluation completed under `conda hsg` with GPU DDP.

## Claim

`checkpoint_backed_evaluation` for official EGTR SGDet full VG. The R@50 and
mR@50 values align with the official archive reference within small numerical
and implementation-bridge tolerance.
