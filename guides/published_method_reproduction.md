# Published Method Checkpoint Evaluations

This document records traced checkpoint-backed OpenSGG evaluation results for
published paper methods only. It does not include exploratory or self-designed
experiments, and it is not a substitute for the stricter reproduction workflow
in `AGENTS.md` and `reproduction/`.

## Scope

- Dataset: VisualGenome.
- Reported values are percentages.
- Metrics follow the OpenSGG evaluation names in `src/core/metrics.py`.
- Source artifacts under `outputs/` are local/generated and may not be tracked
  in git, so this file keeps the PR-facing summary in a stable location.
- A section may be called paper-aligned reproduction only when method,
  checkpoint, config, inference flow, and evaluator semantics have all been
  checked against the official paper or repository.

## EGTR

Paper: [EGTR](https://arxiv.org/abs/2304.07670)

Venue: CVPR 2024

Implementation entry points:

- Config: `configs/VisualGenome/EGTR.py`
- Method wrapper: `src/methods/egtr_method.py`
- Model code: `src/modules/egtr/`

Checkpoint:

```text
outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71.ckpt
```

Evaluation command recorded in the run metadata:

```bash
python train.py -m EGTR -d VisualGenome --test \
  --ckpt_path outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71.ckpt \
  --val_batch_size 1 \
  --gpus 0
```

Source metric artifact:

```text
outputs/runs/egtr/2026-06-12_Debug_003/eval/sgdet/metrics.json
```

### Metrics

| Task | Metric | Value |
|------|--------|-------|
| PredCLS | R@20 | 51.10 |
| PredCLS | mR@20 | 18.87 |
| SGDet | R@20 | 23.47 |
| SGDet | R@50 | 30.08 |
| SGDet | mR@20 | 9.72 |

### Notes

- OpenSGG uses 1-indexed VisualGenome labels with background
  (`entity_nums=151`, `rel_nums=51`).
- EGTR logits use no-background dimensions
  (`egtr_num_labels=150`, `egtr_num_rel_labels=50`).
- The evaluated checkpoint loads non-zero frequency-bias parameters
  (`rel_dist`, `triplet_dist`).
- SGCLS is not listed here because EGTR currently needs a dedicated SGCLS
  adapter before the metric can be reported as a supported checkpoint-backed
  evaluation.

## Additions Policy

Add a new row or section here only when all of the following are true:

- The method is from a published paper or accepted venue.
- The metric is from the standard OpenSGG evaluation path, not a custom
  exploratory analysis.
- The checkpoint, command, dataset split, and metric artifact can be traced.
- Any label-space or evaluation-mode caveat is documented.
