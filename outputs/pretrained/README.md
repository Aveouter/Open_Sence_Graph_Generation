# Pretrained Weights

Download pretrained model weights into the corresponding subdirectories:

## EGTR

- **File:** `egtr/egtr_vg.tar.gz`
- **Source:** Extract from the official EGTR repository checkpoint.
- **Usage:** Set `egtr_pretrained_path` in the EGTR config, or pass `--egtr_pretrained_path outputs/pretrained/egtr/egtr_vg.tar.gz`.

## RelTR

- **File:** `reltr/reltr_vg.pth`
- **Source:** Official RelTR Visual Genome checkpoint.
- **Usage:** Pass via `--ckpt_path outputs/pretrained/reltr/reltr_vg.pth` for evaluation.

## Other

- **File:** `other/checkpoint0149.pth`
- **Usage:** Miscellaneous checkpoints — kept for reference.
