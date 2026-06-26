<div align="center">

# OpenSGG — Open Scene Graph Generation

**Modular PyTorch Lightning framework for Scene Graph Generation.**
Unifies 17 SGG methods under a single training and evaluation harness.

[![Python](https://img.shields.io/badge/python-≤3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.10-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-≥11.3-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF.svg)](https://github.com/Aveouter/Sence_Graph_Generation_of_Open_Surgery/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Overview

OpenSGG provides a unified training and evaluation framework for Scene Graph Generation models across three standard modes (**SGDet**, **SGCLS**, **PredCLS**), with Recall@K / mean Recall@K metrics and Head/Body/Tail predicate breakdowns.

- 🔌 **17 SGG methods** — from classic two-stage baselines to modern end-to-end transformers
- ⚡ **PyTorch Lightning** — clean training loop, DDP, checkpoint management, FP16 mixed precision
- 📊 **Unified evaluation** — standard R@K, mR@K, and Head/Body/Tail breakdowns
- 🧩 **Modular** — drop in new models, losses, and datasets with minimal friction
- 🤖 **CI/CD** — GitHub Actions pipeline with smoke tests and LLM-based code review

---

## Supported Methods

| Method | Venue | Type | CLI Name | Description |
|--------|-------|------|----------|-------------|
| [RelTR](https://arxiv.org/abs/2201.11450) | ECCV 2022 | End-to-end | `RelTR` | Relation Transformer — triplet-query-based SGG with Hungarian matching |
| [EGTR](https://arxiv.org/abs/2304.07670) | CVPR 2024 | End-to-end | `EGTR` | Deformable DETR backbone + lightweight relation head |
| [FlowSG](https://arxiv.org/abs/2504.03180) | CVPR 2026 | End-to-end | `FlowSG` | Flow matching with CLIP ViT-B/16, slotwise VQ-VAE, and DiT-style graph transformer |
| [USG-Par](https://arxiv.org/abs/2506.05582) | CVPR 2025 | End-to-end | `USG` | Universal Scene Graph Parser — learnable queries + RPC + transformer decoder |
| [Neural Motifs](https://arxiv.org/abs/1711.06640) | CVPR 2018 | Two-stage | `Motifs` | LSTM-based global context for object and predicate prediction |
| [VCTree](https://arxiv.org/abs/1812.01880) | CVPR 2019 | Two-stage | `VCTree` | Dynamic tree-structured object context encoding |
| [TDE](https://arxiv.org/abs/2002.11949) | CVPR 2020 | Two-stage | `TDE` | Total Direct Effect causal debiasing (built on Motifs) |
| [IMP](https://arxiv.org/abs/1701.02426) | CVPR 2017 | Two-stage | `IMP` | Iterative Message Passing for graph refinement |
| [Transformer](https://arxiv.org/abs/2006.05676) | CVPR 2020 | Two-stage | `Transformer` | Transformer-based context predictor for two-stage SGG |
| [GPS-Net](https://arxiv.org/abs/2006.05676) | CVPR 2020 | Two-stage | `GPSNet` | Graph Property Sensing Network for relation proposal |
| [PE-Net](https://arxiv.org/abs/2303.13020) | CVPR 2023 | Two-stage | `PENet` | Prototype-based Embedding Network with hierarchical alignment |
| [REACT](https://arxiv.org/abs/2406.18412) | BMVC 2025 | Two-stage | `REACT` | Prototype-regularized efficient SGG with composition analysis |
| [SHA-GCL](https://arxiv.org/abs/2203.15249) | CVPR 2022 | Two-stage | `SHAGCL` | Hybrid attention with group collaborative learning |
| [SQUAT](https://arxiv.org/abs/2303.13020) | CVPR 2023 | Two-stage | `SQUAT` | Selective quad attention for edge modeling |
| HSTRNet | — | Two-stage | `HSTRNet` | Hierarchical prototype relation learning with temporal encoding |
| CVC | — | Two-stage | `CVC` | Compositionally Verified Concept relation head with adversarial debiasing |

---

## Installation

```bash
git clone https://github.com/Aveouter/Sence_Graph_Generation_of_Open_Surgery.git
cd OpenSGG

# Create conda environment
conda env create -f environment.yml
conda activate hsg

# Install PyTorch (adjust for your CUDA version)
# CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

> If Tsinghua mirror channels in `environment.yml` are inaccessible, replace them with `defaults` and `conda-forge`, or remove the `channels` section to use your local `.condarc`.

| Dependency | Version |
|------------|---------|
| Python | ≤ 3.10 |
| PyTorch | ≥ 1.10 |
| CUDA | ≥ 11.3 (GPU) |
| PyTorch Lightning | 2.x |
| Transformers | ≥ 4.18.0 |

---

## Data Preparation

### VisualGenome

1. Download VisualGenome images (parts 1–2) from [visualgenome.org](https://visualgenome.org/)
2. Organize under `data/VisualGenome/`:

```
data/VisualGenome/
├── images/                     # Image files (from VG_100K and VG_100K_2)
├── train.json                  # COCO-format training annotations
├── val.json                    # COCO-format validation annotations
├── test.json                   # COCO-format test annotations
├── rel.json                    # Relationship metadata
├── clip_prototypes.pth         # (optional) CLIP prototypes for hierarchical alignment
└── predicate_frequencies.json  # (optional) for Head/Body/Tail analysis
```

### OpenImageV6 (Experimental)

Place COCO-format annotations and images under `data/OpenImage/`.

---

## Quick Start

### Training

```bash
# End-to-end methods
python train.py --method RelTR  --dataname VisualGenome --gpus 0 --batch_size 4
python train.py --method EGTR   --dataname VisualGenome --gpus 0 --batch_size 4
python train.py --method FlowSG --dataname VisualGenome --gpus 0 --batch_size 16
python train.py --method USG    --dataname VisualGenome --gpus 0 --batch_size 4

# Two-stage baselines
python train.py --method Motifs     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method VCTree     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method TDE        --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method IMP        --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method Transformer --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method GPSNet     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method PENet      --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method REACT      --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method SHAGCL     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method SQUAT      --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method HSTRNet    --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method CVC        --dataname VisualGenome --gpus 0 --batch_size 8
```

Config files are auto-loaded from `configs/<Dataset>/<Method>.py`. CLI arguments override config values.

### Evaluation

```bash
# Lightning checkpoint (.ckpt)
python train.py --test --method RelTR --dataname VisualGenome \
  --ckpt_path outputs/runs/reltr/<exp>/checkpoints/best-epoch=*.ckpt --gpus 0

# PyTorch checkpoint (.pth)
python train.py --test --method RelTR --dataname VisualGenome \
  --ckpt_path outputs/pretrained/other/checkpoint0149.pth --gpus 0

# Smoke test (20 samples)
python train.py --test --method EGTR --dataname VisualGenome \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-*.ckpt \
  --val_batch_size 1 --test_dataset_size 20 --num_workers 0 --gpus 0
```

### Multi-GPU

```bash
# DDP with 4 GPUs
python train.py --method RelTR --dataname VisualGenome --gpus 0 1 2 3 --dist

# Single GPU (no DDP overhead)
python train.py --method RelTR --dataname VisualGenome --gpus 0
```

---

## EGTR VisualGenome Reproduction

Validated against the pretrained checkpoint.

**Label space:** OpenSGG uses 1-indexed labels with background (`entity_nums=151`, `rel_nums=51`). EGTR logits use explicit no-background dimensions (`egtr_num_labels=150`, `egtr_num_rel_labels=50`). The checkpoint loads non-zero `rel_dist` and `triplet_dist` frequency-bias parameters.

| Task | Metric | Value |
|------|--------|-------|
| PredCLS | R@20 | 51.10 |
| PredCLS | mR@20 | 18.87 |
| SGDet | R@20 | 23.47 |
| SGDet | R@50 | 30.08 |
| SGDet | mR@20 | 9.72 |

> SGCLS is skipped for EGTR unless a dedicated adapter is added.

---

## Configuration

Configs use a two-layer system: **CLI arguments** override **per-method config files**.

```
configs/
├── _template.py                    # All available options
└── VisualGenome/
    ├── RelTR.py, EGTR.py, FlowSG.py, USG.py
    ├── HSTRNet.py, CVC.py
    ├── Motifs.py, VCTree.py, TDE.py
    ├── IMP.py, Transformer.py
    ├── GPS_Net.py, PE_NET.py
    └── REACT.py, SHA_GCL.py, SQUAT.py
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Model method | `HSTRNet` |
| `dataname` | Dataset (`VisualGenome` / `OpenImageV6`) | `VisualGenome` |
| `batch_size` | Training batch size | `4` |
| `dataset_size` | Train subset size (smoke tests) | `None` |
| `epoch` | Max training epochs | `200` |
| `lr` / `lr_backbone` | Learning rate (model / backbone) | `1e-4` / `1e-5` |
| `backbone` | CNN backbone | `resnet50` |
| `hidden_dim` | Transformer hidden dim | `256` |
| `enc_layers` / `dec_layers` | Encoder / decoder layers | `6` |
| `num_entities` / `num_triplets` | Query counts (RelTR) | `100` / `200` |
| `entity_nums` / `rel_nums` | Classes + background | `151` / `51` |

---

## Evaluation Metrics

| Format | Example | Description |
|--------|---------|-------------|
| `sgdet_R@K` | `sgdet_R@50` | SGDet Recall@K |
| `sgdet_mR@K` | `sgdet_mR@50` | SGDet mean Recall@K |
| `sgcls_R@K` | `sgcls_R@20` | SGCLS Recall@K |
| `sgcls_mR@K` | `sgcls_mR@20` | SGCLS mean Recall@K |
| `predcls_R@K` | `predcls_R@50` | PredCLS Recall@K |
| `predcls_mR@K` | `predcls_mR@20` | PredCLS mean Recall@K |

**Modes:** **SGDet** (boxes + labels + predicates from raw images) · **SGCLS** (boxes given, predict labels + predicates) · **PredCLS** (boxes + labels given, predict predicates)

---

## Reports And Guides

- [Published Method Reproduction Results](guides/published_method_reproduction.md) — concise PR-ready summary of reproduced results for published paper methods.
- [Adding A New Model](guides/adding_new_model.md) — step-by-step contributor guide for integrating a new method into the main training and evaluation pipeline.

---

## CI/CD Pipeline

GitHub Actions runs on every PR and push to `main`:

- **Smoke tests** — forward pass + loss convergence on all 17 methods with synthetic data
- **LLM code review** — automated review via Claude/DeepSeek API on changed files

Trigger paths: `src/`, `configs/`, `utils/`, `train.py`, `tools/`, `.github/workflows/`.

---

## Project Structure

```
OpenSGG/
├── train.py                               # Entry point
├── .github/workflows/ci.yml               # CI pipeline
├── configs/                               # Per-dataset, per-method configs
│   ├── _template.py
│   └── VisualGenome/                      # 17 method configs
├── guides/                                # Reproduction reports and contributor guides
├── src/                                   # Core framework
│   ├── exp.py                             # Experiment orchestration
│   ├── methods/                           # LightningModule wrappers (17 methods)
│   │   ├── base_method.py                 # Base class (DDP eval, loss averaging)
│   │   └── reltr_method.py, egtr_method.py, flowsg_method.py, usg_method.py, ...
│   ├── models/                            # Model definitions
│   │   ├── backbone.py                    # Shared backbone factory
│   │   ├── reltr.py, egtr.py, flowsg.py, usg.py, hstrnet.py, cvc.py, gen_sgg.py
│   │   └── motifs.py, vctree.py, imp.py, transformer_sgg.py, ...
│   ├── modules/                           # Reusable building blocks
│   │   ├── layers/                        # Backbone, Transformer, Matcher, CLIP
│   │   └── egtr/                          # EGTR: Deformable DETR + SGG head
│   ├── losses/                            # Loss function registry
│   │   ├── loss.py                        # HSTRCriterion + LOSS_FACTORY
│   │   └── reweight_loss.py               # Focal, class-balanced losses
│   └── core/                              # Metrics, optimizers, schedulers
│       ├── metrics.py                     # Unified evaluation pipeline
│       └── optim_scheduler.py             # Optimizer/scheduler factory
├── data/dataloaders/                      # Data loading
│   ├── dataloader.py                      # Dataset router
│   ├── dataloader_VisualGenome.py         # VisualGenome loader
│   ├── coco.py                            # COCO dataset utilities
│   └── egtr/                              # EGTR-specific data processing
├── lib/                                   # Vendored libraries
│   └── evaluation/sg_eval.py              # Recall computation (Danfei Xu benchmark)
├── tools/analysis/                        # Analysis & evaluation scripts
├── utils/                                 # Utilities
│   ├── parser.py                          # Argument parsing
│   ├── main_utils.py                      # Config loading, env setup
│   └── misc.py                            # NestedTensor, collate, distributed helpers
└── outputs/                               # Run artifacts (not tracked)
    ├── pretrained/                        # Pretrained weights
    └── runs/                              # Experiment outputs
```

---

## How to Add a New Method

See [Adding A New Model](guides/adding_new_model.md) for the complete model,
method wrapper, config, registry, CLI, metric, and smoke-test checklist.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use OpenSGG, please cite the framework:

```bibtex
@misc{opensgg,
  title     = {OpenSGG: Open Scene Graph Generation Framework},
  author    = {Xinyu Liu and Xiaoguang Lin and contributors},
  year      = {2025},
  note      = {https://github.com/Aveouter/Sence_Graph_Generation_of_Open_Surgery},
}
```

Please also cite the original papers for any methods you use: RelTR (ECCV 2022), EGTR (CVPR 2024), FlowSG (CVPR 2026), USG-Par (CVPR 2025), Neural Motifs (CVPR 2018), VCTree (CVPR 2019), TDE (CVPR 2020), IMP (CVPR 2017), Transformer (CVPR 2020), GPS-Net (CVPR 2020), PE-Net (CVPR 2023), REACT (BMVC 2025), SHA-GCL (CVPR 2022), SQUAT (CVPR 2023).

---

<div align="center">
  <sub>Built by the CIGIT HPC Lab team and open-source contributors.</sub>
</div>
