<div align="center">

# OpenSGG — Open Scene Graph Generation Framework

**Modular, extensible PyTorch Lightning framework for Scene Graph Generation.**
Unifies 15+ SGG methods under a single training and evaluation harness.

[![Python](https://img.shields.io/badge/python-≤3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.10-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-≥11.3-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Supported Methods](#supported-methods)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Multi-GPU Training](#multi-gpu-training)
- [EGTR VisualGenome Reproduction](#egtr-visualgenome-reproduction)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [How to Add a New Method](#how-to-add-a-new-method)
- [License](#license)
- [Citation](#citation)

---

## Overview

OpenSGG provides a unified PyTorch Lightning framework for training, evaluating, and comparing Scene Graph Generation models. It supports three standard evaluation modes (**SGDet**, **SGCLS**, **PredCLS**) and comprehensive Recall@K / mean Recall@K metrics with Head/Body/Tail predicate analysis.

Key features:

- 🔌 **15+ SGG methods** — from classic two-stage baselines to modern end-to-end transformers
- ⚡ **PyTorch Lightning** — clean training loop, DDP support, checkpoint management
- 📊 **Unified evaluation** — standard Recall@K, mean Recall@K, and Head/Body/Tail breakdowns
- 🧩 **Modular architecture** — drop in new models, losses, and datasets with minimal friction
- 🚀 **Production-ready** — FP16 mixed precision, multi-GPU DDP, configurable backbones

---

## Supported Methods

| Method | Venue | Type | CLI Name | Description |
|--------|-------|------|----------|-------------|
| [RelTR](https://arxiv.org/abs/2201.11450) | ECCV 2022 | End-to-end | `RelTR` | Relation Transformer — triplet-query-based SGG with Hungarian matching |
| [EGTR](https://arxiv.org/abs/2304.07670) | CVPR 2024 | End-to-end | `EGTR` | Deformable DETR backbone + lightweight relation head |
| [FlowSG](https://arxiv.org/abs/2504.03180) | CVPR 2026 | End-to-end | `FlowSG` | Flow matching with CLIP ViT-B/16, slotwise VQ-VAE, and DiT-style graph transformer |
| [Neural Motifs](https://arxiv.org/abs/1711.06640) | CVPR 2018 | Two-stage | `Motifs` | LSTM-based global context for object and predicate prediction |
| [VCTree](https://arxiv.org/abs/1812.01880) | CVPR 2019 | Two-stage | `VCTree` | Dynamic tree-structured object context encoding |
| [TDE](https://arxiv.org/abs/2002.11949) | CVPR 2020 | Two-stage | `TDE` | Total Direct Effect causal debiasing (built on Motifs) |
| [IMP](https://arxiv.org/abs/1701.02426) | CVPR 2017 | Two-stage | `IMP` | Iterative Message Passing for graph refinement |
| [Transformer](https://arxiv.org/abs/2006.05676) | CVPR 2020 | Two-stage | `Transformer` | Transformer-based context predictor for two-stage SGG |
| [GPS-Net](https://arxiv.org/abs/2006.05676) | CVPR 2020 | Two-stage | `GPS_Net` | Graph Property Sensing Network for relation proposal |
| [PE-Net](https://arxiv.org/abs/2303.13020) | CVPR 2023 | Two-stage | `PE_NET` | Prototype-based Embedding Network with hierarchical alignment |
| [REACT](https://arxiv.org/abs/2406.18412) | BMVC 2025 | Two-stage | `REACT` | Prototype-regularized efficient SGG with composition analysis |
| [SHA-GCL](https://arxiv.org/abs/2203.15249) | CVPR 2022 | Two-stage | `SHA_GCL` | Hybrid attention with group collaborative learning |
| [SQUAT](https://arxiv.org/abs/2303.13020) | CVPR 2023 | Two-stage | `SQUAT` | Selective quad attention for edge modeling |
| HSTRNet | — | Two-stage | `HSTRNet` | Hierarchical prototype relation learning with temporal encoding |
| CVC | — | Two-stage | `CVC` | Compositionally Verified Concept relation head with adversarial debiasing |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Aveouter/Sence_Graph_Generation_of_Open_Surgery.git
cd OpenSGG

# Create conda environment
conda env create -f environment.yml
conda activate hsg

# Install PyTorch (adjust for your CUDA version)
# CUDA 11.8 example:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 example:
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** If the Tsinghua mirror channels in `environment.yml` are inaccessible from your region, replace them with `defaults` and `conda-forge` channels, or remove the `channels` section to use your local `.condarc` configuration.

| Dependency | Version |
|------------|---------|
| Python | ≤ 3.10 |
| PyTorch | ≥ 1.10 |
| CUDA | ≥ 11.3 (optional, for GPU) |
| PyTorch Lightning | 2.x |
| Transformers | ≥ 4.18.0 |

---

## Data Preparation

### VisualGenome

1. Download VisualGenome images (parts 1–2) from [visualgenome.org](https://visualgenome.org/)
2. Organize under `data/VisualGenome/`:

```
data/VisualGenome/
├── VG_100K/                    # Raw images (part 1)
├── VG_100K_2/                  # Raw images (part 2)
├── annotations/                # COCO-style JSON annotations
├── clip_prototypes.pth         # (optional) CLIP prototypes for hierarchical alignment
└── predicate_frequencies.json  # (optional) for Head/Body/Tail analysis
```

### OpenImageV6 (Experimental)

Place COCO-format annotations under `data/OpenImage/`.

---

## Quick Start

### Training

```bash
# End-to-end methods
python train.py --method RelTR  --dataname VisualGenome --gpus 0 --batch_size 4
python train.py --method EGTR   --dataname VisualGenome --gpus 0 --batch_size 4
python train.py --method FlowSG --dataname VisualGenome --gpus 0 --batch_size 16

# Two-stage baselines
python train.py --method Motifs     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method VCTree     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method TDE        --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method IMP        --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method Transformer --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method GPS_Net    --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method PE_NET     --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method REACT      --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method SHA_GCL    --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method SQUAT      --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method HSTRNet    --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method CVC        --dataname VisualGenome --gpus 0 --batch_size 8
```

Config files are auto-loaded from `configs/<DatasetName>/<MethodName>.py`. Command-line arguments override config file values.

### Evaluation

```bash
# Test with Lightning checkpoint (.ckpt)
python train.py --test \
  --method RelTR \
  --dataname VisualGenome \
  --ckpt_path outputs/runs/reltr/<exp_name>/checkpoints/best-epoch=*.ckpt \
  --gpus 0

# Test with PyTorch checkpoint (.pth / .pt)
python train.py --test \
  --method RelTR \
  --dataname VisualGenome \
  --ckpt_path outputs/pretrained/other/checkpoint0149.pth \
  --gpus 0

# Smoke test (20 samples, single GPU, no extra workers)
python train.py --test \
  --method EGTR \
  --dataname VisualGenome \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-validation_loss=1.71.ckpt \
  --val_batch_size 1 \
  --test_dataset_size 20 \
  --num_workers 0 \
  --gpus 0
```

### Multi-GPU Training

```bash
# DDP training with 4 GPUs
python train.py --method RelTR --dataname VisualGenome --gpus 0 1 2 3 --dist

# Single-GPU mode (pass a single GPU to avoid DDP overhead)
python train.py --method RelTR --dataname VisualGenome --gpus 0
```

---

## EGTR VisualGenome Reproduction

The EGTR VisualGenome evaluation has been validated against the pretrained checkpoint.

**Label space alignment:**
- OpenSGG labels: 1-indexed with background (`entity_nums=151`, `rel_nums=51`)
- EGTR logits: explicit no-background dimensions (`egtr_num_labels=150`, `egtr_num_rel_labels=50`)
- The VisualGenome checkpoint loads non-zero `rel_dist` and `triplet_dist` frequency-bias parameters
- `egtr_sgdet_postprocess='query'` is the default; `qc_topk` is available for DETR-style `Q*C` top-k experiments

**Reproduced results on VisualGenome test set:**

| Task | Metric | Value |
|------|--------|-------|
| PredCLS | R@20 | 51.10 |
| PredCLS | mR@20 | 18.87 |
| SGDet | R@20 | 23.47 |
| SGDet | R@50 | 30.08 |
| SGDet | mR@20 | 9.72 |

> **Note:** SGCLS is currently skipped for EGTR unless a dedicated adapter is added.

---

## Configuration

Configuration uses a two-layer system: **command-line arguments** override **per-method config files**.

### Config File Convention

```
configs/
├── _template.py                    # Template with all available options
└── VisualGenome/
    ├── RelTR.py                    # RelTR config
    ├── EGTR.py                     # EGTR config
    ├── FlowSG.py                   # FlowSG config
    ├── HSTRNet.py                  # HSTRNet config
    ├── Motifs.py / VCTree.py       # Two-stage baseline configs
    ├── TDE.py / CVC.py             # Debiasing method configs
    ├── IMP.py / Transformer.py     # Message-passing / transformer configs
    ├── GPS_Net.py / PE_NET.py      # Graph / prototype configs
    └── REACT.py / SHA_GCL.py / SQUAT.py  # Advanced two-stage configs
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Model method (see supported list above) | `HSTRNet` |
| `dataname` | Dataset name (`VisualGenome` / `OpenImageV6`) | `VisualGenome` |
| `batch_size` | Training batch size | `4` |
| `dataset_size` | Optional train subset size (smoke tests) | `None` |
| `val_dataset_size` | Optional val subset size | `None` |
| `test_dataset_size` | Optional test subset size (overrides `val_dataset_size` in test mode) | `None` |
| `epoch` | Max training epochs | `200` |
| `lr` | Learning rate | `1e-4` |
| `lr_backbone` | Backbone learning rate | `1e-5` |
| `backbone` | CNN backbone (`resnet50` / `resnet101`) | `resnet50` |
| `hidden_dim` | Transformer hidden dimension | `256` |
| `enc_layers` | Number of encoder layers | `6` |
| `dec_layers` | Number of decoder layers | `6` |
| `num_entities` | Entity queries (RelTR) | `100` |
| `num_triplets` | Triplet queries (RelTR) | `200` |
| `entity_nums` | Entity classes (+ background) | `151` |
| `rel_nums` | Predicate classes (+ background) | `51` |
| `egtr_num_labels` | EGTR object logit classes (no background) | `150` |
| `egtr_num_rel_labels` | EGTR predicate logit classes (no background) | `50` |
| `egtr_sgdet_postprocess` | EGTR SGDet postprocess (`query` / `qc_topk`) | `query` |

---

## Evaluation Metrics

Metrics follow the standard SGG convention: `{task}_{metric}@{K}`.

| Format | Example | Description |
|--------|---------|-------------|
| `sgdet_R@K` | `sgdet_R@50` | SGDet Recall@K |
| `sgdet_mR@K` | `sgdet_mR@50` | SGDet mean Recall@K (per-predicate average) |
| `sgcls_R@K` | `sgcls_R@20` | SGCLS Recall@K |
| `sgcls_mR@K` | `sgcls_mR@20` | SGCLS mean Recall@K |
| `predcls_R@K` | `predcls_R@50` | PredCLS Recall@K |
| `predcls_mR@K` | `predcls_mR@20` | PredCLS mean Recall@K |

Configure in the `metrics` config field, e.g.: `["sgdet_R@50", "sgdet_mR@50", "predcls_R@20"]`

### Evaluation Modes

| Mode | Description |
|------|-------------|
| **SGDet** | Full end-to-end: predict boxes, object labels, and predicates from raw images |
| **SGCLS** | Given ground-truth boxes, predict object labels and predicates |
| **PredCLS** | Given ground-truth boxes and labels, predict predicates only |

---

## Project Structure

```
OpenSGG/
├── train.py                               # Main entry point
├── configs/                               # Per-dataset, per-method configs
│   ├── _template.py                       # Template config
│   └── VisualGenome/                      # VisualGenome configs
│       ├── RelTR.py, EGTR.py, FlowSG.py, HSTRNet.py
│       ├── Motifs.py, VCTree.py, TDE.py, CVC.py
│       └── IMP.py, Transformer.py, GPS_Net.py, PE_NET.py,
│           REACT.py, SHA_GCL.py, SQUAT.py
├── src/                                   # Core framework
│   ├── exp.py                             # Experiment class (train/test orchestration)
│   ├── loss.py                            # Loss backward-compat shim
│   ├── core/                              # Metrics, optimizers, schedulers
│   │   ├── metrics.py                     # Unified evaluation pipeline
│   │   ├── optim_scheduler.py             # Optimizer/scheduler factory
│   │   ├── optim_constant.py              # Optimizer parameter grouping
│   │   └── drop_scheduler.py              # Drop path scheduling
│   ├── methods/                           # LightningModule wrappers (15 methods)
│   │   ├── __init__.py                    # method_maps registry
│   │   ├── base_method.py                 # Base class (DDP eval, loss averaging)
│   │   ├── reltr_method.py, egtr_method.py, flowsg_method.py
│   │   ├── hstrnet_method.py
│   │   └── motifs_method.py, vctree_method.py, tde_method.py,
│   │       cvc_method.py, imp_method.py, transformer_method.py,
│   │       gpsnet_method.py, penet_method.py, react_method.py,
│   │       shagcl_method.py, squat_method.py
│   ├── models/                            # Model definitions (15 models)
│   │   ├── reltr.py, HSTRNet.py, flowsg.py, cvc.py
│   │   ├── backbone.py                    # Shared backbone factory
│   │   └── motifs.py, vctree.py, imp.py, transformer_sgg.py,
│   │       gpsnet.py, penet.py, react_sgg.py, shagcl.py, squat.py
│   ├── modules/                           # Reusable building blocks
│   │   ├── layers/                        # Backbone, Transformer, Matcher, CLIP
│   │   └── egtr/                          # EGTR: Deformable DETR + SGG head + transforms
│   └── losses/                            # Loss function registry
│       ├── loss.py                        # HSTRCriterion + LOSS_FACTORY
│       └── reweight_loss.py               # Focal, class-balanced, reweighting losses
├── data/                                  # Data loading
│   └── dataloaders/
│       ├── dataloader.py                  # Routing to dataset-specific loaders
│       ├── dataloader_VisualGenome.py     # VisualGenome data loading
│       ├── coco.py                        # COCO dataset + build functions
│       ├── base_data.py                   # LightningDataModule wrapper
│       ├── dataset_constant.py            # Per-dataset default parameters
│       └── egtr/                          # EGTR-specific data processing
├── lib/                                   # Vendored libraries
│   ├── evaluation/sg_eval.py              # Core recall computation (Danfei Xu benchmark)
│   ├── fpn/box_utils.py                   # Bounding box utilities
│   └── pytorch_misc.py                    # Intersection utilities
├── utils/                                 # Utilities
│   ├── parser.py                          # Argument parsing
│   ├── main_utils.py                      # Config loading, env setup, throughput
│   ├── callbacks.py                       # Lightning callbacks
│   ├── box_ops.py                         # Box coordinate operations
│   ├── misc.py                            # NestedTensor, collate, distributed helpers
│   ├── config_utils.py                    # Config file parser (mmcv-compatible)
│   └── transforms.py                      # DETR-style data transforms
├── scripts/                               # Utility scripts
│   ├── clip/                              # CLIP prototype/embedding builders
│   └── cluster/                           # Clustering utilities
└── outputs/                               # All run artifacts
    ├── pretrained/                        # Pretrained weights
    └── runs/                              # Experiment outputs (logs, checkpoints, eval)
```

---

## How to Add a New Method

1. **Create your model** under `src/models/` — implement the SGG model
2. **Create a LightningModule wrapper** under `src/methods/` — subclass `BaseMethod`
3. **Register** in `src/methods/__init__.py` — add to `method_maps` dict and `__all__`
4. **Add a config file** at `configs/<Dataset>/<Method>.py`
5. (Optional) Add an evaluation adapter in `src/core/metrics.py`
6. **Add the CLI choice** in `utils/parser.py` if it's a new method name

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you use OpenSGG in your research, please cite both this framework and the original papers for each method you use:

```bibtex
@misc{opensgg,
  title        = {OpenSGG: Open Scene Graph Generation Framework},
  author       = {Xinyu Liu, Xiaoguang Lin, and contributors},
  year         = {2025},
  note         = {https://github.com/Aveouter/Sence_Graph_Generation_of_Open_Surgery},
}
```

**Original method papers:**

- **RelTR**: Cong et al., "RelTR: Relation Transformer for Scene Graph Generation", ECCV 2022
- **EGTR**: Im et al., "EGTR: Extracting Graph from Transformer for Scene Graph Generation", CVPR 2024
- **FlowSG**: Hu et al., "FlowSG: Progressive Image-Conditioned Scene Graph Generation with Flow Matching", CVPR 2026
- **Neural Motifs**: Zellers et al., "Neural Motifs: Scene Graph Parsing with Global Context", CVPR 2018
- **VCTree**: Tang et al., "Learning to Compose Dynamic Tree Structures for Visual Contexts", CVPR 2019
- **TDE**: Tang et al., "Unbiased Scene Graph Generation from Biased Training", CVPR 2020
- **IMP**: Xu et al., "Scene Graph Generation by Iterative Message Passing", CVPR 2017
- **Transformer**: Tang et al., "Scene Graph Generation with Transformer", CVPR 2020
- **GPS-Net**: Lin et al., "GPS-Net: Graph Property Sensing Network for Scene Graph Generation", CVPR 2020
- **PE-Net**: Zheng et al., "Prototype-based Embedding Network for Scene Graph Generation", CVPR 2023
- **REACT**: Li et al., "Regularized Composition-Aware Prototype Learning for Scene Graph Generation", BMVC 2025
- **SHA-GCL**: Li et al., "Stacked Hybrid Attention with Group Collaborative Learning", CVPR 2022
- **SQUAT**: Jung et al., "SQUAT: Selective Quad Attention for Scene Graph Generation", CVPR 2023

---

<div align="center">
  <sub>Built with ❤️ by the CIGIT HPC Lab team and open-source contributors.</sub>
</div>
