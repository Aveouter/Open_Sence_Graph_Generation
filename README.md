# OpenSGG — Open Scene Graph Generation Framework

A modular, extensible PyTorch Lightning framework for **Scene Graph Generation** (SGG).

OpenSGG unifies multiple state-of-the-art SGG methods under one training and evaluation harness, supporting three standard evaluation modes (**SGDet**, **SGCLS**, **PredCLS**) and comprehensive Recall@K / mean Recall@K metrics with Head/Body/Tail analysis.

## Supported Methods

| Method | Venue | Description | Status |
|--------|-------|-------------|--------|
| **RelTR** | ECCV 2022 | Relation Transformer — triplet-query-based SGG with Hungarian matching | Stable |
| **EGTR** | CVPR 2024 | Extracting Graph from Transformer — Deformable DETR backbone + lightweight relation head | Reproduced on VG |
| **HSTRNet** | Custom | Hierarchical prototype relation learning with temporal encoding | Experimental |
| **Motifs** | CVPR 2018 | Neural Motifs two-stage SGG baseline | Experimental |
| **VCTree** | CVPR 2019 | Tree-structured object context for SGG | Experimental |
| **TDE** | CVPR 2020 | Total Direct Effect debiasing variant built on Motifs | Experimental |
| **IMP** | CVPR 2017 | Iterative Message Passing for scene graph generation | Experimental |
| **Transformer** | CVPR 2020 | Transformer context predictor for two-stage SGG | Experimental |
| **GPSNet** | CVPR 2020 | Graph property sensing network | Experimental |
| **PENet** | CVPR 2023 | Prototype-based embedding network | Experimental |
| **REACT** | BMVC 2025 | Prototype-regularized efficient SGG model | Experimental |
| **SHA-GCL** | CVPR 2022 | Hybrid attention with group collaborative learning | Experimental |
| **SQUAT** | CVPR 2023 | Selective quad attention for edge modeling | Experimental |

## EGTR VisualGenome Reproduction

The EGTR VisualGenome evaluation path has been aligned with the pretrained
VisualGenome checkpoint convention:

- OpenSGG labels remain 1-indexed with background counts (`entity_nums=151`,
  `rel_nums=51`).
- EGTR logits use explicit no-background dimensions
  (`egtr_num_labels=150`, `egtr_num_rel_labels=50`).
- The VisualGenome checkpoint loads non-zero `rel_dist` and `triplet_dist`
  frequency-bias parameters.
- `egtr_sgdet_postprocess='query'` is the default because it matches the
  observed EGTR SGDet reproduction result. `qc_topk` is available for
  Deformable-DETR-style `Q*C` object top-k experiments.

Full VisualGenome test with the pretrained EGTR checkpoint reached:

| Metric | Value |
|--------|-------|
| PredCLS R@20 | 51.10 |
| PredCLS mR@20 | 18.87 |
| SGDet R@20 | 23.47 |
| SGDet R@50 | 30.08 |
| SGDet mR@20 | 9.72 |

Example command:

```bash
python train.py -m EGTR -d VisualGenome --test \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-validation_loss=1.71.ckpt \
  --val_batch_size 1 \
  --gpus 0
```

For quick smoke tests, use:

```bash
python train.py -m EGTR -d VisualGenome --test \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-validation_loss=1.71.ckpt \
  --val_batch_size 1 \
  --test_dataset_size 20 \
  --num_workers 0 \
  --gpus 0
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd OpenSGG

# Create conda environment
conda env create -f environment.yml
conda activate hsg

# Install PyTorch (CUDA 11.8 example — adjust for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**Requirements:** Python ≤ 3.10, PyTorch ≥ 1.10, CUDA ≥ 11.3 (optional), transformers ≥ 4.18.0

## Data Preparation

### VisualGenome

```bash
# Download VisualGenome images (parts 1-2) from https://visualgenome.org/
# Expected structure after extraction:
data/VisualGenome/
├── VG_100K/           # Raw images
├── VG_100K_2/
├── annotations/       # COCO-style JSON annotations
├── clip_prototypes.pth   # (optional) CLIP prototypes for hierarchical alignment
└── predicate_frequencies.json  # (optional) for head/body/tail analysis
```

### OpenImageV6 (experimental)

Set up under `data/OpenImage/` with COCO-format annotations.

## Quick Start

### Training

```bash
# Train RelTR on VisualGenome
python train.py --method RelTR --dataname VisualGenome --gpus 0 --batch_size 4

# Train EGTR on VisualGenome
python train.py --method EGTR --dataname VisualGenome --gpus 0 --batch_size 4

# Train HSTRNet on VisualGenome
python train.py --method HSTRNet --dataname VisualGenome --gpus 0 --batch_size 16

# Train a two-stage baseline, e.g. Neural Motifs
python train.py --method Motifs --dataname VisualGenome --gpus 0 --batch_size 8
```

Config files are auto-loaded from `configs/<DatasetName>/<MethodName>.py`. Command-line arguments override config values.

### Evaluation

```bash
# Test with Lightning checkpoint (.ckpt)
python train.py --test \
  --method RelTR \
  --dataname VisualGenome \
  --ckpt_path outputs/runs/<method>/<ex_name>/checkpoints/best-epoch=*.ckpt \
  --gpus 0

# Test with PyTorch checkpoint (.pth / .pt)
python train.py --test \
  --method RelTR \
  --dataname VisualGenome \
  --ckpt_path outputs/pretrained/other/checkpoint0149.pth \
  --gpus 0
```

### Multi-GPU Training

```bash
# DDP training with 4 GPUs
python train.py --method RelTR --dataname VisualGenome --gpus 0 1 2 3 --dist

# Single-GPU mode (pass a list of one GPU to avoid DDP overhead)
python train.py --method RelTR --dataname VisualGenome --gpus 0
```

## Configuration

Configuration uses a two-layer system: **command-line arguments** override **per-method config files**.

### Config File Convention

```
configs/
├── _template.py                    # Template with all available options
└── <DatasetName>/
    ├── RelTR.py                    # RelTR config
    ├── EGTR.py                     # EGTR config
    └── HSTRNet.py                  # HSTRNet config
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Model method, e.g. RelTR / EGTR / HSTRNet / Motifs / IMP / REACT | — |
| `dataname` | Dataset (VisualGenome / OpenImageV6) | VisualGenome |
| `batch_size` | Training batch size | 4 |
| `dataset_size` | Optional train subset size for smoke tests | None |
| `val_dataset_size` | Optional val/test subset size for smoke tests | None |
| `test_dataset_size` | Optional test subset size; overrides `val_dataset_size` in test mode | None |
| `epoch` | Max training epochs | 200 |
| `lr` | Learning rate | 1e-4 |
| `lr_backbone` | Backbone learning rate | 1e-5 |
| `backbone` | CNN backbone (resnet50 / resnet101) | resnet50 |
| `hidden_dim` | Transformer hidden dimension | 256 |
| `enc_layers` | Number of encoder layers | 6 |
| `dec_layers` | Number of decoder layers | 6 |
| `num_entities` | Number of entity queries (RelTR) | 100 |
| `num_triplets` | Number of triplet queries (RelTR) | 200 |
| `entity_nums` | Number of entity classes (+ bg) | 151 |
| `rel_nums` | Number of predicate classes (+ bg) | 51 |
| `egtr_num_labels` | EGTR object logit classes without background | 150 |
| `egtr_num_rel_labels` | EGTR predicate logit classes without background | 50 |
| `egtr_sgdet_postprocess` | EGTR SGDet object postprocess: `query` or `qc_topk` | query |

### Evaluation Metrics

Metrics follow the standard SGG convention: `{task}_{metric}@{K}`

| Format | Example | Description |
|--------|---------|-------------|
| `sgdet_R@K` | `sgdet_R@50` | SGDet Recall@K |
| `sgdet_mR@K` | `sgdet_mR@50` | SGDet mean Recall@K (per-predicate average) |
| `sgcls_R@K` | `sgcls_R@20` | SGCLS Recall@K |
| `predcls_R@K` | `predcls_R@50` | PredCLS Recall@K |
| `predcls_mR@K` | `predcls_mR@20` | PredCLS mean Recall@K |

Configure in `metrics` field, e.g.: `["sgdet_R@50", "sgdet_mR@50", "predcls_R@20"]`

## Project Structure

```
OpenSGG/
├── train.py                          # Main entry point
├── configs/                          # Per-dataset, per-method configs
│   ├── _template.py                  # Template config
│   └── VisualGenome/
│       ├── RelTR.py
│       ├── EGTR.py
│       ├── HSTRNet.py
│       ├── Motifs.py / VCTree.py / TDE.py
│       └── IMP.py / Transformer.py / GPS_Net.py / PE_NET.py / REACT.py / SHA_GCL.py / SQUAT.py
├── src/                              # Core framework
│   ├── exp.py                        # Experiment class (train/test orchestration)
│   ├── loss.py                       # Loss backward-compat shim
│   ├── core/                         # Metrics, optimizers, schedulers
│   │   ├── metrics.py                # Unified evaluation pipeline
│   │   ├── optim_scheduler.py        # Optimizer/scheduler factory
│   │   ├── optim_constant.py         # Optimizer parameter grouping
│   │   └── drop_scheduler.py         # Drop path scheduling
│   ├── methods/                      # LightningModule wrappers
│   │   ├── base_method.py            # Base class (DDP eval, loss averaging)
│   │   ├── reltr_method.py           # RelTR wrapper
│   │   ├── egtr_method.py            # EGTR wrapper
│   │   ├── hstrnet_method.py         # HSTRNet wrapper
│   │   └── *_method.py               # Two-stage baseline wrappers
│   ├── models/                       # Model definitions
│   │   ├── reltr.py                  # RelTR model + criterion + postprocess
│   │   ├── HSTRNet.py                # HSTRNet model
│   │   └── motifs.py / imp.py / gpsnet.py / react_sgg.py / ...
│   ├── modules/                      # Reusable building blocks
│   │   ├── layers/                   # Backbone, Transformer, Matcher, CLIP alignment
│   │   └── egtr/                     # EGTR model components
│   │       ├── deformable_detr.py    # Deformable DETR backbone (3016 lines)
│   │       ├── egtr.py               # EGTR SGG head + loss
│   │       ├── util.py               # Box ops, NestedTensor, losses
│   │       ├── transform.py          # Data augmentation transforms
│   │       └── load_custom.py        # CUDA kernel loader
│   └── losses/                       # Loss function registry
│       ├── loss.py                   # HSTRCriterion + LOSS_FACTORY
│       └── reweight_loss.py          # Focal, class-balanced, and reweighting losses
├── data/                             # Data loading
│   └── dataloaders/
│       ├── dataloader.py             # Routing to dataset-specific loaders
│       ├── dataloader_VisualGenome.py # VisualGenome data loading
│       ├── coco.py                   # COCO dataset + build functions
│       ├── base_data.py              # LightningDataModule wrapper
│       ├── dataset_constant.py       # Per-dataset default params
│       └── egtr/                     # EGTR-specific data processing
├── lib/                              # Vendored libraries
│   ├── evaluation/
│   │   └── sg_eval.py               # Core recall computation (Danfei Xu's benchmark)
│   ├── fpn/
│   │   └── box_utils.py             # Bounding box utilities
│   └── pytorch_misc.py              # Intersection utilities
├── utils/                            # Utilities
│   ├── parser.py                     # Argument parsing
│   ├── main_utils.py                 # Config, env, throughput
│   ├── callbacks.py                  # Lightning callbacks
│   ├── box_ops.py                    # Box coordinate operations
│   ├── misc.py                       # NestedTensor, collate, distributed helpers
│   ├── config_utils.py               # Config file parser
│   └── transforms.py                 # DETR-style data transforms
├── scripts/                          # Utility scripts
│   ├── clip/                         # CLIP prototype/embedding builders
│   └── cluster/                      # Clustering utilities
└── outputs/                           # All run artifacts
    ├── pretrained/                    # Downloaded pretrained weights
    └── runs/                          # Experiment outputs (logs, ckpts, eval)
```

## Evaluation Modes

### SGDet (Scene Graph Detection)
Full end-to-end: predict bounding boxes, object labels, and predicates from raw images.

### SGCLS (Scene Graph Classification)
Given ground-truth bounding boxes, predict object labels and predicates.

### PredCLS (Predicate Classification)
Given ground-truth boxes and labels, predict predicates only.

EGTR currently reports PredCLS and SGDet through its compact evaluator cache.
SGCLS is skipped for EGTR unless a dedicated adapter is added.

## How to Add a New Method

1. Create your model under `src/models/`
2. Create a LightningModule wrapper under `src/methods/` (subclass `Base_method`)
3. Register in `src/methods/__init__.py` → `method_maps`
4. Add a config file at `configs/<Dataset>/<Method>.py`
5. (Optional) Add an evaluation adapter in `src/core/metrics.py`

## License

MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite the original papers for each method:

- **RelTR**: Cong et al., "RelTR: Relation Transformer for Scene Graph Generation", ECCV 2022
- **EGTR**: Im et al., "EGTR: Extracting Graph from Transformer for Scene Graph Generation", CVPR 2024
- **Neural Motifs**: Zellers et al., "Neural Motifs: Scene Graph Parsing with Global Context", CVPR 2018
- **IMP**: Xu et al., "Scene Graph Generation by Iterative Message Passing", CVPR 2017
