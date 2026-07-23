<div align="center">

# OpenSGG — Open Scene Graph Generation

**Modular PyTorch Lightning framework for Scene Graph Generation.**
Unifies 17 SGG methods under a single training and evaluation harness.

[![Python](https://img.shields.io/badge/python-≤3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.10-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-≥11.3-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF.svg)](https://github.com/Aveouter/Open_Sence_Graph_Generation/actions)
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
git clone https://github.com/Aveouter/Open_Sence_Graph_Generation.git
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

### Docker

For a containerized GPU setup, see [Docker Environment](guides/docker.md).

```bash
docker build -t opensgg:dev .
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/workspace/OpenSGG \
  -v "$PWD/data":/workspace/OpenSGG/data \
  -v "$PWD/outputs":/workspace/OpenSGG/outputs \
  opensgg:dev bash
```

For a Compose workflow, use `docker compose run --rm opensgg bash` for CPU
checks or add `-f compose.gpu.yaml` for GPU runs.

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
2. Place images under `data/VisualGenome/images/`
3. Provide COCO-format `train.json`, `val.json`, `test.json`, and `rel.json`
   under `data/VisualGenome/`
4. Optional analysis assets such as `clip_prototypes.pth` and
   `predicate_frequencies.json` can live in the same directory

### OpenImageV6 (Experimental)

Place COCO-format annotations and images under `data/OpenImage/`.

---

## Quick Start

### Training

```bash
python train.py --method EGTR --dataname VisualGenome --gpus 0 --batch_size 4
```

Replace `EGTR` with any supported CLI method name. Config files are auto-loaded
from `configs/<Dataset>/<Method>.py`, and CLI arguments override config values.

### Evaluation

```bash
python train.py --test --method EGTR --dataname VisualGenome \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-*.ckpt --gpus 0
```

### Multi-GPU

```bash
# DDP with 4 GPUs
python train.py --method RelTR --dataname VisualGenome --gpus 0 1 2 3 --dist

# Single GPU (no DDP overhead)
python train.py --method RelTR --dataname VisualGenome --gpus 0
```

---

## VisualGenome Paper Results

The table below summarizes paper-reported Visual Genome **SGDet** results, not
OpenSGG reproduction results. Values are percentages under graph-constraint
evaluation when reported by the cited paper table. Source:
[EGTR Table 1](https://arxiv.org/html/2404.02072v4), with each row attributed
to the corresponding original method paper. In that table, `LA` denotes the
logit adjustment setting proposed by SSR-CNN.

| Method | Paper / Venue | Detector / backbone in paper | AP | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|--------|---------------|------------------------------|----|------|------|-------|-------|-------|--------|
| IMP | CVPR 2017 | Faster R-CNN / ResNeXt-101-FPN | 28.1 | 18.1 | 25.9 | 31.2 | 2.8 | 4.2 | 5.4 |
| Neural Motifs | CVPR 2018 | Faster R-CNN / ResNeXt-101-FPN | 28.1 | 25.1 | 32.1 | 36.9 | 4.1 | 5.5 | 6.8 |
| VCTree | CVPR 2019 | Faster R-CNN / ResNeXt-101-FPN | 28.1 | 24.8 | 31.8 | 36.1 | 4.9 | 6.6 | 7.7 |
| VCTree-TDE | CVPR 2020 | Faster R-CNN / ResNeXt-101-FPN | 28.1 | 14.0 | 19.4 | 23.2 | 6.9 | 9.3 | 11.1 |
| GPS-Net | CVPR 2020 | Faster R-CNN / ResNeXt-101-FPN | - | - | 31.1 | 35.9 | - | 6.7 | 8.6 |
| RelTR | TPAMI 2023 | DETR-50 | 26.4 | 21.2 | 27.5 | - | 6.8 | 10.8 | - |
| EGTR | CVPR 2024 | Deformable DETR-50 | 30.8 | 23.5 | 30.2 | 34.3 | 5.5 | 7.9 | 10.1 |
| EGTR + logit adjustment | CVPR 2024 | Deformable DETR-50 | 30.8 | 15.7 | 18.7 | 20.5 | 12.1 | 17.8 | 21.7 |

For PredCls / SGCls and method-specific unbiased-SGG tables, consult the
original papers. This README table is a compact literature reference only.

---

## Configuration

Configs use a two-layer system: **CLI arguments** override **per-method config files**.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Model method | `HSTRNet` |
| `dataname` | Dataset (`VisualGenome` / `OpenImageV6`) | `VisualGenome` |
| `batch_size` | Training batch size | `4` |
| `dataset_size` | Train subset size (smoke tests) | `None` |
| `epoch` | Max training epochs | `200` |
| `no_progress_bar` | Disable train/validation/test progress bars | `False` |
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

- [Published Method Checkpoint Evaluations](guides/published_method_reproduction.md) — concise PR-ready summary of traced checkpoint-backed OpenSGG evaluation results for published paper methods.
- [Adding A New Model](guides/adding_new_model.md) — step-by-step contributor guide for integrating a new method into the main training and evaluation pipeline.
- [Reproduction Workflow Usage](reproduction/USAGE.md) — evidence-first workflow for deciding whether a baseline is reproduced, audited, smoke-tested, or deferred.
- [Baseline Reproduction Workflow](reproduction/README.md) — current reproduction standard, deferred audit matrix, and evidence gates.
- [Contributing](CONTRIBUTING.md) — PR expectations, validation commands, and repository hygiene rules.
- [Security Policy](SECURITY.md) — private reporting path for dependency, checkpoint-loading, credential, and CI-token issues.

---

## Community Resources

- [ChocoWu/Awesome-Scene-Graph-Generation](https://github.com/ChocoWu/Awesome-Scene-Graph-Generation) — friendly link and broad community index for scene graph datasets, papers, toolkits, workshops, surveys, and applications.
- [Scene Graph Series](https://scene-graph.github.io/) — project hub for scene graph research updates and community activities.
- [Scene Graph for Structured Intelligence Workshop](https://scene-graph.github.io/SG4SI-WACV26/) — WACV 2026 workshop page for structured scene understanding.
- [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) — widely used SGG baseline and evaluation reference.
- [SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark) — maintained benchmark implementation for common SGG methods.
- [SGG-Annotate](https://github.com/Maelic/SGG-Annotate) — COCO-format visual relationship annotation tool.

---

## CI/CD Pipeline

GitHub Actions runs on every PR and push to `main`:

- **Smoke tests** — forward pass + loss convergence on all 17 methods with synthetic data
- **LLM code review** — automated review via Claude/DeepSeek API on changed files

Trigger paths include code, configs, tools, reproduction guardrails, reproduction
reports, `AGENTS.md`, and workflow files.

PRs that touch reproduction claims must preserve the status language in
`AGENTS.md` and `reproduction/README.md`; random-init, tiny-slice, all-zero, or
partial-checkpoint outputs are not baseline reproduction evidence.

---

## How to Add a New Method

See [Adding A New Model](guides/adding_new_model.md) for the complete model,
method wrapper, config, registry, CLI, metric, and smoke-test checklist.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

To be added.

Please also cite the original papers for any methods you use: RelTR (ECCV 2022), EGTR (CVPR 2024), FlowSG (CVPR 2026), USG-Par (CVPR 2025), Neural Motifs (CVPR 2018), VCTree (CVPR 2019), TDE (CVPR 2020), IMP (CVPR 2017), Transformer (CVPR 2020), GPS-Net (CVPR 2020), PE-Net (CVPR 2023), REACT (BMVC 2025), SHA-GCL (CVPR 2022), SQUAT (CVPR 2023).

---

<div align="center">
  <sub>Built by the CIGIT HPC Lab team and open-source contributors.</sub>
</div>
