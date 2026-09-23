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
| [FlowSG](https://arxiv.org/abs/2604.18623) | CVPR 2026 | End-to-end | `FlowSG` | Flow matching with CLIP ViT-B/16, slotwise VQ-VAE, and DiT-style graph transformer |
| [USG-Par](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Universal_Scene_Graph_Generation_CVPR_2025_paper.html) | CVPR 2025 | End-to-end | `USG` | Universal Scene Graph Parser — learnable queries + RPC + transformer decoder |
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

### 2025–2026 literature scope and code filter

**Last checked: 2026-09-22.** This survey covers static-image scene graph
generation on **Visual Genome (VG150)** and **Open Images** (the `OpenImageV6`
dataset in this repository). It screens CVPR, ICCV, ECCV, ICLR, ICML, NeurIPS,
AAAI, IJCAI and ACM MM main papers, and TPAMI, IJCV, TIP and TMM articles;
JAS is included as a related journal. The date refers to the publication venue,
not the first arXiv upload. This is a dated search snapshot, with unresolved
sources listed explicitly, rather than a claim of complete coverage of 2026.

The methods below have public author GitHub repositories containing source
code, not just a project page or a promise to release code. Source availability
does **not** establish checkpoint availability, complete implementation of every
paper variant, or OpenSGG reproduction. These methods are literature references;
they are not additions to the CLI's Supported Methods.

| Method | Publication | Author GitHub | VG / Open Images evidence |
|---|---|---|---|
| HQSG | [CVPR 2025][hqsg-paper] | [HQSG](https://github.com/fujiawei0724/HQSG) | VG SGDet, Table 1; no OI table in the inspected paper |
| RA-SGG | [AAAI 2025][rasgg-paper] | [torch-rasgg](https://github.com/KanghoonYoon/torch-rasgg) | VG three tasks, Table 1; no OI table in the inspected paper |
| RAHP | [AAAI 2025][rahp-paper] | [RAHP](https://github.com/Leon022/RAHP) | VG open-vocabulary Tables 1/3; OI V6 Table 2 |
| NoDIS | [ICML 2025][nodis-paper] | [NoDIS](https://github.com/gavin-gqzhang/NoDIS) | VG Table 1; OI V4/V6 Table 2 |
| VL-IRM | [ICCV 2025][vlirm-paper] | [VL-IRM](https://github.com/myukzzz/VL-IRM) | VG open-vocabulary Tables 1/2/4; OI V6 Table 3 |
| ACC | [NeurIPS 2025][acc-paper] | [ACC](https://github.com/HKUST-LongGroup/ACC) | VG open-vocabulary Tables 1/2; filtered test set |
| MCL | [TIP 2025][mcl-paper] | [G-USGG](https://github.com/XinyuLyu/G-USGG) | VG/OI stated by authors; final journal tables not retrieved; numbers withheld |
| GCM | [TPAMI, online 2025 / issue 2026][gcm-paper] | [GCM](https://github.com/Nora-Zhang98/GCM) | VG/OI V6 code and instructions; final journal tables not retrieved; numbers withheld |
| APT | [ICLR 2026][apt-paper] | [APT](https://github.com/CGCL-codes/APT) | VG Tables 2/3; OI V6 Tables 8/9; component code released, integration and OI protocol need audit |
| HSGG | [ICML 2026][hsgg-paper] | [HSGG](https://github.com/lllyz789/HSGG) | VG150 in author code; primary PDF retrieval blocked; numbers withheld |
| BiRef | [JAS 2026][biref-paper] | [BiRef](https://github.com/gavin-gqzhang/BiRef) | VG/OI stated by journal; numerical full text not retrieved; numbers withheld |

See the [source and exclusion ledger](guides/sgg_literature_2025_2026.md)
for paper titles, implementation evidence, search coverage, missing results and
excluded candidates, including README-only repositories and code mismatches.

All following values are **paper-reported percentages**, not local evaluation
results. `a / b` means `@50 / @100`; `NR` means not reported in the cited table.
Different detectors, supervision, splits and postprocessing prevent treating
these tables as a common leaderboard. We retain each paper's precision.

### Closed-vocabulary VG150: SGDet

| Method / paper variant | Detector or architecture | R@50 / 100 | mR@50 / 100 | Original source |
|---|---|---|---|---|
| HQSG | One-stage, ResNet-101 | 34.1 / 38.3 | 16.0 / 20.5 | [Table 1][hqsg-paper] |
| RA-SGG | PE-Net, Faster R-CNN / ResNeXt-101-FPN | 26.0 / 30.3 | 14.4 / 17.1 | [Table 1][rasgg-paper] |
| VTransE + NoDIS | Two-stage, Faster R-CNN | 27.96 / 31.97 | 13.77 / 15.83 | [Table 1][nodis-pdf] |
| Motifs + NoDIS | Two-stage, Faster R-CNN | 30.91 / 35.42 | 13.58 / 16.35 | [Table 1][nodis-pdf] |
| Transformer + NoDIS | Two-stage, Faster R-CNN | 25.26 / 28.77 | 16.91 / 19.25 | [Table 1][nodis-pdf] |
| PE-Net + NoDIS | Two-stage, Faster R-CNN | 23.16 / 26.95 | 14.78 / 17.16 | [Table 1][nodis-pdf] |
| Motif + APT | Paper's two-stage variant | 33.3 / 37.6 | 9.2 / 10.3 | [Table 2][apt-paper] |
| HQSG + APT | Paper's one-stage variant | 36.5 / 39.9 | 18.2 / 21.7 | [Table 2][apt-paper] |

APT rows are representative variants from its own table. Released prompt
components do not establish that all these integrations can be run from the
repository. See the [APT caveats](guides/sgg_literature_2025_2026.md#apt).

### Closed-vocabulary VG150: PredCls and SGCls

| Method / variant | PredCls R@50 / 100 | PredCls mR@50 / 100 | SGCls R@50 / 100 | SGCls mR@50 / 100 | Source |
|---|---|---|---|---|---|
| RA-SGG | 62.2 / 64.1 | 36.2 / 39.1 | 38.2 / 39.1 | 20.9 / 22.5 | [Table 1][rasgg-paper] |
| VTransE + NoDIS | 57.83 / 59.74 | 35.68 / 38.04 | 38.84 / 39.96 | 20.83 / 22.14 | [Table 1][nodis-pdf] |
| Motifs + NoDIS | 62.72 / 64.98 | 33.83 / 36.48 | 36.46 / 37.48 | 22.68 / 24.35 | [Table 1][nodis-pdf] |
| Transformer + NoDIS | 49.96 / 53.21 | 37.25 / 39.97 | 34.42 / 35.63 | 21.53 / 23.35 | [Table 1][nodis-pdf] |
| PE-Net + NoDIS | 50.13 / 53.87 | 38.72 / 41.93 | 30.65 / 32.19 | 22.31 / 23.69 | [Table 1][nodis-pdf] |
| Motif + APT | 66.5 / 68.2 | 17.4 / 18.1 | 40.3 / 40.8 | 10.5 / 11.1 | [Table 2][apt-paper] |
| HQSG + APT | 58.7 / 61.2 | 35.1 / 37.3 | 37.7 / 38.3 | 21.9 / 22.7 | [Table 2][apt-paper] |

RA-SGG's author implementation uses relation NMS for PredCls/SGCls. Do not
compare these tasks with SGDet, which must also predict the object boxes.

### Open-vocabulary VG150: unseen relations (OvR)

`Total` evaluates base + novel predicates; `Novel` evaluates novel predicates.
The split source and task are part of each result's identity. Equal novel
percentages alone do not establish identical category lists or evaluation sets.

| Method / variant | Task; novel predicate split | Total R@50 / 100 | Total mR@50 / 100 | Novel R@50 / 100 | Novel mR@50 / 100 | Source |
|---|---|---|---|---|---|---|
| SGTR† + RAHP | PredCls; EPIC, 30% | 39.92 / 46.03 | 16.88 / 22.18 | 15.46 / 20.37 | 11.82 / 15.46 | [Table 1][rahp-paper] |
| PE-Net + RAHP | PredCls; EPIC, 30% | 64.70 / 69.11 | 24.50 / 28.25 | 20.79 / 29.00 | 15.70 / 23.73 | [Table 1][rahp-paper] |
| OvSGTR + RAHP | SGDet; OvSGTR, 30% | 21.50 / 25.74 | 4.51 / 5.37 | 15.59 / 19.92 | 3.01 / 4.04 | [Table 1][rahp-paper] |
| VL-IRM (Ours) | SGDet; PGSG, 50% | 13.7 / 19.7 | 7.5 / 10.6 | 10.1 / 14.3 | 5.2 / 7.5 | [Table 1][vlirm-paper] |
| VL-IRM (Ours*) | SGDet; OvSGTR, 30% | 14.3 / 20.4 | 8.7 / 12.7 | 9.0 / 12.6 | NR | [Table 1][vlirm-paper] |
| OvSGTR + VL-IRM (Ours*) | SGDet; OvSGTR, 30%; extra pretraining | 21.1 / 25.0 | 4.2 / 4.9 | 14.2 / 17.7 | NR | [Table 1][vlirm-paper] |
| VL-IRM | PredCls; PGSG, 50% | 27.8 / 37.5 | 11.1 / 14.5 | 11.9 / 18.3 | 5.9 / 8.2 | [Table 2][vlirm-paper] |
| ACC, Swin-T | SGDet; 30%; ACC filtered test set | 23.22 / 27.40 | NR | 17.89 / 21.70 | NR | [Table 1][acc-paper] |
| ACC, Swin-B | SGDet; 30%; ACC filtered test set | 24.81 / 29.28 | NR | 20.04 / 24.66 | NR | [Table 1][acc-paper] |

RAHP's `†` denotes its adapted SGTR variant. ACC evaluates 14,700 VG test
images after removing overlap with GroundingDINO pretraining data. Its results
must not be compared directly with full-test-set rows. Use ACC's NeurIPS paper;
the repository's older INOVA link is not the source of these numbers.

### Open-vocabulary VG150: unseen objects and relations (OvD+R)

| Method / variant | Total R@50 / 100 | Novel-object R@50 / 100 | Novel-relation R@50 / 100 | Source |
|---|---|---|---|---|
| VS3 + RAHP, Swin-T | 12.66 / 15.39 | 13.01 / 14.82 | 3.75 / 5.12 | [Table 3][rahp-paper] |
| OvSGTR + RAHP, Swin-T | 13.83 / 16.52 | 12.45 / 15.38 | 13.31 / 16.46 | [Table 3][rahp-paper] |
| OvSGTR-T + VL-IRM | 14.87 / 18.27 | NR | 12.27 / 15.26 | [Table 4][vlirm-paper] |
| ACC, Swin-T, filtered test set | 17.43 / 21.27 | 17.16 / 21.10 | 15.90 / 19.46 | [Table 2][acc-paper] |
| ACC, Swin-B, filtered test set | 18.88 / 23.19 | 18.84 / 23.29 | 17.50 / 21.73 | [Table 2][acc-paper] |

### Historical SGDet reference (before 2025)

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

This historical table is retained for context; it is outside the 2025–2026
selection window.

## Open Images Paper Results

### Closed-vocabulary SGDet: weighted mAP protocol

The weighted score is `0.2 × R@50 + 0.4 × wmAP_rel + 0.4 × wmAP_phr`.
Relation detection matches both object boxes; phrase detection matches their
union box. OI V4 and V6 are different benchmarks and are kept in separate rows.

| Method | Dataset | R@50 | mR@50 | wmAP_rel | wmAP_phr | Weighted score | Source |
|---|---|---|---|---|---|---|---|
| Transformer + NoDIS | OI V4 | 74.84 | 70.34 | 38.21 | 42.35 | 47.19 | [Table 2][nodis-pdf] |
| Transformer + NoDIS | OI V6 | 74.11 | 48.93 | 38.87 | 38.95 | 45.95 | [Table 2][nodis-pdf] |

### Open-vocabulary OI V6: recall protocol

Both papers follow the PGSG split for OI; these are recall results, not weighted
mAP scores. The RAHP adaptation and VL-IRM use different architectures and
training, so the shared split is not an otherwise controlled comparison.

| Method / variant | Task | Total R@50 / 100 | Total mR@50 / 100 | Novel R@50 / 100 | Novel mR@50 / 100 | Source |
|---|---|---|---|---|---|---|
| SGTR† + RAHP, ResNet-101 | SGDet | 62.42 / 64.86 | 30.79 / 34.46 | 49.61 / 56.28 | 29.43 / 34.16 | [Table 2][rahp-paper] |
| VL-IRM, GLIP | SGDet | 46.3 / 59.4 | 19.8 / 24.6 | 14.6 / 18.7 | 4.5 / 10.1 | [Table 3][vlirm-paper] |

### APT's OI V6 appendix: separate protocol pending clarification

**Protocol mismatch:** APT §4.1 specifies 288 entity classes, 30 relations and
5,322 test images; Appendix B specifies 301 objects, 31 predicates and 6,322
test images. Its Table 8 reports recall rather than weighted mAP. Preserve the
published values below as a separate literature record until this discrepancy
and the released integrations are clarified.

| Paper variant | Task | R@50 / 100 | mR@50 / 100 | Source |
|---|---|---|---|---|
| Motif + APT | PredCls | 67.0 / 68.6 | 17.6 / 18.5 | [Table 8][apt-paper] |
| Motif + APT | SGCls | 40.5 / 41.0 | 10.6 / 11.3 | [Table 8][apt-paper] |
| Motif + APT | SGDet | 33.7 / 38.0 | 9.3 / 10.5 | [Table 8][apt-paper] |
| SGTR + APT | SGDet | 33.7 / 37.0 | 13.1 / 15.0 | [Table 8][apt-paper] |

MCL, GCM and BiRef have public code and OI-related author evidence, but verified
final-paper numbers are still missing from this survey. Missing data is not a
zero score. No results here are labeled as an OpenSGG baseline reproduction.

[hqsg-paper]: https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Hybrid_Reciprocal_Transformer_with_Triplet_Feature_Alignment_for_Scene_Graph_CVPR_2025_paper.pdf
[rasgg-paper]: https://ojs.aaai.org/index.php/AAAI/article/download/33036/35191
[rahp-paper]: https://ojs.aaai.org/index.php/AAAI/article/download/32594/34749
[nodis-paper]: https://proceedings.mlr.press/v267/zhang25ak.html
[nodis-pdf]: https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhang25ak/zhang25ak.pdf
[vlirm-paper]: https://openaccess.thecvf.com/content/ICCV2025/papers/Min_Vision-Language_Interactive_Relation_Mining_for_Open-Vocabulary_Scene_Graph_Generation_ICCV_2025_paper.pdf
[acc-paper]: https://proceedings.neurips.cc/paper_files/paper/2025/file/f7b118ed1bfd2a9f366d55021a8bc1e0-Paper-Conference.pdf
[mcl-paper]: https://doi.org/10.1109/TIP.2025.3540296
[gcm-paper]: https://doi.org/10.1109/TPAMI.2025.3635152
[apt-paper]: https://proceedings.iclr.cc/paper_files/paper/2026/file/23bce04d18001e4b4eee05aba89706ee-Paper-Conference.pdf
[hsgg-paper]: https://icml.cc/virtual/2026/poster/65975
[biref-paper]: https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2026.125786

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

- [2025–2026 SGG Literature Audit](guides/sgg_literature_2025_2026.md) — primary sources, public-code screening, VG/Open Images protocol caveats, and unresolved candidates.
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

- **Validation** — AST registration checks plus the unit test suite
- **Smoke tests** — forward pass + loss convergence on all 17 methods with synthetic data
- **LLM code review** — automated review via Claude/DeepSeek API on changed files

Trigger paths include code, configs, tools, tests, reproduction guardrails,
reproduction reports, `AGENTS.md`, `CONTRIBUTING.md`, `CONTEXT.md`, and
workflow files.

The test suite includes `tools/reproduction/check_repository_boundary.py`, which
keeps research-only work out of this repository. See
[AGENTS.md](AGENTS.md#repository-boundary) for the rule.

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
