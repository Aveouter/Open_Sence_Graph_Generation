<div align="center">

# OpenSGG

**A research-grade Scene Graph Generation workspace for modern baselines, partial-label analysis, and reproducible evidence.**

[![Python](https://img.shields.io/badge/python-%E2%89%A43.10-2563eb.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-7c3aed.svg)](https://lightning.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-%E2%89%A511.3-16a34a.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-059669.svg)](LICENSE)

OpenSGG unifies classic two-stage SGG models, transformer/query baselines,
checkpoint-backed evaluation utilities, and a strict reproduction workflow in
one PyTorch Lightning codebase.

</div>

---

## Why This Repository Exists

Scene Graph Generation is no longer only a detector-plus-classifier benchmark.
Modern SGG research must reason about sparse predicate supervision, long-tail
mean recall, partial labels, retrieval-augmented supervision, and the difference
between a runnable adapter and an original-paper reproduction.

OpenSGG is organized around three goals:

| Goal | What OpenSGG Provides |
|---|---|
| Unified SGG experimentation | Standard SGDet, SGCls, and PredCls training/evaluation paths for classic and transformer methods. |
| Partial-label research | Tooling for predicate-set recovery, retrieval-style candidate analysis, and lattice/evidence-type method prototyping. |
| Reproduction discipline | Guardrails that separate checkpoint-backed reproduction from implementation audits, smoke tests, and deferred baselines. |

The active research direction is **lattice-constrained retrieval-augmented SGG**:
starting from PE-Net/RA-SGG-style relation embeddings and memory retrieval,
predicate candidates are promoted only when their structural relationship to
the observed predicate is reliable: entailment, compatibility, conflict,
granularity, and generic-predicate risk are treated explicitly.

This direction is a research roadmap, not a completed benchmark claim.

---

## Core Features

- **Broad method coverage**: RelTR, EGTR, FlowSG, USG-Par, Motifs, VCTree, TDE,
  IMP, Transformer, GPSNet, PE-Net, REACT, SHA-GCL, SQUAT, HSTRNet, and CVC.
- **Standard evaluator surface**: Recall@K, mean Recall@K, and head/body/tail
  predicate analysis across SGDet, SGCls, and PredCls.
- **Lightning training stack**: reproducible experiment setup, DDP support,
  checkpoint loading, and smoke-test-friendly dataset caps.
- **Research utilities**: analysis scripts for predicate priors, hidden-positive
  probes, relation export, checkpoint audits, and retrieval-oriented diagnostics.
- **Evidence-first docs**: baseline reports are labeled as reproduction,
  implementation audit, pipeline smoke, checkpoint unavailable, protocol
  mismatch, or deferred reproduction.

---

## Supported Methods

| Method | Venue | Family | CLI Name | Notes |
|---|---:|---|---|---|
| [RelTR](https://arxiv.org/abs/2201.11460) | ECCV 2022 | Query / end-to-end | `RelTR` | Triplet queries with set prediction. |
| [EGTR](https://arxiv.org/abs/2404.02072) | CVPR 2024 | Query / end-to-end | `EGTR` | Efficient graph extraction from DETR-style attention. |
| [FlowSG](https://arxiv.org/abs/2504.03180) | CVPR 2026 | End-to-end | `FlowSG` | Flow-matching SGG path with graph transformer components. |
| [USG-Par](https://arxiv.org/abs/2506.05582) | CVPR 2025 | Query / parser | `USG` | Universal Scene Graph Parser path. |
| [Neural Motifs](https://arxiv.org/abs/1711.06640) | CVPR 2018 | Two-stage | `Motifs` | LSTM object/context baseline. |
| [VCTree](https://arxiv.org/abs/1812.01880) | CVPR 2019 | Two-stage | `VCTree` | Tree-structured visual context baseline. |
| [TDE](https://arxiv.org/abs/2002.11949) | CVPR 2020 | Two-stage | `TDE` | Causal debiasing path built on Motifs-style context. |
| [IMP](https://arxiv.org/abs/1701.02426) | CVPR 2017 | Two-stage | `IMP` | Iterative Message Passing. |
| Transformer | CVPR 2020 | Two-stage | `Transformer` | Transformer context predictor. |
| GPS-Net | CVPR 2020 | Two-stage | `GPSNet` | Graph property sensing relation head. |
| [PE-Net](https://arxiv.org/abs/2303.07096) | CVPR 2023 | Prototype | `PENet` | Predicate prototype embedding baseline. |
| REACT | BMVC 2025 | Prototype | `REACT` | Prototype-regularized efficient SGG path. |
| [SHA-GCL](https://arxiv.org/abs/2203.15249) | CVPR 2022 | Debiasing | `SHAGCL` | Hybrid attention and group collaborative learning. |
| SQUAT | CVPR 2023 | Attention | `SQUAT` | Selective quad attention relation modeling. |
| HSTRNet | - | Prototype | `HSTRNet` | Hierarchical prototype relation learner. |
| CVC | - | Concept | `CVC` | Compositionally verified concept relation head. |

> A matching method name in OpenSGG is not automatically an official
> reproduction. Use the reproduction workflow before making paper-alignment
> claims.

---

## Active Research Direction

The current method-design thread is motivated by the gap between PE-Net/RA-SGG
retrieval and modern counterfactual/evidence-aware SGG work:

```text
observed predicate label
        +
relation memory retrieval
        +
predicate semantic lattice
        ->
structure-safe multi-label targets
```

The target contribution is a **lattice-constrained retrieval augmentation**
framework for partial predicate annotation:

- **Retrieval candidates** come from relation embeddings rather than pure text
  similarity.
- **Predicate lattice edges** distinguish entailment, compatibility, conflict,
  coarse/fine granularity, sibling families, and generic predicates.
- **Promotion scores** decide whether a retrieved candidate becomes a strong
  soft positive, weak soft positive, ignored candidate, or conflict penalty.
- **Coarse/fine preservation** prevents fine-grained candidates from erasing the
  observed coarse predicate.
- **Evaluation** should report standard R/mR, tail mR, no-graph-constraint
  recall, pseudo-label conflict rate, coarse/fine preservation, and generic
  predicate over-promotion.

This line is designed to improve RA-SGG-style pseudo-label reliability without
depending on unavailable dense visual-language provenance or overclaiming visual
relation understanding.

---

## Installation

```bash
git clone https://github.com/Aveouter/Open_Sence_Graph_Generation.git
cd Open_Sence_Graph_Generation

conda env create -f environment.yml
conda activate hsg

# Pick the wheel index that matches your CUDA runtime.
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

If mirror channels in `environment.yml` are unavailable, replace them with
`defaults` and `conda-forge`, or rely on your local `.condarc`.

| Dependency | Expected Version |
|---|---|
| Python | <= 3.10 |
| PyTorch | >= 1.10 |
| CUDA | >= 11.3 for GPU runs |
| PyTorch Lightning | 2.x |
| Transformers | >= 4.18 |

---

## Data Layout

Visual Genome is expected under `data/VisualGenome/`:

```text
data/VisualGenome/
|-- images/
|-- train.json
|-- val.json
|-- test.json
|-- rel.json
|-- predicate_frequencies.json        # optional, head/body/tail analysis
|-- clip_prototypes.pth               # optional, semantic/prototype analysis
```

OpenImageV6 experiments can be placed under `data/OpenImage/` with matching
COCO-style annotations.

---

## Quick Start

### Train

```bash
# Query / end-to-end methods
python train.py --method RelTR --dataname VisualGenome --gpus 0 --batch_size 4
python train.py --method EGTR  --dataname VisualGenome --gpus 0 --batch_size 4

# Two-stage methods
python train.py --method Motifs --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method PENet  --dataname VisualGenome --gpus 0 --batch_size 8
python train.py --method TDE    --dataname VisualGenome --gpus 0 --batch_size 8
```

Configs are loaded from `configs/<Dataset>/<Method>.py`; CLI arguments override
config values.

### Evaluate

```bash
python train.py --test --method EGTR --dataname VisualGenome \
  --ckpt_path outputs/pretrained/egtr/<run>/checkpoints/epoch=03-*.ckpt \
  --val_batch_size 1 --test_dataset_size 20 --num_workers 0 --gpus 0
```

### Multi-GPU

```bash
python train.py --method RelTR --dataname VisualGenome --gpus 0 1 2 3 --dist
```

---

## Metrics

| Mode | Given | Predicted |
|---|---|---|
| SGDet | raw image | boxes, object labels, predicates |
| SGCls | ground-truth boxes | object labels, predicates |
| PredCls | ground-truth boxes and object labels | predicates |

| Metric | Meaning |
|---|---|
| `R@K` | Recall over top-K predicted triplets. |
| `mR@K` | Mean recall across predicate classes. |
| Head/body/tail mR | Long-tail sensitivity by predicate frequency group. |
| No-graph-constraint recall | Allows multiple predicates for one subject-object pair. |

PredCls is useful for partial-label predicate analysis, but it is not evidence
of complete SGDet visual understanding.

---

## Reproduction Standard

OpenSGG uses a strict baseline reproduction standard. A baseline is reproduced
only when all of the following align with the original paper or official
repository:

- method implementation;
- checkpoint identity and provenance;
- configuration and preprocessing;
- inference flow;
- evaluator and metric semantics.

Random initialization, all-zero metrics, tiny-slice runs, partial checkpoint
remapping, and OpenSGG-compatible adapters are **not** reproduction evidence.

Start here:

- [Reproduction workflow](reproduction/README.md)
- [Usage guide](reproduction/USAGE.md)
- [Baseline tracker](docs/reproduction/README.md)
- [Evidence gates](docs/reproduction/evidence_gates.md)

Before opening reproduction-related PRs, run the guardrails listed in
`AGENTS.md` and `reproduction/USAGE.md`.

---

## Project Map

```text
OpenSGG/
|-- train.py                         # training and evaluation entrypoint
|-- configs/                         # dataset and method configs
|-- data/dataloaders/                # Visual Genome and dataset adapters
|-- src/
|   |-- exp.py                       # experiment orchestration
|   |-- methods/                     # LightningModule wrappers
|   |-- models/                      # model definitions
|   |-- modules/                     # reusable model blocks
|   |-- losses/                      # loss registry and criteria
|   `-- core/                        # metrics, optimizers, schedulers
|-- tools/
|   |-- analysis/                    # research and evaluation utilities
|   `-- reproduction/                # reproduction guardrails
|-- reproduction/                    # workflow templates and ADRs
|-- docs/
|   `-- reproduction/                # baseline reports and evidence gates
`-- outputs/                         # local run artifacts, not tracked
```

---

## Guides

- [Adding a new model](guides/adding_new_model.md)
- [Published method checkpoint evaluations](guides/published_method_reproduction.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

---

## Citation

If you use OpenSGG, please cite the framework and the original papers for the
methods you run.

```bibtex
@misc{opensgg,
  title  = {OpenSGG: Open Scene Graph Generation Framework},
  author = {Xinyu Liu and Xiaoguang Lin and contributors},
  year   = {2025},
  note   = {https://github.com/Aveouter/Open_Sence_Graph_Generation},
}
```

---

<div align="center">
  <sub>Built for careful SGG experimentation: runnable code, explicit evidence, and honest claims.</sub>
</div>
