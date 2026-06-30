# TDE Reproduction: Official Sources

Last updated: 2026-06-22.

## Scope

This file records the authoritative sources for reproducing Total Direct Effect
(TDE) from Tang et al., "Unbiased Scene Graph Generation from Biased Training".
It is intentionally evidence-oriented: every later implementation step must be
traceable to a paper section, official repository file, official checkpoint, or
explicitly documented deviation.

## Paper

- Title: Unbiased Scene Graph Generation from Biased Training
- Authors: Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, Hanwang Zhang
- Venue: CVPR 2020, Oral
- arXiv: https://arxiv.org/abs/2002.11949
- CVF PDF: https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Unbiased_Scene_Graph_Generation_From_Biased_Training_CVPR_2020_paper.pdf
- Core equation: TDE is defined as `Y_x(u) - Y_{\bar{x}, z}(u)` and the final unbiased predicate logits are `y_e - y_e(\bar{x}, z_e)`.

## Official Repository

- Repository: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
- Repository description: PyTorch Scene Graph Benchmark and official implementation for the CVPR 2020 TDE paper.
- Inspected commit: `ceb71fa88461c2a97a6258a80f47669d89207296`
- Persistent local checkout:
  `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
- Inspection command:

```bash
git ls-remote https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git HEAD refs/heads/master
git clone --depth 1 https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git /tmp/tde_official/Scene-Graph-Benchmark.pytorch
cd /tmp/tde_official/Scene-Graph-Benchmark.pytorch
git rev-parse HEAD
```

Persistent checkout command used for follow-up reproduction:

```bash
mkdir -p /workspace/external/tde_official
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git \
  /workspace/external/tde_official/Scene-Graph-Benchmark.pytorch
cd /workspace/external/tde_official/Scene-Graph-Benchmark.pytorch
git checkout ceb71fa88461c2a97a6258a80f47669d89207296
git rev-parse HEAD
```

Observed persistent checkout hash:

```text
ceb71fa88461c2a97a6258a80f47669d89207296
```

## Official Checkpoints

The official README states that the original paper models are lost and provides
freshly trained causal MOTIFS-SUM checkpoints. For checkpoint-based
reproduction, these downloadable checkpoints should be the primary target, while
paper Table 1 remains a historical reference.

| Protocol | Official checkpoint URL | Context layer | Fusion | Test effect |
|---|---|---|---|---|
| SGDet | https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602 | motifs | sum | TDE |
| SGCls | https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s | motifs | sum | TDE |
| PredCls | https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws | motifs | sum | TDE |

The official repository also gives alternate Baidu and Weiyun links for
pretrained models and dataset annotations:

- Baidu: https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA, extraction code `1234`
- Weiyun: https://share.weiyun.com/ViTWrFxG

## Official Environment Requirements

Official `INSTALL.md` requirements:

- Python `<= 3.8`
- PyTorch `>= 1.2`; author environment was PyTorch `1.4.0` with CUDA `10.1`
- torchvision `>= 0.4`; author environment was torchvision `0.5.0` with CUDA `10.1`
- cocoapi
- yacs
- matplotlib
- GCC `>= 4.9`
- OpenCV
- Apex built against the selected PyTorch/CUDA stack

The current OpenSGG runtime is not suitable for official-code reproduction,
even though it is suitable for the current OpenSGG project:

- Active project env requested by user: `hsg`
- Python: `3.10.8`
- Torch: `2.5.1+cu121`
- torchvision: `0.20.1+cu121`
- CUDA availability: available, `NVIDIA A100-PCIE-40GB`

The official TDE repository targets the older `maskrcnn-benchmark` stack
around Python 3.7, PyTorch 1.4.0, torchvision 0.5.0, and CUDA 10.1. Therefore
official reproduction must use an isolated official environment rather than
installing old dependencies into `hsg`.

## Dataset Requirements

Official Visual Genome setup:

- Images:
  - https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
  - https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
- Expected image directory: `datasets/vg/VG_100K`
- Scene graph annotations:
  - Official OneDrive link in `DATASET.md` for `VG-SGG-with-attri.h5`
  - Expected annotation file: `datasets/vg/VG-SGG-with-attri.h5`
  - Dictionary file included in repo: `datasets/vg/VG-SGG-dicts-with-attri.json`
- Dataset split/protocol:
  - Visual Genome 150 object classes and 50 predicate classes
  - Standard PredCls, SGCls, SGDet protocols controlled by:
    - `MODEL.ROI_RELATION_HEAD.USE_GT_BOX`
    - `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL`

## Official Configuration

Primary config:

- `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`

Key settings for causal MOTIFS-SUM TDE:

```yaml
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  WEIGHT: "catalog://ImageNetPretrained/FAIR/20171220/X-101-32x8d"
  BACKBONE:
    CONV_BODY: "R-101-FPN"
  RELATION_ON: True
  ATTRIBUTE_ON: False
  ROI_BOX_HEAD:
    NUM_CLASSES: 151
    MLP_HEAD_DIM: 4096
  ROI_RELATION_HEAD:
    NUM_CLASSES: 51
    BATCH_SIZE_PER_IMAGE: 1024
    CONTEXT_POOLING_DIM: 4096
    CONTEXT_HIDDEN_DIM: 512
    PREDICTOR: "CausalAnalysisPredictor"
    CAUSAL:
      EFFECT_TYPE: "none"
      FUSION_TYPE: "sum"
      CONTEXT_LAYER: "motifs"
      SPATIAL_FOR_VISION: True
      EFFECT_ANALYSIS: True
```

## Official Training Commands

Training uses `EFFECT_TYPE none`; TDE is only applied at inference.

PredCls causal MOTIFS-SUM training template:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
  MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
  MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
  SOLVER.IMS_PER_BATCH 12 \
  TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR /path/to/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT /path/to/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR /path/to/causal-motifs-predcls
```

For SGCls, set `USE_GT_BOX True` and `USE_GT_OBJECT_LABEL False`.
For SGDet, set both to `False`.

## Official Inference Commands

Inference switches `EFFECT_TYPE` from `none` to `TDE`.

PredCls causal MOTIFS-SUM TDE evaluation template:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
  MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
  MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
  MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
  TEST.IMS_PER_BATCH 1 \
  DTYPE "float16" \
  GLOVE_DIR /path/to/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT /path/to/causal-motifs-predcls \
  OUTPUT_DIR /path/to/causal-motifs-predcls
```

For SGCls and SGDet, adjust the two protocol switches as above and point
`MODEL.PRETRAINED_DETECTOR_CKPT`/`OUTPUT_DIR` to the matching checkpoint
directory.

## Target Results

### Paper Table 1: MOTIFS-SUM TDE mR

The paper's main RR table reports mean Recall, not conventional Recall.

| Protocol | mR@20 | mR@50 | mR@100 |
|---|---:|---:|---:|
| PredCls | 18.5 | 25.5 | 29.1 |
| SGCls | 9.8 | 13.1 | 14.9 |
| SGDet | 5.8 | 8.2 | 9.8 |

### Official README Fresh Checkpoint Results

These are the most appropriate acceptance targets for the downloadable official
checkpoints.

| Protocol | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGDet TDE | 11.92 | 16.56 | 20.15 | 6.58 | 8.94 | 10.99 | 1.54 | 2.33 | 3.03 |
| SGCls TDE | 20.47 | 26.31 | 28.79 | 9.80 | 13.21 | 15.06 | 1.91 | 2.95 | 4.10 |
| PredCls TDE | 33.38 | 45.88 | 51.25 | 17.85 | 24.75 | 28.70 | 8.28 | 14.31 | 18.04 |

## Reproduction Rule

Do not treat a current-framework run as a TDE reproduction until all of the
following are true:

1. The matching official checkpoint has been downloaded and hashed.
2. The checkpoint has been evaluated in the official repository.
3. Official-repo metrics match the official README target within the agreed
   tolerance or the gap is explained.
4. The current-framework implementation loads the same checkpoint or a
   documented, deterministic conversion of it.
5. Current-framework metrics match the official-repo metrics under the same
   dataset/protocol/evaluator.
