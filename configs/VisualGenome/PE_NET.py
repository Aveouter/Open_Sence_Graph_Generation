# =========================
# PE-NET config for VisualGenome
# =========================
# Prototype-based Embedding Network for SGG (Zheng et al., CVPR 2023)
# Strictly aligned with official VL-Group/PENET:
#   https://github.com/VL-Group/PENET
#
# =========================================================================
# 预训练权重说明
# =========================================================================
#
# 训练时自动从 torchvision 加载（无需手动下载）：
#   1. 主干网络  ← ImageNet 预训练的 ResNeXt-101-32×8d (312 params)
#
# 以下模块从 kaiming_uniform_ 随机初始化、从头训练：
#   2. FPN 横向/平滑卷积    (16 params)
#   3. Box head fc6/fc7     (4096-dim, 68M params)
#   4. Union 特征提取器      (rect_conv + fc6/fc7, 68M params)
#   5. PENet 关系预测器      (118M params)
#
# -----------------------------------------------------------------------
# 如果要与官方完全一致（FPN + box-head 也从 COCO 预训练加载），需要
# 官方的 pretrained_faster_rcnn checkpoint：
#
#   penet_detector_ckpt = "checkpoints/pretrained_faster_rcnn/model_final.pth"
#
# 该文件来自 Scene-Graph-Benchmark，可从官方 PENet README 链接获取。
# 不设置此路径不影响训练 — FPN + box-head 会从 kaiming_init 开始学习。
# -----------------------------------------------------------------------
#
# Official hyperparameters (from configs/e2e_relation_X_101_32_8_FPN_1x.yaml
# and roi_relation_predictors.py):
#   mlp_dim              = 2048   (hardcoded)
#   context_hidden_dim   = 512    (CONTEXT_HIDDEN_DIM)
#   embed_dim            = 300    (GloVe 300d, PENET_EMBED_DIM)
#   pooling_dim          = 4096   (CONTEXT_POOLING_DIM)
#   obj_dim (visual_dim) = 4096   (MLP_HEAD_DIM, via FPN2MLPFeatureExtractor)
#   dropout              = 0.2    (PENET_DROPOUT)
#   batch_size_per_image = 512
#   positive_fraction    = 0.25
#   lr                   = 1e-3
#   steps                = (28000, 48000)
#   max_iter             = 60000
#   weight_decay         = 1e-4
#   grad_norm_clip       = 5.0
#   IMS_PER_BATCH        = 8
#   GLOVE_DIR            = ./datasets/vg/

method = 'penet'
loss = 'motifs_loss'

# ===== optimizer (official: SGD + WarmupMultiStepLR) =====
lr = 1e-3                       # BASE_LR
lr_backbone = 1e-5
weight_decay = 1e-4             # WEIGHT_DECAY
momentum = 0.9                  # official SGD momentum
clip_max_norm = 5.0             # GRAD_NORM_CLIP
opt = "sgd"
sched = "warmup_multi_step"
warmup_factor = 0.1             # official WARMUP_FACTOR
steps = [28000, 48000]          # official SOLVER.STEPS

# ===== training (official) =====
epoch = 50                      # approximately 60000 iter / steps-per-epoch
batch_size = 8                  # IMS_PER_BATCH
val_batch_size = 8              # TEST.IMS_PER_BATCH
max_iter = 60000                # SOLVER.MAX_ITER

# ===== pair sampling (official) =====
# Matches gtbox_relsample: BATCH_SIZE_PER_IMAGE=512, POSITIVE_FRACTION=0.25
penet_train_pairs = 512
penet_pos_frac = 0.25

# ===== model architecture (official) =====
hidden_dim = 2048               # mlp_dim
visual_dim = 4096               # obj_dim (MLP_HEAD_DIM via box feature extractor)
penet_embed_dim = 300           # GloVe 300d
penet_pooling_dim = 4096        # CONTEXT_POOLING_DIM
penet_context_hidden_dim = 512  # CONTEXT_HIDDEN_DIM
dropout = 0.2                   # PENET_DROPOUT

# ===== GloVe (official: GLOVE_DIR ./datasets/vg/) =====
# Download glove.6B.300d.txt from https://nlp.stanford.edu/data/glove.6B.zip
glove_dir = None

# ===== frequency bias =====
# Official training script sets PREDICT_USE_BIAS True, but
# PrototypeEmbeddingNetwork does NOT use it internally.
use_freq_bias = False
freq_bias_eps = 1e-12

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151
rel_nums = 51

# ===== backbone (official: ResNeXt-101-32x8d) =====
backbone_arch = "resnext101_32x8d"
backbone_pretrained = True
backbone_frozen = True
use_backbone = True
roi_output_size = 7

# 官方 COCO-pretrained Faster R-CNN detector checkpoint:
#   checkpoints/pretrained_faster_rcnn/model_final.pth
# 提供 backbone + FPN + box-head fc6/fc7 的 COCO 预训练权重。
# 不设置时：backbone, FPN 自动从 torchvision 加载；box-head 用 kaiming_init。
penet_detector_ckpt = None

# ===== SGDet NMS (official LATER_NMS_PREDICTION_THRES) =====
penet_nms_thresh = 0.5

# ===== evaluation =====
eval_mode = 'predcls'
metrics = ["R@50", "R@100", "mR@50", "mR@100"]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
