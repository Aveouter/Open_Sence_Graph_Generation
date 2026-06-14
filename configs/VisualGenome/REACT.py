# =========================
# REACT config for VisualGenome
# =========================
# REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in SGG
# (Neau et al., BMVC 2025)

method = 'REACT'
loss = 'react_loss'

# ===== optimizer =====
lr = 1e-3
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 5.0

# ===== training =====
epoch = 50
batch_size = 8
val_batch_size = 8

# ===== model architecture =====
hidden_dim = 512          # mlp_dim in paper
visual_dim = 2048
dropout = 0.1
react_embed_dim = 200     # GloVe word embedding dimension
react_use_union = True    # Union feature debiasing
react_text_only = False   # Ablation: text-only mode

# ===== prototype regularization =====
react_l21_weight = 1.0    # L21 semantic matrix sparsity loss weight
react_dist_weight = 1.0   # Prototype distance/separation loss weight

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-12

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151
rel_nums = 51

# ===== backbone =====
backbone_arch = "resnet50"
backbone_pretrained = True
backbone_frozen = True
use_backbone = True
roi_output_size = 7

# ===== evaluation =====
eval_mode = 'predcls'
metrics = ["R@50", "R@100", "mR@50", "mR@100"]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
