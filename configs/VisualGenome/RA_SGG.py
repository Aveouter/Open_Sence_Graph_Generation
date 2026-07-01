# =========================
# RA-SGG config for VisualGenome
# =========================
# This is an OpenSGG adapter config for audit/smoke workflows. It is not an
# official ReTAG reproduction config without the official checkpoint, PE-Net
# initialization, feature-bank memory artifacts, and evaluator parity evidence.

method = "RA_SGG"
loss = "motifs_loss"

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
hidden_dim = 512
visual_dim = 2048
dropout = 0.1
ra_sgg_embed_dim = 200
ra_sgg_memory_bank_path = None
ra_sgg_num_retrievals = 5
ra_sgg_retrieval_logit_coef = 0.0

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-12

# ===== dataset =====
dataset = "VisualGenome"
dataname = "VisualGenome"
entity_nums = 151
rel_nums = 51

# ===== backbone =====
backbone_arch = "resnet50"
backbone_pretrained = True
backbone_frozen = True
use_backbone = True
roi_output_size = 7

# ===== evaluation =====
eval_mode = "predcls"
metrics = ["R@50", "R@100", "mR@50", "mR@100"]

# ===== misc =====
device = "cuda"
num_workers = 4
seed = 42
