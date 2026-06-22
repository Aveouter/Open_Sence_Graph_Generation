# =========================
# USG-Par config for VisualGenome
# =========================
# CVPR 2025 — Wu, Fei, Chua
# Single-modality (image-only) VG experiment settings.
#
# Architecture:
#   ResNet-50 backbone + 6-layer transformer mask decoder
#   3-layer RPC with bidirectional cross-attention
#   6-layer transformer relation decoder
#
# Training:
#   AdamW, lr=1e-4, wd=1e-4, bs=4, 150 epochs
#   Hungarian matching for object detection + CE for predicates

method = 'usg'
loss = 'usg_loss'

# ===== optimizer =====
lr = 1e-4
lr_backbone = 1e-5          # backbone LR 10x lower
weight_decay = 1e-4
clip_max_norm = 1.0          # gradient clipping

# ===== scheduler =====
sched = 'cosine'             # cosine LR schedule
warmup_epoch = 5             # linear warmup
min_lr = 1e-6

# ===== training =====
epoch = 150
batch_size = 4               # limited by GPU memory
val_batch_size = 4

# ===== backbone =====
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
hidden_dim = 256              # d_model (standard DETR dimension)
return_interm_layers = True   # needed for multi-scale features

# ===== transformer (mask decoder) =====
enc_layers = 6                # not used directly, for matcher compat
dec_layers = 6                # mask decoder layers
nheads = 8                    # attention heads
dropout = 0.1
dim_feedforward = 2048        # FFN hidden dim (256*8)
pre_norm = True               # pre-layer normalization
aux_loss = True               # auxiliary decoding losses

# ===== RPC (Relation Proposal Constructor) =====
rpc_layers = 3                # bidirectional cross-attn layers
top_k_pairs = 64              # top-K pairs for relation classification

# ===== Relation Decoder =====
rel_dec_layers = 6            # relation decoder layers

# ===== queries =====
num_queries = 200             # object queries

# ===== loss coefficients =====
ce_loss_coef = 1.0            # classification loss weight
bbox_loss_coef = 5.0          # L1 box loss weight
giou_loss_coef = 2.0          # GIoU loss weight
rel_loss_coef = 1.0           # relation loss weight
eos_coef = 0.1                # background class weight

# ===== matcher =====
set_cost_class = 1.0          # Hungarian matcher class cost
set_cost_bbox = 5.0           # Hungarian matcher bbox cost
set_cost_giou = 2.0           # Hungarian matcher GIoU cost
set_iou_threshold = 0.7       # Hungarian matcher IoU threshold

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151             # 150 object classes + 1 bg
rel_nums = 51                 # 50 predicate classes + 1 bg

# ===== evaluation =====
metrics = ["sgdet_R@50", "sgdet_R@100", "sgdet_mR@50", "sgdet_mR@100"]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
