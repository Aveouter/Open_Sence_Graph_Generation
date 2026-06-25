# =========================
# USG-Par config for VisualGenome
# =========================
# Aligned with official implementation: https://github.com/ChocoWu/USG
# configs/psg.yaml + usg_par/encoders/builders.py
#
# Architecture:
#   1. Frozen OpenCLIP ConvNeXt-L (or B fallback) + Transformer Pixel Decoder
#   2. Shared Mask Decoder (9 layers, Mask2Former-style masked cross-attn)
#   3. Detection Head (class + box regression for VG bbox)
#   4. RPC (4 layers, two-way cross-attn + self-attn + FFN)
#   5. Relation Decoder (6 layers, cross-attn + self-attn + FFN)

method = "usg"
loss = "usg_loss"

# ===== optimizer (official: configs/psg.yaml) =====
opt = "adamw"
lr = 1e-4
lr_backbone = 0  # CLIP encoder is frozen
weight_decay = 1e-4
clip_max_norm = 0.1  # grad_clip (official)

# ===== training (official: epochs=50, batch_size=8, warmup_steps=1000) =====
epoch = 50
batch_size = 4  # RTX 4090D 24GB limit (official: 8)
val_batch_size = 4
sched = "cosine"
warmup_steps = 1000  # official; auto-converted to warmup_epoch at runtime
warmup_lr = 1e-5

# ===== model architecture (official: configs/psg.yaml) =====
hidden_dim = 256  # dim
num_queries = 100  # N_q
mask_decoder_layers = 9  # L_mask (official: 9)
rpc_layers = 4  # L_RPC (official: 4)
relation_layers = 6  # L_rel (official: 6)
nheads = 8  # num_heads
ffn_dim = 2048  # mask decoder / RPC / relation decoder FFN dim
pixel_ffn_dim = 1024  # pixel decoder FFN dim (official: 1024)
top_k = 100  # RPC top-k pairs (official: 100)
dropout = 0.0  # official defaults to 0.0

# ===== loss coefficients (official: alpha=1.0, gamma=0.8) =====
class_loss_coef = 2.0  # L_obj (official α=1.0, VG: 2.0 for CE)
bbox_loss_coef = 5.0  # L1 box
giou_loss_coef = 2.0  # GIoU box
rel_loss_coef = 0.8  # L_rel (official γ=0.8)
pair_loss_coef = 0.5  # L_pair within L_rel
eos_coef = 0.1  # background class weight

# ===== dataset =====
dataset = "VisualGenome"
dataname = "VisualGenome"
entity_nums = 151
rel_nums = 51

# ===== OpenCLIP ConvNeXt encoder (official: builders.py) =====
# ConvNeXt-L @ 320px, LAION 2B pretrained
clip_model = "convnext_large_d_320"
clip_pretrained = "laion2b_s29b_b131k_ft_soup"
backbone_frozen = True

# ===== evaluation =====
eval_mode = "sgdet"

metrics = [
    "sgdet_R@10",
    "sgdet_R@20",
    "sgdet_R@50",
    "sgdet_mR@10",
    "sgdet_mR@20",
    "sgdet_mR@50",
]

# ===== misc =====
device = "cuda"
num_workers = 4
seed = 42
