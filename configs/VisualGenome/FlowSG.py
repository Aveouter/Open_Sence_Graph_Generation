# =========================
# FlowSG config for VisualGenome
# =========================
# CVPR 2026 — §5.1 Implementation Details
# "5 Transformer blocks, 8-head attention, dim=512, dropout=0.1"
# "AdamW, 500K iterations, bs=128, lr=1e-4, wd=0.02, 4xA100"
#
# Iteration → epoch alignment:
#   Paper:  500K iters × bs=128 / 57,723 imgs ≈ 1,109 epochs
#   Ours:   epoch = ceil(500K / (N / effective_bs))
#           effective_bs = batch_size × accumulate_grad_batches
#   Examples:
#     bs=4  accum=1  → 500K / (57723/4)    ≈ 35 epochs  (matches iter count, not data volume)
#     bs=8  accum=4  → 500K / (57723/32)   ≈ 277 epochs (effective bs=32)
#     bs=8  accum=16 → 500K / (57723/128)  ≈ 1109 epochs (matches paper exactly)
#     bs=16 accum=8  → 500K / (57723/128)  ≈ 1109 epochs
#     bs=8  accum=1  → 500K / (57723/8)    ≈ 69 epochs

method = 'flowsg'
loss = 'flowsg_loss'

# ===== optimizer (§5.1) =====
opt = 'adamw'                # paper: AdamW (decoupled weight decay)
lr = 1e-4
lr_backbone = 1e-5         # backbone LR 10x lower
weight_decay = 0.02         # paper: 0.02
clip_max_norm = 1.0         # gradient clipping

# ===== training (§5.1) =====
# 500K iterations alignment on VG150 (57,723 images):
#   bs=4 → 57,723/4=14,431 steps/epoch → 500K/14,431 ≈ 35 epochs
#   bs=6 → 57,723/6= 9,620 steps/epoch → 500K/ 9,620 ≈ 52 epochs
#   bs=8 → 57,723/8= 7,215 steps/epoch → 500K/ 7,215 ≈ 69 epochs
# RTX 4090D (24GB): bs=4 uses ~20.9GB VRAM → bs=4 is the practical max
# Use accumulate_grad_batches to simulate larger effective batch if needed
epoch = 35                   # 500K iters with bs=4 on VG150
# NOTE: epoch=35 matches paper ITERATION count, not DATA VOLUME.
# Paper trains 500K iters × bs=128 = 64M samples.
# This config trains 500K iters × bs=4 = 2M samples (1/32 of paper data).
# Use accumulate_grad_batches=32 to match paper data volume (~1109 epochs).
batch_size = 4               # RTX 4090D 24GB limit (~0.24s/step, ~16.9 imgs/sec)
val_batch_size = 4
accumulate_grad_batches = 1  # set >1 to simulate larger batch (e.g. 8 for eff bs=32)
sched = 'cosine'             # cosine LR schedule (standard for diffusion/flow models)

# ===== image encoders (§4.2, §5.1) =====
# PRIMARY (hardcoded in model, not read from config):
#   - Frozen CLIP ViT-B/16 → global + spatial patch features (§4.2)
#   - Frozen Mask2Former (Swin-S, COCO) → object proposals (§5.1)
# FALLBACK (this config line, used when M2F weights unavailable):
#   - ResNet50 → grid-based detection heads
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
hidden_dim = 512             # paper: 512
return_interm_layers = True  # needed for multi-scale features

# ===== transformer (§4.3, §5.1) =====
dropout = 0.1                # paper: 0.1
nheads = 8                   # paper: 8
num_blocks = 5               # paper: 5 DiT blocks
dim_feedforward = 2048       # mlp_ratio=4, 512*4=2048

# ===== VQ-VAE (§4.1, §5.1) =====
num_slots = 4                # M=4 ordered slots
codebook_size = 64           # K=64 entries
code_dim = 512               # d=512
vq_commitment_cost = 0.25    # beta

# ===== Flow Matching (§4.2) =====
num_flow_steps = 10          # ODE solver steps at inference
edge_only_prob = 0.2         # stochastic edge-only training probability

# ===== queries / graph =====
num_queries = 100            # N object proposals (paper uses detector proposals)

# ===== loss coefficients (§4.2) =====
bbox_loss_coef = 5
giou_loss_coef = 2
flow_loss_coef = 1.0         # CFM loss weight
discrete_flow_coef = 1.0     # DFM loss weight (λ in L = L_CFM + λ·L_DFM)
vq_loss_coef = 1.0           # VQ-VAE loss weight
eos_coef = 0.1               # background class weight

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151            # 150 object classes + 1 bg
rel_nums = 51                # 50 predicate classes + 1 bg

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
