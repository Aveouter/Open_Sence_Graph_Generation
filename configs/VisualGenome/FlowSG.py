# =========================
# FlowSG config for VisualGenome
# =========================
# CVPR 2026 — §5.1 Implementation Details
# "5 Transformer blocks, 8-head attention, dim=512, dropout=0.1"
# "AdamW, 500K iterations, bs=128, lr=1e-4, wd=0.02, 4xA100"

method = 'flowsg'
loss = 'flowsg_loss'

# ===== optimizer (§5.1) =====
lr = 1e-4
lr_backbone = 1e-5         # backbone LR 10x lower
weight_decay = 0.02         # paper: 0.02
clip_max_norm = 1.0         # gradient clipping

# ===== scheduler =====
sched = 'cosine'             # cosine LR schedule
warmup_epoch = 5             # linear warmup (paper: 5K steps ≈ 5 epochs)
min_lr = 1e-6

# ===== training (§5.1) =====
epoch = 150                  # ~500K iters for VG (~75K images)
batch_size = 4               # limited by GPU; paper uses 128 with 4xA100
val_batch_size = 4

# ===== backbone =====
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
