# =========================
# FlowSG config for VisualGenome
# =========================
# CVPR 2026 — §5.1 Implementation Details
# "5 Transformer blocks, 8-head attention, dim=512, dropout=0.1"
# "AdamW, 500K iterations, bs=128, lr=1e-4, wd=0.02, 4xA100"
#
# ---- Training budget alignment ----
# Paper: 500K optimizer steps, eff_bs=128, 64M samples, 4×A100 (bs=32 each).
# effective_bs = batch_size × num_gpus × accumulate_grad_batches
# optimizer_steps_per_epoch = ceil(57723 / (batch_size × num_gpus)) / accum
# epochs = ceil(500_000 / optimizer_steps_per_epoch)
#
#   Scaling examples (VG150 = 57,723 images):
#     GPU(s)        bs    accum   eff_bs  epochs   est. time    notes
#     ───────────   ───   ─────   ──────  ──────   ─────────    ────────────────────
#     1 GPU         4     1       4       35       <1 day        quick sanity check
#     1×A100        16    16      64      554      ~30 days      500K steps, eff_bs=64
#     2×A100        32    2       128     1109     ~4 days       ★ DEFAULT — paper match
#     2×A100        16    4       128     1109     ~6 days       500K steps, paper match
#     4×A100        32    1       128     1109     ~2 days       paper config
#
# NOTE: A100 40GB supports bs=32 per GPU for FlowSG. For smaller GPUs
# (24GB), reduce batch_size and compensate with accumulate_grad_batches.

method = 'flowsg'
loss = 'flowsg_loss'

# ===== optimizer (§5.1) =====
opt = 'adamw'                # paper: AdamW (decoupled weight decay)
lr = 1e-4
lr_backbone = 1e-5         # backbone LR 10x lower
weight_decay = 0.02         # paper: 0.02
clip_max_norm = 1.0         # gradient clipping

# ===== training (§5.1) =====
# Default: 2×A100, eff_bs=128, 500K optimizer steps, ~4 days.
# Matches paper exactly: bs=128, 500K steps, 64M samples, 1109 epochs.
batch_size = 32              # per-GPU batch size (A100 40GB; reduce for 24GB GPUs)
val_batch_size = 32
accumulate_grad_batches = 2  # eff_bs = 32 × 2 GPUs × 2 = 128 (paper spec)
epoch = 1109                 # 500K steps at eff_bs=128 on 57,723 images (~4 days 2×A100)
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
