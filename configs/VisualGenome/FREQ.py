# =========================
# FREQ config for VisualGenome
# =========================
# Frequency baseline: P(predicate | subject class, object class)

method = 'FREQ'
loss = 'motifs_loss'

# ===== optimizer =====
lr = 1e-3
weight_decay = 0.0
clip_max_norm = 5.0
opt = 'adam'
sched = 'onecycle'
warmup_epoch = 0
decay_epoch = 100
decay_rate = 0.1

# ===== training =====
epoch = 1
batch_size = 8
val_batch_size = 8

# ===== model architecture =====
hidden_dim = 512
visual_dim = 2048
use_backbone = False
freq_bias_eps = 1e-3
freq_predicate_bg_index = "first"

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151
rel_nums = 51

# ===== evaluation =====
eval_mode = 'predcls'
metrics = [
    "predcls_R@10", "predcls_R@20", "predcls_R@50", "predcls_R@100",
    "predcls_mR@10", "predcls_mR@20", "predcls_mR@50", "predcls_mR@100",
]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
