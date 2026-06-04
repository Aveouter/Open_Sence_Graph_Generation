# =========================
# HSTRNet config for VisualGenome
# =========================

method = 'hstrnet'
loss = 'hstrnet_loss'

# ===== optimizer =====
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 0.1

# ===== training =====
epoch = 150
batch_size = 16
val_batch_size = 16

# ===== model architecture =====
hidden_dim = 256
num_object_queries = 16
num_object_classes = 23         # Note: HSTRNet defaults; for VG set via --entity_nums
num_predicates = 29             # Note: HSTRNet defaults; for VG set via --rel_nums
max_relation_pairs = 32
in_channels = 3
temporal_num_layers = 2
temporal_num_heads = 8
temporal_dropout = 0.1
prototype_dim = 256
prototype_levels = [8, 16]

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151               # VG: 150 object classes + bg
rel_nums = 51                   # VG: 50 predicate classes + bg

# ===== evaluation =====
# HSTRNet supports PredCLS evaluation (GT boxes + labels, predicate-only prediction)
metrics = [
    "predcls_R@10", "predcls_R@20", "predcls_R@50",
    "predcls_mR@10", "predcls_mR@20", "predcls_mR@50",
]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
