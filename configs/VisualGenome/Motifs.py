# =========================
# Motifs config for VisualGenome
# =========================
# Neural Motifs: Scene Graph Parsing with Global Context (Zellers et al., CVPR 2018)

method = 'Motifs'
loss = 'motifs_loss'

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
visual_dim = 2048          # ResNet-101 / VGG ROI feature dimension
dropout = 0.1
obj_lstm_layers = 1
edge_lstm_layers = 1

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-12

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151             # 150 obj classes + 1 background
rel_nums = 51                 # 50 pred classes + 1 background

# ===== backbone =====
backbone_arch = "resnet50"        # resnet50 or resnet101
backbone_pretrained = True          # Use ImageNet pretrained weights
backbone_frozen = True              # Freeze backbone during training
use_backbone = True                 # Set False for embedding placeholder
roi_output_size = 7                 # ROI Align output spatial size

# ===== evaluation =====
eval_mode = 'predcls'         # PredCLS: GT boxes + GT labels

metrics = ["R@50", "R@100", "mR@50", "mR@100"]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
