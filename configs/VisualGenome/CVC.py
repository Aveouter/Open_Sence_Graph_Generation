# =========================
# CVC config for VisualGenome
# =========================
# CVC: Compositionally Verified Concept Relation Head
# LCompo-SGG — Long-tail Compositional Scene Graph Generation

method = 'CVC'
loss = 'cvc_loss'

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
hidden_dim = 512               # Hidden dim for backbone AND CVC head
visual_dim = 2048              # Raw ROI feature dim (for future backbone integration)
class_embed_dim = 256           # Class embedding dim for CVC branches
dropout = 0.1
obj_lstm_layers = 1
edge_lstm_layers = 1

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-12

# ===== CVC specific =====
cvc_bias_lambda = 0.5        # λ: inference bias subtraction strength
cvc_alpha = 0.1              # α: KL alignment loss weight
cvc_beta = 0.05              # β: prototype diversity loss weight
cvc_gamma = 0.1              # γ: adversarial composition removal weight
cvc_temperature = 2.0        # Temperature for softmax in KL div
cvc_num_compositions = 10068  # Number of seen compositions (from split analysis)

# ===== composition split (for inference correction) =====
composition_split = 'data/VisualGenome/composition_splits/split_0/composition_split.json'

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151
rel_nums = 51

# ===== backbone =====
backbone_arch = "resnet50"        # resnet50 or resnet101
backbone_pretrained = True          # Use ImageNet pretrained weights
backbone_frozen = True              # Freeze backbone during training
use_backbone = True                 # Set False for embedding placeholder
roi_output_size = 7                 # ROI Align output spatial size

# ===== evaluation =====
eval_mode = 'predcls'
metrics = ["R@50", "R@100", "mR@50", "mR@100"]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
