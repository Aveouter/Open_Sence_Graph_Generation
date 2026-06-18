# =========================
# TDE config for VisualGenome
# =========================
# Unbiased Scene Graph Generation from Biased Training (Tang et al., CVPR 2020)
#
# TDE uses Motifs backbone + causal intervention at inference time.
# Training is identical to Motifs; the causal subtraction is applied
# only during evaluation.

method = 'TDE'
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
visual_dim = 2048
motifs_obj_feat_dim = 4096
pooling_dim = 4096
embed_dim = 200
dropout = 0.2
obj_lstm_layers = 1
edge_lstm_layers = 1
motifs_order = "leftright"
use_vision = True
use_tanh = False
motifs_pos_embed_dim = 128
motifs_pos_batchnorm = True
motifs_obj_feat_to_edge = True
motifs_effect_analysis = True
motifs_include_bg_predicate = True
motifs_predicate_bg_index = "first"

# ===== TDE specific =====
tde_fusion = 'subtract'         # 'subtract' or 'softmax_subtract'

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-3

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 151
rel_nums = 51

# ===== backbone =====
backbone_arch = "resnet101"         # SGB checkpoint uses R-101-FPN; OpenSGG uses plain ResNet ROI features.
backbone_pretrained = True          # Use ImageNet pretrained weights
backbone_frozen = True              # Freeze backbone during training
use_backbone = True                 # Set False for embedding placeholder
roi_output_size = 7                 # ROI Align output spatial size

# ===== evaluation =====
eval_mode = 'predcls'

metrics = [
    "predcls_R@10", "predcls_R@20", "predcls_R@50",
    "predcls_mR@10", "predcls_mR@20", "predcls_mR@50",
]

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
