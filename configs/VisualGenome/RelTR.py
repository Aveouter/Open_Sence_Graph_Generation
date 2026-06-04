# =========================
# RelTR config for VisualGenome
# =========================

method = 'RelTR'
loss = 'reltr_loss'
# ===== optimizer =====
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4

# ===== training =====
epoch = 20
batch_size = 4
val_batch_size = 4
clip_max_norm = 0.1

# ===== backbone =====
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
return_interm_layers = False

# ===== transformer =====
hidden_dim = 256
dropout = 0.1
nheads = 8
dim_feedforward = 2048
enc_layers = 6
dec_layers = 6
pre_norm = False

# ===== queries =====
num_entities = 100
num_triplets = 200

# ===== loss / matcher =====
aux_loss = True
set_cost_class = 1
set_cost_bbox = 5
set_cost_giou = 2
set_iou_threshold = 0.7

bbox_loss_coef = 5
giou_loss_coef = 2
rel_loss_coef = 1
eos_coef = 0.1

# ===== dataset =====
dataset = 'VisualGenome'
num_classes = 151
num_rel_classes = 51
entity_nums = 151
rel_nums = 51

# ===== CLIP alignment =====
use_alignment = False
clip_model = 'ViT-B-32'
clip_dim = 512
num_hierarchy_levels = 3
hierarchy_weights = [0.2, 0.3, 0.5]
temperature = 0.07
align_loss_coef = 0.2
prototype_path = 'data/VisualGenome/clip_prototypes.pth'

# ===== misc =====
device = 'cuda'
masks = False
frozen_weights = None
num_workers = 4
seed = 42