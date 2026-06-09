# =========================
# EGTR config for VisualGenome
# =========================

method = 'EGTR'
loss = 'egtr_loss'

# ===== optimizer =====
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 0.1

# ===== training =====
epoch = 20
batch_size = 4
val_batch_size = 4

# ===== backbone =====
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'

# ===== transformer =====
hidden_dim = 256
dropout = 0.1
nheads = 8
dim_feedforward = 1024
enc_layers = 6
dec_layers = 6

# ===== queries =====
num_queries = 200

# ===== loss / matcher =====
aux_loss = True
bbox_loss_coef = 5
giou_loss_coef = 2
ce_loss_coef = 2.0
rel_loss_coef = 15.0
connectivity_loss_coef = 30.0
eos_coef = 0.1
focal_alpha = 0.25
rel_sample_negatives = 80
rel_sample_nonmatching = 80
rel_sample_negatives_largest = True
rel_sample_nonmatching_largest = True
smoothing = 1e-14

# ===== frequency bias =====
use_freq_bias = True
freq_bias_eps = 1e-12
use_log_softmax = False

# ===== logit adjustment =====
logit_adjustment = True
logit_adj_tau = 0.3

# ===== dataset =====
dataset = 'VisualGenome'
dataname = 'VisualGenome'
entity_nums = 150            # EGTR convention: 150 object classes, bg handled via matcher
rel_nums = 50                # EGTR convention: 50 predicate classes, no separate bg dim

# ===== image preprocessing =====
min_size = 800
max_size = 1333

# ===== model architecture =====
architecture = 'SenseTime/deformable-detr'

# ===== misc =====
device = 'cuda'
num_workers = 4
seed = 42
egtr_pretrained_path = 'outputs/pretrained/egtr/egtr_vg.tar.gz'  # set to pretrained DeformableDETR path for training
