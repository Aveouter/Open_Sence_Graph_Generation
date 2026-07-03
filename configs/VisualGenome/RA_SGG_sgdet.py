# =====================================
# RA-SGG — SGDet (Scene Graph Detection)
# =====================================
# Aligned with official: scripts/sgdet_train_retag.sh
#   USE_GT_BOX=False, USE_GT_OBJECT_LABEL=False
#   NUM_RETRIEVALS=10, MAX_ITER=20000, BASE_LR=1e-4, IMS_PER_BATCH=12
#
# Note: SGDet mode requires object proposals from the detector.
# The model uses self.mode='sgdet' so refine_obj_labels predicts
# object classes (no GT labels available).
#
# Two-phase workflow:
#   1. Pretrain PE-Net in sgdet mode:
#      python train.py --method RA_SGG --config_file configs/VisualGenome/RA_SGG_sgdet.py --gpus 0
#   2. Build memory bank.
#   3. Train RA-SGG:
#      python train.py --method RA_SGG --config_file configs/VisualGenome/RA_SGG_sgdet.py \
#        --ra_sgg_memory_bank_path data/VisualGenome/ra_sgg/sgdet_fb_train.npy --gpus 0

method = "RA_SGG"
loss = "rasgg_loss"

# ===== Model Architecture =====
ra_sgg_mlp_dim = 2048
ra_sgg_embed_dim = 300
hidden_dim = 512
visual_dim = 2048
dropout = 0.2
ra_sgg_use_bias = False
ra_sgg_use_union = True
ra_sgg_glove_dir = "data/glove"

# ===== Task Mode =====
eval_mode = "sgdet"

# ===== Retrieval (SGDet: K=10) =====
ra_sgg_memory_size = 8
ra_sgg_num_retrievals = 10
ra_sgg_threshold = 0.3
ra_sgg_num_correct_bg = 1
ra_sgg_memory_bank_path = ""

# ===== Mixup =====
ra_sgg_mixup = True
ra_sgg_mixup_alpha = 20.0
ra_sgg_mixup_beta = 5.0

# ===== Loss =====
rel_loss_type = "ce"
reweight_beta = 0.99999

# ===== Optimizer (official: SGD, MOMENTUM=0.9, BASE_LR=1e-4) =====
opt = "momentum"
momentum = 0.9
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 5.0

# ===== Scheduler (official: WarmupMultiStepLR, STEPS=(28000,48000), MAX_ITER=20000) =====
# Both STEPS > MAX_ITER → no LR decay during training (constant LR)
sched = "warmup_multistep"
warmup_lr = 1e-5
warmup_epoch = 0
decay_epoch = [28000, 48000]
decay_rate = 0.1

# ===== Training (SGDet: IMS_PER_BATCH=12, MAX_ITER=20000) =====
epoch = 16                   # 20000 iter / ~1250 steps-per-epoch ≈ 16
batch_size = 12
val_batch_size = 12
freq_bias_eps = 1e-12

# ===== Dataset =====
dataset = "VisualGenome"
dataname = "VisualGenome"
entity_nums = 151
rel_nums = 51

# ===== Backbone =====
backbone_arch = "resnet101"
backbone_pretrained = True
backbone_frozen = True
use_backbone = True
roi_output_size = 7

# ===== Evaluation =====
metrics = ["sgdet_R@50", "sgdet_R@100", "sgdet_mR@50", "sgdet_mR@100"]

# ===== Misc =====
device = "cuda"
num_workers = 4
seed = 42
