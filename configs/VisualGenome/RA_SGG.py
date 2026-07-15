# =====================================
# RA-SGG — PredCls (Predicate Classification)
# =====================================
# Aligned with official: scripts/predcls_train_retag.sh
#   USE_GT_BOX=True, USE_GT_OBJECT_LABEL=True
#   NUM_RETRIEVALS=20, MAX_ITER=60000, BASE_LR=1e-3
#
# Official PE-Net checkpoint + memory bank are available → skip PE-Net pretrain.
# RA-SGG training:
#   python train.py --method RA_SGG --config_file configs/VisualGenome/RA_SGG.py \
#     --ckpt_path checkpoints/PE-NET_PredCls/model_final.pth \
#     --ra_sgg_memory_bank_path data/ra_sgg/predcls_fb_train.npy --gpus 0
#
# Evaluation (after training):
#   python train.py --test --method RA_SGG --config_file configs/VisualGenome/RA_SGG.py \
#     --ckpt_path outputs/runs/.../best-*.ckpt --gpus 0

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
eval_mode = "predcls"

# ===== Retrieval (PredCls: K=20) =====
ra_sgg_memory_size = 8
ra_sgg_num_retrievals = 20
ra_sgg_threshold = 0.3
ra_sgg_num_correct_bg = 1
ra_sgg_memory_bank_path = "checkpoints/PE-NET_PredCls/predcls_bg_processed_fb_train_8.npy"

# Official test.sh disables retrieval and frequency logits at inference.
# These values are persisted in hparams for protocol auditing; the local
# evaluation forward uses the model cosine logits only.
ra_sgg_model_logit_coef = 1.0
ra_sgg_retrieval_logit_coef = 0.0
ra_sgg_freq_logit_coef = 0.0

# Official PE-Net evaluator used by RA-SGG for PredCls/SGCls.
ra_sgg_relation_nms = True
ra_sgg_relation_nms_iou_threshold = 0.6
ra_sgg_relation_nms_l21_threshold = 0.7

# ===== Mixup =====
ra_sgg_mixup = True
ra_sgg_mixup_alpha = 20.0
ra_sgg_mixup_beta = 5.0

# ===== Loss =====
rel_loss_type = "ce"
reweight_beta = 0.99999

# ===== Optimizer (official: SGD, MOMENTUM=0.9, BASE_LR=1e-3) =====
opt = "momentum"
momentum = 0.9
lr = 1e-3
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 5.0

# ===== Scheduler (step-based, matches official STEPS=(28000,48000)) =====
sched = "warmup_multistep"
warmup_lr = 1e-5
warmup_epoch = 0
decay_epoch = [28000, 48000]
decay_rate = 0.1

# ===== Training =====
epoch = 50
batch_size = 6
val_batch_size = 6
freq_bias_eps = 1e-12

# ===== Dataset =====
dataset = "VisualGenome"
dataname = "VisualGenome"
entity_nums = 151
rel_nums = 51

# ===== Test image size (official: MIN_SIZE_TEST=600, MAX_SIZE_TEST=1000) =====
test_min_size = 600
test_max_size = 1000
# Use the official Stanford H5 targets at evaluation time. The legacy COCO
# conversion rounds boxes and removes exact duplicate relation annotations.
vg_use_official_h5_eval_annotations = True

# ===== Backbone =====
backbone_arch = "resnext101_32x8d"
# The official RA-SGG checkpoint supplies the complete visual extractor.
# Avoid an unnecessary torchvision ImageNet download before checkpoint load.
backbone_pretrained = False
backbone_frozen = True
use_backbone = True
roi_output_size = 7

# ===== Evaluation =====
metrics = [
    "predcls_R@20", "predcls_R@50", "predcls_R@100",
    "predcls_mR@20", "predcls_mR@50", "predcls_mR@100",
]

# ===== Misc =====
device = "cuda"
num_workers = 4
seed = 42
