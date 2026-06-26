# Template configuration file for new SGG models
# Copy this file to configs/<DatasetName>/<ModelName>.py and modify as needed.
#
# Convention:
#   configs/<DatasetName>/<ModelName>.py

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataname = "VisualGenome"         # Dataset identifier
data_root = "./data"              # Root data directory
batch_size = 4                    # Training batch size
val_batch_size = 4                # Validation batch size
num_workers = 4                   # DataLoader workers
use_augment = False               # Enable image augmentation
drop_last = False                 # Drop last incomplete batch

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
method = "RelTR"                  # Model method name
in_shape = [3, 640, 640]          # Input shape [C, H, W]
hidden_dim = 256                  # Hidden dimension
dropout = 0.1                     # Dropout rate
drop_path = 0.0                   # Drop path rate

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
epoch = 200                       # Max training epochs
log_step = 1                      # Logging interval (epochs)
seed = 42                         # Random seed
test = False                      # Test-only mode
ckpt_path = None                  # Resume/finetune checkpoint path

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
opt = "adamw"                     # Optimizer type
lr = 1e-4                         # Learning rate
lr_backbone = 1e-5                # Backbone learning rate
weight_decay = 1e-4               # Weight decay
clip_grad = None                  # Gradient clipping (None = disabled)
clip_mode = "norm"                # Clip mode: "norm", "value"

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
sched = "cosine"                  # Scheduler type
warmup_epoch = 0                  # Warmup epochs
warmup_lr = 1e-5                  # Warmup learning rate
min_lr = 1e-6                     # Minimum learning rate
decay_epoch = 100                 # Decay interval (step scheduler)
decay_rate = 0.1                  # Decay rate (step scheduler)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
gpus = [0]                        # GPU device indices
metric_for_bestckpt = "val_loss"  # Metric to select best checkpoint
dist = False                      # Distributed training flag
fp16 = False                      # Mixed precision

# ---------------------------------------------------------------------------
# Task-specific
# ---------------------------------------------------------------------------
rel_nums = 51                     # Number of predicate classes (+1 for bg)
entity_nums = 151                 # Number of entity classes (+1 for bg)
metrics = ["R@50", "R@100", "mR@50", "mR@100"]  # Evaluation metrics
