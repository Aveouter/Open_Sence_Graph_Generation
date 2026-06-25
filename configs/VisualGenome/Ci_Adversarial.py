method = "Ci_Adversarial"
loss = "ce"

# optimizer
lr = 1e-3
lr_backbone = 1e-5
weight_decay = 1e-4
clip_max_norm = 5.0

# training
epoch = 50
batch_size = 8
val_batch_size = 8

# model architecture
hidden_dim = 512
visual_dim = 2048

# dataset
dataset = "VisualGenome"
dataname = "VisualGenome"
entity_nums = 151
rel_nums = 51

# evaluation
eval_mode = "sgdet"
metrics = [
    "sgdet_R@10",
    "sgdet_R@20",
    "sgdet_R@50",
    "sgdet_R@100",
    "sgdet_mR@10",
    "sgdet_mR@20",
    "sgdet_mR@50",
    "sgdet_mR@100",
]

device = "cuda"
num_workers = 4
seed = 42
