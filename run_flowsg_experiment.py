#!/usr/bin/env python
"""FlowSG VG experiment: train for N epochs on subset, evaluate, report results.

Usage:
  python run_flowsg_experiment.py [--epochs 20] [--train_size 500] [--batch_size 4]
"""
import sys, os, json, time, gc, argparse
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from types import SimpleNamespace

# ---- Parse args ----
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--train_size', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
args_cli = parser.parse_args()

# ---- Config ----
config = dict(
    method='flowsg', dataname='VisualGenome', backbone='resnet50', dilation=False,
    position_embedding='sine', hidden_dim=512, return_interm_layers=True,
    dropout=0.1, nheads=8, dim_feedforward=2048, num_blocks=5,
    pre_norm=False, num_queries=100,
    num_slots=4, codebook_size=64, num_flow_steps=10,
    entity_nums=151, rel_nums=51, lr_backbone=1e-5, device='cuda',
    bbox_loss_coef=5, giou_loss_coef=2,
    flow_loss_coef=1.0, discrete_flow_coef=1.0, vq_loss_coef=1.0,
    edge_only_prob=0.2, eos_coef=0.1,
    batch_size=args_cli.batch_size, lr=args_cli.lr, weight_decay=0.02,
    dataset='vg', data_root='./data/VisualGenome/', distributed=False,
    num_workers=4, seed=42, eval=False,
)
a = SimpleNamespace(**config)

print(f"FlowSG Experiment: {args_cli.epochs} epochs, {args_cli.train_size} train images, batch_size={args_cli.batch_size}")
print(f"Model: 47M params, ResNet-50 backbone, 6 enc/6 dec layers")
print("=" * 70)

# ---- Build model ----
from src.models.flowsg import build_flowsg
model, criterion = build_flowsg(a)
model = model.cuda()
criterion = criterion.cuda()

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], "lr": a.lr},
    {"params": [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], "lr": a.lr_backbone},
]
optimizer = torch.optim.AdamW(param_dicts, weight_decay=a.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_cli.epochs)

# ---- Load data ----
print("Loading VG data...")
from data.dataloaders.dataloader_VisualGenome import load_data as load_vg
train_loader, val_loader, _ = load_vg(args=a)

train_n = min(args_cli.train_size, len(train_loader.dataset))
val_n = min(200, len(val_loader.dataset))
train_subset = Subset(train_loader.dataset, range(train_n))
val_subset = Subset(val_loader.dataset, range(val_n))

from utils.misc import collate_fn
train_dl = DataLoader(train_subset, batch_size=args_cli.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_dl = DataLoader(val_subset, batch_size=args_cli.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
print(f"Train: {train_n} images ({len(train_dl)} batches), Val: {val_n} images ({len(val_dl)} batches)")

# ---- Training ----
print("Training...")
all_train_losses = []

for epoch in range(args_cli.epochs):
    model.train()
    epoch_losses = []
    t0 = time.time()

    for batch in train_dl:
        images, targets = batch
        if hasattr(images, 'tensors'):
            images.tensors = images.tensors.cuda()
            images.mask = images.mask.cuda() if images.mask is not None else None
        targets = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        outputs = model(images, targets=targets)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        epoch_losses.append(total_loss.item())

    scheduler.step()
    avg_loss = np.mean(epoch_losses)
    all_train_losses.append(avg_loss)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1:3d}/{args_cli.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s", flush=True)

    if epoch % 5 == 0 or epoch == args_cli.epochs - 1:
        gc.collect()
        torch.cuda.empty_cache()

print(f"\nTraining complete: loss {all_train_losses[0]:.2f} → {all_train_losses[-1]:.2f}")

# ---- Evaluation ----
print("\nEvaluating...")
from src.core.metrics import metric

model.eval()
all_preds = {}
all_targets = []

with torch.no_grad():
    for batch in val_dl:
        images, targets = batch
        if hasattr(images, 'tensors'):
            images.tensors = images.tensors.cuda()
            images.mask = images.mask.cuda() if images.mask is not None else None
        else:
            from utils.misc import nested_tensor_from_tensor_list
            images = nested_tensor_from_tensor_list(images)
            images.tensors = images.tensors.cuda()
            images.mask = images.mask.cuda() if images.mask is not None else None

        out = model(images, targets=None)

        for k, v in out.items():
            if torch.is_tensor(v):
                if k not in all_preds:
                    all_preds[k] = []
                all_preds[k].append(v.cpu())

        for t in targets:
            all_targets.append({kk: vv.cpu() if torch.is_tensor(vv) else vv for kk, vv in t.items()})

# Merge batch predictions
merged_preds = {}
for k, batch_list in all_preds.items():
    if isinstance(batch_list[0], torch.Tensor):
        merged_preds[k] = torch.cat(batch_list, dim=0)
    elif isinstance(batch_list[0], np.ndarray):
        merged_preds[k] = np.concatenate(batch_list, axis=0)
    else:
        merged_preds[k] = batch_list

metrics_list = [
    "sgdet_R@10", "sgdet_R@20", "sgdet_R@50",
    "sgdet_mR@10", "sgdet_mR@20", "sgdet_mR@50",
    "predcls_R@10", "predcls_R@20",
    "predcls_mR@10", "predcls_mR@20",
]

eval_res, eval_log = metric(
    pred=merged_preds, true=all_targets, metrics=metrics_list,
    rel_nums=a.rel_nums - 1, entity_nums=a.entity_nums,
)

print(eval_log)
print("\n" + "=" * 70)
print("FLOWSG VG EXPERIMENT RESULTS")
print("=" * 70)
print(f"Config: {args_cli.epochs} epochs, {train_n} train / {val_n} val images, batch={args_cli.batch_size}")
print(f"Training loss: {all_train_losses[0]:.4f} → {all_train_losses[-1]:.4f}")
print()
for k, v in sorted(eval_res.items()):
    print(f"  {k}: {v:.4f}")

# Save results
from utils.path_utils import (
    resolve_output_paths, normalize_ex_name, ensure_unique_run_dir,
    save_eval_results, collect_metadata, write_metadata,
)

raw_name = f"{args_cli.epochs}e_{args_cli.train_size}n"
ex_name = normalize_ex_name(raw_name)
mock_args = SimpleNamespace(method='flowsg', ex_name=ex_name, output_dir='./outputs', seed=a.seed)
paths = resolve_output_paths(mock_args)
run_dir, actual_ex_name = ensure_unique_run_dir(paths['run_dir'])
paths = resolve_output_paths(mock_args, run_dir_override=run_dir)

metadata = collect_metadata(mock_args, run_dir, datetime.now().isoformat())
metadata['ex_name'] = actual_ex_name
write_metadata(run_dir, metadata)

save_eval_results(osp.join(run_dir, 'eval'), 'sgdet', eval_res)
print(f"\nResults saved to {run_dir}/eval/sgdet/")
