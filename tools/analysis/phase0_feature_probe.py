#!/usr/bin/env python3
"""Phase 0B feature probe diagnostics.

Tests whether extracted relation features contain fine-vs-parent and fine-sibling
information for ON-family predicates.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PARENT = "on"
FINE = [
    "sitting on",
    "standing on",
    "lying on",
    "laying on",
    "walking on",
    "parked on",
    "mounted on",
]


def load_meta(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def auc_score(y_true, scores):
    y = np.asarray(y_true).astype(np.int32)
    s = np.asarray(scores).astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUC
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(s)) + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Probe(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, min(256, max(32, dim // 4))),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(min(256, max(32, dim // 4)), out),
        )

    def forward(self, x):
        return self.net(x)


def split_indices(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def standardize_train(x_train, x_all):
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp(min=1e-6)
    return (x_all - mean) / std, mean, std


def train_probe(x, y, out_dim, seed=42, epochs=50, device="cpu"):
    train_idx, val_idx, test_idx = split_indices(x.size(0), seed)
    x_std, mean, std = standardize_train(x[train_idx], x)
    x_std = x_std.to(device)
    y = y.to(device)
    model = Probe(x.size(1), out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = -1
    for ep in range(epochs):
        model.train()
        logits = model(x_std[train_idx.to(device)])
        yy = y[train_idx.to(device)]
        loss = F.cross_entropy(logits, yy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_std[val_idx.to(device)])
            val_pred = val_logits.argmax(-1)
            val_acc = (val_pred == y[val_idx.to(device)]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(x_std[test_idx.to(device)]).detach().cpu()
        test_y = y[test_idx.to(device)].detach().cpu()
        pred = test_logits.argmax(-1)
        acc = (pred == test_y).float().mean().item()
    return model, acc, test_logits, test_y, mean, std


def binary_probe_for_pred(x, names, fine_name, device):
    idx = [i for i, n in enumerate(names) if n in (fine_name, PARENT)]
    if len(idx) < 20:
        return {"fine": fine_name, "n": len(idx), "auc": None, "acc": None}
    y = torch.tensor([1 if names[i] == fine_name else 0 for i in idx], dtype=torch.long)
    xx = x[idx]
    model, acc, logits, yy, mean, std = train_probe(xx, y, 2, seed=42, device=device)
    prob = logits.softmax(-1)[:, 1].numpy()
    auc = auc_score(yy.numpy(), prob)
    return {
        "fine": fine_name,
        "n": len(idx),
        "auc": auc,
        "acc": acc,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 0B feature probe")
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.features, map_location="cpu")
    x = data["features"].float()
    meta = load_meta(args.metadata)
    names = [m["gt_predicate_name"] for m in meta]

    binary_results = [binary_probe_for_pred(x, names, f, device) for f in FINE]
    probe_weights = {
        r["fine"]: {
            "state_dict": r.pop("state_dict", None),
            "mean": r.pop("mean", None),
            "std": r.pop("std", None),
        }
        for r in binary_results
    }
    binary_rows = binary_results

    # fine-sibling classification only on fine predicates
    fine_idx = [i for i, n in enumerate(names) if n in FINE]
    fine_label_names = sorted(set(names[i] for i in fine_idx))
    name_to_idx = {n: j for j, n in enumerate(fine_label_names)}
    if len(fine_idx) >= 50 and len(fine_label_names) > 1:
        y = torch.tensor([name_to_idx[names[i]] for i in fine_idx], dtype=torch.long)
        xx = x[fine_idx]
        fine_model, fine_acc, _, _, fine_mean, fine_std = train_probe(
            xx, y, len(fine_label_names), seed=123, device=device
        )
        random_acc = 1.0 / len(fine_label_names)
    else:
        fine_acc = None
        random_acc = None
        fine_model = None
        fine_mean = None
        fine_std = None

    with open(out / "fine_vs_parent_probe.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fine", "n", "auc", "acc"])
        writer.writeheader()
        writer.writerows(binary_rows)

    torch.save(
        {
            "binary_probes": probe_weights,
            "fine_sibling_probe": {
                "state_dict": {
                    k: v.detach().cpu() for k, v in fine_model.state_dict().items()
                }
                if fine_model is not None
                else None,
                "mean": fine_mean.detach().cpu() if fine_mean is not None else None,
                "std": fine_std.detach().cpu() if fine_std is not None else None,
                "label_names": fine_label_names,
            },
            "feature_dim": int(x.shape[1]),
        },
        out / "probe_weights.pt",
    )

    summary = {
        "model": args.model,
        "num_features": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "binary_probes": binary_rows,
        "mean_auc": float(
            np.nanmean([r["auc"] for r in binary_rows if r["auc"] is not None])
        )
        if any(r["auc"] is not None for r in binary_rows)
        else None,
        "fine_sibling_acc": fine_acc,
        "fine_sibling_random_acc": random_acc,
        "passes_probe_gate": False,
    }
    if summary["mean_auc"] is not None:
        summary["passes_probe_gate"] = summary["mean_auc"] > 0.75
    with open(out / "probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
