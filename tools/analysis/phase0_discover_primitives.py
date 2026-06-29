#!/usr/bin/env python3
"""Phase 0 primitive discovery on extracted relation features.

Runs a lightweight sparse autoencoder / dictionary diagnostic and reports
sharedness, specificity, family validity, and random family baselines.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ON_FAMILY = [
    "on",
    "sitting on",
    "standing on",
    "lying on",
    "laying on",
    "walking on",
    "parked on",
    "mounted on",
]
ON_FINE = [p for p in ON_FAMILY if p != "on"]


class SparseAE(nn.Module):
    def __init__(self, dim, k, num_classes):
        super().__init__()
        self.encoder = nn.Linear(dim, k)
        self.dictionary = nn.Parameter(torch.randn(k, dim) * (1.0 / math.sqrt(dim)))
        self.classifier = nn.Linear(k, num_classes)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        recon = z @ self.dictionary
        logits = self.classifier(z)
        return z, recon, logits


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def standardize(x):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std, mean, std


def make_balanced_indices(labels, rng, per_class):
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[int(lab)].append(i)
    idxs = []
    for lab, arr in by.items():
        if len(arr) >= per_class:
            idxs.extend(rng.sample(arr, per_class))
        else:
            idxs.extend(rng.choices(arr, k=per_class))
    rng.shuffle(idxs)
    return idxs


def compute_metrics(z, labels, label_names, recon, x, k):
    n_classes = len(label_names)
    mean_z = torch.zeros(k, n_classes)
    counts = torch.zeros(n_classes)
    for cls in range(n_classes):
        mask = labels == cls
        counts[cls] = mask.sum()
        if mask.any():
            mean_z[:, cls] = z[mask].mean(0)

    # Reconstruction R2
    ss_res = ((x - recon) ** 2).sum().item()
    ss_tot = ((x - x.mean(0, keepdim=True)) ** 2).sum().item() + 1e-8
    r2 = 1.0 - ss_res / ss_tot

    shared_scores = []
    specificity = torch.zeros(k, n_classes)
    for kk in range(k):
        vals = mean_z[kk]
        tau = 0.5 * vals.max().item()
        shared = float((vals > tau).float().mean().item()) if vals.max() > 0 else 0.0
        shared_scores.append(shared)
        denom = vals.sum().item() + 1e-8
        specificity[kk] = vals / denom

    # Shared_F: fraction of primitives activated by >= half family
    shared_f = sum(1 for s in shared_scores if s > 0.5) / max(k, 1)
    # Private_F: fraction of fine predicates with a primitive Spec>0.7 (excluding parent cls 0)
    private_hits = 0
    for cls in range(1, n_classes):
        if (specificity[:, cls] > 0.7).any():
            private_hits += 1
    private_f = private_hits / max(n_classes - 1, 1)
    v_f = 0.4 * max(r2, 0.0) + 0.3 * shared_f + 0.3 * private_f

    return {
        "r2": r2,
        "shared_f": shared_f,
        "private_f": private_f,
        "family_validity": v_f,
        "mean_z": mean_z,
        "specificity": specificity,
        "shared_scores": shared_scores,
        "counts": counts,
    }


def train_one(x, labels, label_names, k, lambda_disc, lambda_sparse, seed, device):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    x = x.to(device)
    labels = labels.to(device)
    model = SparseAE(x.size(1), k, len(label_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    per_class = min(64, min(int((labels == c).sum().item()) for c in range(len(label_names))))
    per_class = max(per_class, 8)
    all_label_list = labels.detach().cpu().tolist()
    for epoch in range(80):
        idxs = make_balanced_indices(all_label_list, rng, per_class)
        batch = torch.tensor(idxs, dtype=torch.long, device=device)
        xb = x[batch]
        yb = labels[batch]
        z, recon, logits = model(xb)
        loss_rec = F.mse_loss(recon, xb)
        loss_disc = F.cross_entropy(logits, yb)
        loss_sparse = z.abs().mean()
        loss = loss_rec + lambda_disc * loss_disc + lambda_sparse * loss_sparse
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z, recon, logits = model(x)
    metrics = compute_metrics(z.cpu(), labels.cpu(), label_names, recon.cpu(), x.cpu(), k)
    return model, metrics


def make_random_baselines(labels, label_names, features, args, device):
    # Approximate random family baseline: shuffle labels while preserving class counts.
    # This destroys real predicate-feature association but keeps frequency distribution.
    baselines = []
    labels_cpu = labels.cpu().clone()
    for r in range(args.random_families):
        shuffled = labels_cpu[torch.randperm(labels_cpu.numel())]
        best_v = -1
        for k in args.k_values:
            _, m = train_one(features, shuffled, label_names, k, args.lambda_disc_values[0],
                             args.lambda_sparse, args.seed + 1000 + r * 17 + k, device)
            best_v = max(best_v, m["family_validity"])
        baselines.append(best_v)
    return baselines


def main():
    parser = argparse.ArgumentParser(description="Phase 0 primitive discovery")
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--lambda_disc_values", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--lambda_sparse", type=float, default=0.05)
    parser.add_argument("--random_families", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.features, map_location="cpu")
    x = data["features"].float()
    meta = load_jsonl(args.metadata)
    if x.numel() == 0:
        raise RuntimeError("No features found")

    # Keep only ON-family plus parent if present in metadata. If parent 'on' absent,
    # discovery still runs over the seven fine predicates and reports no parent coverage.
    labels_names_sorted = sorted(set(row["gt_predicate_name"] for row in meta))
    # Put 'on' first if exists, otherwise stable order
    if "on" in labels_names_sorted:
        labels_names_sorted = ["on"] + [p for p in labels_names_sorted if p != "on"]
    name_to_idx = {n: i for i, n in enumerate(labels_names_sorted)}
    labels = torch.tensor([name_to_idx[row["gt_predicate_name"]] for row in meta], dtype=torch.long)
    x, mean, std = standardize(x)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    best = None
    for k in args.k_values:
        for ld in args.lambda_disc_values:
            model, metrics = train_one(x, labels, labels_names_sorted, k, ld,
                                       args.lambda_sparse, args.seed + k * 31 + int(ld * 1000), device)
            row = {
                "k": k,
                "lambda_disc": ld,
                "lambda_sparse": args.lambda_sparse,
                "r2": metrics["r2"],
                "shared_f": metrics["shared_f"],
                "private_f": metrics["private_f"],
                "family_validity": metrics["family_validity"],
            }
            results.append(row)
            if best is None or row["family_validity"] > best[0]["family_validity"]:
                best = (row, model, metrics)
            print(row, flush=True)

    best_row, best_model, best_metrics = best
    random_vals = make_random_baselines(labels, labels_names_sorted, x, args, device)
    rand_mean = float(sum(random_vals) / len(random_vals)) if random_vals else 0.0
    v_margin = best_row["family_validity"] - rand_mean

    torch.save({
        "model_state": best_model.cpu().state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "label_names": labels_names_sorted,
        "best_config": best_row,
    }, out_dir / "best_model.pt")

    with open(out_dir / "k_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)

    mean_z = best_metrics["mean_z"]
    spec = best_metrics["specificity"]
    with open(out_dir / "sharedness_specificity.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["primitive", "shared_score"] + [f"mean_{n}" for n in labels_names_sorted] + [f"spec_{n}" for n in labels_names_sorted]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for kk, sh in enumerate(best_metrics["shared_scores"]):
            row = {"primitive": kk, "shared_score": sh}
            for ci, n in enumerate(labels_names_sorted):
                row[f"mean_{n}"] = float(mean_z[kk, ci])
                row[f"spec_{n}"] = float(spec[kk, ci])
            writer.writerow(row)

    # Heatmap as same matrix in CSV compact form
    with open(out_dir / "primitive_activation_heatmap.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["primitive"] + labels_names_sorted)
        for kk in range(mean_z.shape[0]):
            writer.writerow([kk] + [float(mean_z[kk, ci]) for ci in range(len(labels_names_sorted))])

    family_validity = {
        "best_config": best_row,
        "random_family_values": random_vals,
        "random_family_mean": rand_mean,
        "v_margin": v_margin,
        "passes_reltr_gate": best_row["family_validity"] > 0.5 and v_margin > 0.1,
        "labels": labels_names_sorted,
        "num_features": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
    }
    with open(out_dir / "family_validity.json", "w", encoding="utf-8") as f:
        json.dump(family_validity, f, indent=2)
    with open(out_dir / "random_family_baseline.json", "w", encoding="utf-8") as f:
        json.dump({"values": random_vals, "mean": rand_mean}, f, indent=2)
    print(json.dumps(family_validity, indent=2))


if __name__ == "__main__":
    main()
