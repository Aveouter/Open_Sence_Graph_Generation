"""
Build hierarchical CLIP text prototypes for VisualGenome predicate classes.

Usage:
    python scripts/build_clip_prototypes.py \
        --predicate_json data/VisualGenome/predicate_descriptions.json \
        --out_path data/VisualGenome/clip_prototypes.pth \
        --num_levels 3

Produces a .pth file with:
    predicate_names: list[str]                # 50 predicate names
    predicate_embeddings: Tensor[50, E]       # per-predicate text embeddings (L2-normed)
    hierarchy: list[dict]                     # one dict per level
        each dict:
            assignments: LongTensor[50]       # cluster index per predicate
            prototypes: Tensor[C_l, E]        # cluster centroids (L2-normed)
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import open_clip


def main():
    parser = argparse.ArgumentParser(description="Build CLIP text prototypes with hierarchical clustering")
    parser.add_argument("--predicate_json", type=str, required=True,
                        help="JSON file with {predicate_name: [descriptions]}")
    parser.add_argument("--out_path", type=str, default="data/VisualGenome/clip_prototypes.pth")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--num_levels", type=int, default=3,
                        help="Number of hierarchy cut levels (default 3)")
    parser.add_argument("--cluster_dist_thresholds", type=float, nargs="+", default=None,
                        help="Custom distance thresholds for each cut level. "
                             "If not set, clusters will be auto-split by quantiles.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # --- Load predicate descriptions ---
    with open(args.predicate_json, "r", encoding="utf-8") as f:
        pred_descs = json.load(f)

    predicate_names = list(pred_descs.keys())
    R = len(predicate_names)
    assert R > 0, "Predicate JSON must be non-empty"

    # Each predicate has M descriptions
    paragraph_lists = [pred_descs[name] for name in predicate_names]
    M = len(paragraph_lists[0])
    assert all(len(ps) == M for ps in paragraph_lists), \
        "All predicates must have the same number of descriptions"
    print(f"[Data] {R} predicates, {M} descriptions each")

    # --- Load OpenCLIP text encoder ---
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # --- Encode all descriptions ---
    flat_texts = [desc for paras in paragraph_lists for desc in paras]
    E = None
    feats = []
    with torch.no_grad():
        for i in range(0, len(flat_texts), args.batch_size):
            batch = flat_texts[i:i + args.batch_size]
            tok = tokenizer(batch).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.detach().cpu())
            if E is None:
                E = f.shape[-1]
    feats = torch.cat(feats, dim=0).view(R, M, -1)  # [R, M, E]

    # Mean pooling across paragraphs, then L2 normalize
    predicate_embeddings = feats.mean(dim=1)         # [R, E]
    predicate_embeddings = predicate_embeddings / predicate_embeddings.norm(dim=-1, keepdim=True)
    print(f"[Embedding] shape={tuple(predicate_embeddings.shape)}, dim={E}")

    # --- Hierarchical clustering ---
    emb_np = predicate_embeddings.numpy().astype(np.float64)
    # cosine distance matrix for clustering
    dist = pdist(emb_np, metric="cosine")
    Z = linkage(dist, method="average")  # agglomerative, average linkage

    # Determine cut thresholds
    if args.cluster_dist_thresholds is not None:
        thresholds = sorted(args.cluster_dist_thresholds, reverse=True)
        assert len(thresholds) == args.num_levels, \
            f"Got {len(thresholds)} thresholds but num_levels={args.num_levels}"
    else:
        # Auto-compute thresholds from linkage heights
        heights = Z[:, 2]
        max_h = heights.max()
        # Generate N evenly spaced thresholds from 0 to max_h (excluding 0)
        thresholds = np.linspace(max_h * 0.15, max_h * 0.85, args.num_levels)
        thresholds = sorted(thresholds.tolist(), reverse=True)  # descending: coarse -> fine

    print(f"[Clustering] thresholds (coarse->fine): {[f'{t:.4f}' for t in thresholds]}")

    hierarchy = []
    for level, t in enumerate(thresholds):
        cluster_ids = fcluster(Z, t=t, criterion="distance")  # 1-indexed
        unique_clusters = np.unique(cluster_ids)
        C = len(unique_clusters)
        # Map 1-indexed -> 0-indexed
        id_map = {old: new for new, old in enumerate(unique_clusters)}
        assignments = np.array([id_map[c] for c in cluster_ids], dtype=np.int64)

        # Compute cluster centroids
        prototypes = np.zeros((C, E), dtype=np.float32)
        for c in range(C):
            mask = (assignments == c)
            prototypes[c] = emb_np[mask].mean(axis=0)
        # L2 normalize
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=-1, keepdims=True) + 1e-8)

        hierarchy.append({
            "assignments": torch.from_numpy(assignments),
            "prototypes": torch.from_numpy(prototypes),
            "num_clusters": C,
            "threshold": float(t),
        })
        print(f"  Level {level}: {C} clusters (threshold={t:.4f})")

    # --- Save ---
    save_dict = {
        "predicate_names": predicate_names,
        "predicate_embeddings": predicate_embeddings,
        "clip_dim": E,
        "model": args.model,
        "pretrained": args.pretrained,
        "num_levels": args.num_levels,
        "num_predicates": R,
        "hierarchy": hierarchy,
    }
    torch.save(save_dict, args.out_path)
    print(f"[Saved] {args.out_path}")
    print(f"  Levels: {[h['num_clusters'] for h in hierarchy]}")
    print(f"  Predicate embedding norm check: "
          f"mean={predicate_embeddings.norm(dim=-1).mean():.4f}")


if __name__ == "__main__":
    main()
