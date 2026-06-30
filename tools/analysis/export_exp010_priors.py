#!/usr/bin/env python3
"""EXP-010A: Export full 50-way prior scores for every val pair on the unified
hidden-positive support.

Priors (built from TRAIN exposed labels only — no val labels, no train hidden labels):
  - PairFrequencyPrior (PF):   score(y) = count(y | subj_label, obj_label); E included.
  - ExposedConditionalPrior (EC): same counts, but E forced to rank last (E "known").

Unified-support rule (G-SUPPORT):
  Every val manifest pair receives a full 50-way score vector and a full 50-way
  ranking (rank in 1..50). Pairs whose (subj, obj) never appeared in training get
  all-zero counts -> ties broken by a deterministic per-pair random jitter, so their
  expected hidden R@K == K/50 (a fair "no-signal" baseline). Nothing is dropped from
  the denominator. This matches the learned evaluator, which always ranks all 50
  predicates and keeps all 994 hidden positives.

subj/obj labels are read from train.json / val.json COCO in ann_id order, which is
identical to the dataset's `target["labels"]` ordering (iscrowd == 0, verified).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

NUM_PREDICATES = 50


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def img_to_anns_map(coco):
    m = defaultdict(list)
    for ann_id, ann in coco.anns.items():
        m[ann["image_id"]].append(ann)
    return m


def build_train_counts(train_manifest, train_coco):
    """pair_freq[(subj,obj)] = Counter of exposed predicates (train exposed only)."""
    tr_anns = img_to_anns_map(train_coco)
    pair_freq = defaultdict(Counter)
    for p in train_manifest["pairs"]:
        img_id = p["image_id"]
        si = p["sub_idx"]
        oi = p["obj_idx"]
        exp = p["exposed_label"]
        anns = tr_anns.get(img_id, [])
        if si >= len(anns) or oi >= len(anns):
            continue
        sl = anns[si]["category_id"]
        ol = anns[oi]["category_id"]
        pair_freq[(sl, ol)][exp] += 1
    return pair_freq


def jittered_ranking(counts, exposed, rng):
    """Full 50-way ranking: sort by count desc, ties broken by per-pair random jitter.

    `exposed` is forced to the last position for the EC variant when place_exposed_last
    is handled by the caller. Here we return the PF ranking (exposed at natural count
    rank). Returns a list of 50 predicate ids.
    """
    jitter = rng.rand(NUM_PREDICATES) * 1e-9
    keys = list(range(NUM_PREDICATES))
    keys.sort(key=lambda y: (-(counts[y] + jitter[y]), y))
    return keys


def export_prior_for_seed(seed, train_manifest_path, val_manifest_path, out_dir):
    from pycocotools.coco import COCO

    train_manifest = load_manifest(train_manifest_path)
    val_manifest = load_manifest(val_manifest_path)

    train_coco = COCO(str(_PROJECT_ROOT / "data" / "VisualGenome" / "train.json"))
    val_coco = COCO(str(_PROJECT_ROOT / "data" / "VisualGenome" / "val.json"))
    pair_freq = build_train_counts(train_manifest, train_coco)
    val_anns = img_to_anns_map(val_coco)

    out_dir.mkdir(parents=True, exist_ok=True)
    pf_path = out_dir / "scores_pair_frequency.jsonl"
    ec_path = out_dir / "scores_exposed_conditional.jsonl"

    n_written = 0
    n_hidden = 0
    n_empty_pair = 0  # pairs whose (subj,obj) unseen in train
    with open(pf_path, "w") as pf_fh, open(ec_path, "w") as ec_fh:
        for p in val_manifest["pairs"]:
            img_id = p["image_id"]
            si = p["sub_idx"]
            oi = p["obj_idx"]
            E = p["exposed_label"]
            H = p["hidden_labels"]
            anns = val_anns.get(img_id, [])
            if si >= len(anns) or oi >= len(anns):
                # Should not happen (iscrowd=0, verified), but keep the pair with
                # unknown labels rather than dropping it.
                sl = -1
                ol = -1
            else:
                sl = anns[si]["category_id"]
                ol = anns[oi]["category_id"]

            counter = pair_freq.get((sl, ol), Counter())
            total = sum(counter.values())
            counts = [int(counter.get(y, 0)) for y in range(NUM_PREDICATES)]
            prob = [c / total if total > 0 else 0.0 for c in counts]

            # Per-pair deterministic RNG for tie-breaking (seeded by pair key).
            pair_seed = hash((img_id, si, oi)) & 0xFFFFFFFF
            rng = np.random.RandomState(pair_seed)
            pf_ranked = jittered_ranking(counts, E, rng)
            if total == 0:
                n_empty_pair += 1

            # EC ranking: same order but exposed forced to last (E "known / removed").
            ec_ranked = [y for y in pf_ranked if y != E] + [E]

            common = {
                "image_id": int(img_id),
                "sub_idx": int(si),
                "obj_idx": int(oi),
                "exposed_label": int(E),
                "hidden_labels": [int(h) for h in H],
                "is_mapped_fine": bool(p.get("is_mapped_fine", False)),
                "subj_label": int(sl),
                "obj_label": int(ol),
                "scores": counts,  # raw co-occurrence counts (50-way)
                "prob": prob,  # L1-normalized probability (50-way)
            }
            pf_fh.write(json.dumps({**common, "ranked": pf_ranked}) + "\n")
            ec_fh.write(json.dumps({**common, "ranked": ec_ranked}) + "\n")
            n_written += 1
            n_hidden += len(H)

    print(
        f"seed={seed}: wrote {n_written} pairs ({n_hidden} hidden positives) "
        f"to {pf_path.name} + {ec_path.name}; empty-(subj,obj) pairs={n_empty_pair}"
    )


def main():
    parser = argparse.ArgumentParser(description="EXP-010A prior full-score exporter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true")
    args = parser.parse_args()

    seeds = [42, 123, 2027] if args.all_seeds else [args.seed]
    manifest_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "manifests"
    base_out = _PROJECT_ROOT / "outputs" / "gen_sgg" / "exp010_prior_strata"
    for seed in seeds:
        train_path = (
            manifest_dir / f"exp009_train_hidden_head_or_coarse_one_seed{seed}.json"
        )
        val_path = (
            manifest_dir / f"exp009_val_hidden_head_or_coarse_one_seed{seed}.json"
        )
        if not train_path.exists() or not val_path.exists():
            print(f"SKIP seed {seed}: manifest missing")
            continue
        export_prior_for_seed(seed, train_path, val_path, base_out / f"seed{seed}")


if __name__ == "__main__":
    main()
