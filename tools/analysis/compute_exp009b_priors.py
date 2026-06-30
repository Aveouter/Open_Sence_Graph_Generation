#!/usr/bin/env python3
"""EXP-009B: Compute fair prior baselines for hidden-positive recovery.

Priors:
  A. ObjectPairFrequencyPrior: p(predicate | subj_label, obj_label)
  B. ExposedConditionalPrior:   p(predicate | exposed_label, subj_label, obj_label)

Rules:
  - Only train exposed labels may be used (no train hidden labels, no val labels)
  - Output hidden R@K, mAP, mean rank for comparison against learned models
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def build_prior_from_train(manifest_path):
    """Build frequency tables from train exposed labels only."""
    manifest = load_manifest(manifest_path)
    # Load VG annotations to get subject/object labels
    # For each pair in manifest, we have (image_id, sub_idx, obj_idx)
    # We need subject/object labels - these come from VG object annotations
    # Load VG train COCO annotations
    from pycocotools.coco import COCO

    ann_file = str(_PROJECT_ROOT / "data" / "VisualGenome" / "train.json")
    coco = COCO(ann_file)

    # subj→label and obj→label from COCO annotations
    img_to_anns = defaultdict(list)
    for ann_id, ann in coco.anns.items():
        img_to_anns[ann["image_id"]].append(ann)

    # Build: (subj_label, obj_label) → Counter of exposed predicates
    pair_freq = defaultdict(Counter)
    exposed_cond_freq = defaultdict(lambda: defaultdict(Counter))

    pairs = manifest["pairs"]
    for p in pairs:
        img_id = p["image_id"]
        sub_idx = p["sub_idx"]
        obj_idx = p["obj_idx"]
        exposed = p["exposed_label"]

        anns = img_to_anns.get(img_id, [])
        if sub_idx >= len(anns) or obj_idx >= len(anns):
            continue
        subj_label = anns[sub_idx]["category_id"]
        obj_label = anns[obj_idx]["category_id"]

        key = (subj_label, obj_label)
        pair_freq[key][exposed] += 1
        exposed_cond_freq[key][exposed][exposed] += (
            1  # counter of co-occurring exposed labels
        )

    # Also need: for EXPOSED conditional, we need p(hidden | exposed, subj, obj)
    # But we can ONLY use train exposed labels. So we build:
    # For each (subj, obj, exposed), count the other predicates that also appear
    # as exposed for the same (subj, obj) pairs.
    # This is subtle: we're estimating p(other_pred | same_pair, exposed_pred)
    # from the train exposed distribution.

    # For ObjectPairFrequencyPrior: sort by frequency, that's the prior ranking
    def get_pair_prior(subj_label, obj_label):
        counter = pair_freq.get((subj_label, obj_label), Counter())
        if not counter:
            return []
        return sorted(counter.items(), key=lambda x: -x[1])

    # For ExposedConditionalPrior: given the exposed label, what other predicates
    # commonly co-occur on the same (subj, obj) pairs?
    # We estimate this from: pairs that share (subj, obj) and have different exposed labels
    # Build: per (subj, obj), all exposed labels seen across different pairs
    # Then for a new pair with exposed=E, predict other labels seen on same (subj, obj)
    same_pair_labels = defaultdict(set)
    for p in pairs:
        img_id = p["image_id"]
        sub_idx = p["sub_idx"]
        obj_idx = p["obj_idx"]
        anns = img_to_anns.get(img_id, [])
        if sub_idx >= len(anns) or obj_idx >= len(anns):
            continue
        subj_label = anns[sub_idx]["category_id"]
        obj_label = anns[obj_idx]["category_id"]
        key = (subj_label, obj_label)
        same_pair_labels[key].add(p["exposed_label"])

    def get_conditional_prior(subj_label, obj_label, exposed_label):
        """Return predicates that co-occur with exposed on same (subj, obj) in training."""
        key = (subj_label, obj_label)
        all_exposed = same_pair_labels.get(key, set())
        # Rank by frequency from pair_freq
        counter = pair_freq.get(key, Counter())
        candidates = [
            (pred, counter.get(pred, 0))
            for pred in all_exposed
            if pred != exposed_label
        ]
        return sorted(candidates, key=lambda x: -x[1])

    # Build lookup from image_id→anns for val evaluation
    return {
        "pair_freq": pair_freq,
        "same_pair_labels": same_pair_labels,
        "get_pair_prior": get_pair_prior,
        "get_conditional_prior": get_conditional_prior,
    }


def evaluate_prior(train_manifest_path, val_manifest_path, prior_type, seed):
    """Evaluate prior on val hidden-positive manifest."""
    prior_data = build_prior_from_train(train_manifest_path)
    val_manifest = load_manifest(val_manifest_path)

    if prior_type == "pair_frequency":

        def get_ranking(subj, obj, exposed):
            return [pred for pred, _ in prior_data["get_pair_prior"](subj, obj)]

    elif prior_type == "exposed_conditional":

        def get_ranking(subj, obj, exposed):
            return [
                pred
                for pred, _ in prior_data["get_conditional_prior"](subj, obj, exposed)
            ]

    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    # Evaluate on val
    from pycocotools.coco import COCO

    ann_file = str(_PROJECT_ROOT / "data" / "VisualGenome" / "val.json")
    val_coco = COCO(ann_file)
    val_img_to_anns = defaultdict(list)
    for ann_id, ann in val_coco.anns.items():
        val_img_to_anns[ann["image_id"]].append(ann)

    hidden_count = 0
    hidden_hits = {1: 0, 3: 0, 5: 0}
    hidden_rank_sum = 0.0
    hidden_ap_sum = 0.0
    pair_count = 0

    for p in val_manifest["pairs"]:
        if not p["hidden_labels"]:
            continue
        pair_count += 1
        img_id = p["image_id"]
        sub_idx = p["sub_idx"]
        obj_idx = p["obj_idx"]
        exposed = p["exposed_label"]
        hidden = p["hidden_labels"]

        anns = val_img_to_anns.get(img_id, [])
        if sub_idx >= len(anns) or obj_idx >= len(anns):
            continue
        subj_label = anns[sub_idx]["category_id"]
        obj_label = anns[obj_idx]["category_id"]

        ranking = get_ranking(subj_label, obj_label, exposed)
        if not ranking:
            continue

        for h in hidden:
            hidden_count += 1
            try:
                rank = ranking.index(h) + 1
            except ValueError:
                rank = 9999
            hidden_rank_sum += rank
            for k in [1, 3, 5]:
                if rank <= k:
                    hidden_hits[k] += 1
            # Simple AP: precision at each correct position
            correct_at = [i + 1 for i, pred in enumerate(ranking[:50]) if pred == h]
            if correct_at:
                precisions = [(j + 1) / pos for j, pos in enumerate(correct_at)]
                hidden_ap_sum += sum(precisions) / len(correct_at)

    N_h = max(hidden_count, 1)
    summary = {
        "protocol": "exp009b_prior",
        "prior_type": prior_type,
        "seed": seed,
        "train_manifest_sha": file_sha256(train_manifest_path),
        "val_manifest_sha": file_sha256(val_manifest_path),
        "num_val_pairs": pair_count,
        "num_hidden_positives": hidden_count,
        "hidden_recall_full": {f"r@{k}": hidden_hits[k] / N_h for k in [1, 3, 5]},
        "hidden_mAP": hidden_ap_sum / N_h,
        "hidden_mean_rank": hidden_rank_sum / N_h,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP-009B prior baselines")
    parser.add_argument(
        "--prior",
        choices=["pair_frequency", "exposed_conditional", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true")
    args = parser.parse_args()

    seeds = [42, 123, 2027] if args.all_seeds else [args.seed]
    manifest_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "manifests"
    out_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "exp009b_controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_types = (
        ["pair_frequency", "exposed_conditional"]
        if args.prior == "all"
        else [args.prior]
    )

    for prior_type in prior_types:
        for seed in seeds:
            train_path = (
                manifest_dir / f"exp009_train_hidden_head_or_coarse_one_seed{seed}.json"
            )
            val_path = (
                manifest_dir / f"exp009_val_hidden_head_or_coarse_one_seed{seed}.json"
            )
            if not train_path.exists():
                print(f"SKIP seed {seed}: manifest missing")
                continue

            summary = evaluate_prior(str(train_path), str(val_path), prior_type, seed)
            out_path = out_dir / f"{prior_type}_seed{seed}_summary.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(
                f"{prior_type} seed={seed}: "
                f"HP R@1={summary['hidden_recall_full']['r@1']:.4f} "
                f"HP R@3={summary['hidden_recall_full']['r@3']:.4f} "
                f"HP R@5={summary['hidden_recall_full']['r@5']:.4f} "
                f"mAP={summary['hidden_mAP']:.4f} "
                f"mean_rank={summary['hidden_mean_rank']:.1f} "
                f"-> {out_path}"
            )

    print("Done. All prior baselines computed.")


if __name__ == "__main__":
    main()
