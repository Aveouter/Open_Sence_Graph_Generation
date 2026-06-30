#!/usr/bin/env python3
"""Build EXP-009 hidden-positive manifests.

For each multi-label directed pair (image_id, sub_idx, obj_idx), select one exposed label
and hide the rest. The exposed label is chosen by the specified policy.

Policies:
  - head_or_coarse_one: expose coarsest ancestor if mapped, else highest train-frequency
  - random_one: random selection (control)
  - fine_or_tail_one: expose fine/tail label (stress test)

Output: frozen JSON manifest per seed × split.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parent_map(path):
    with open(path) as f:
        data = json.load(f)
    parent_map = {}
    for section in ["primary_on_family"]:
        for item in data[section]["mappings"]:
            child = int(item["child_id"]) - 1
            parent = int(item["parent_id"]) - 1
            parent_map[child] = parent
    return parent_map


def load_manifest_image_ids(path):
    with open(path) as f:
        manifest = json.load(f)
    return [int(iid) for iid in manifest["image_ids"]]


def build_pairs(relations, parent_map):
    """Group relations by (sub_idx, obj_idx). Returns list of {sub_idx, obj_idx, labels}."""
    pairs = defaultdict(set)
    for rel in relations:
        sub_idx, obj_idx, pred_id = map(int, rel[:3])
        pred_idx = pred_id - 1  # 1-indexed -> 0-indexed
        pairs[(sub_idx, obj_idx)].add(pred_idx)
    result = []
    for (sub_idx, obj_idx), labels in pairs.items():
        labels_list = sorted(labels)
        entry = {
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
            "all_labels": labels_list,
        }
        # Check if any label is a mapped fine child
        has_mapped = any(lbl in parent_map for lbl in labels_list)
        entry["has_mapped_fine"] = has_mapped
        result.append(entry)
    return result


def select_exposed_head_or_coarse(labels, parent_map, freq_map):
    """Expose the coarsest ancestor if mapped relationships exist,
    otherwise the highest-frequency label. Tie-break by lowest predicate id."""
    # Check for mapped parent/child relationships
    parents_in_labels = set()
    children_in_labels = set()
    for lbl in labels:
        if lbl in parent_map:
            children_in_labels.add(lbl)
            parent = parent_map[lbl]
            if parent in labels:
                parents_in_labels.add(parent)
    # If we have both parent and child in the labels, expose the coarsest parent
    if parents_in_labels:
        # Pick the parent with highest depth (coarsest)
        def depth(lbl):
            d = 0
            seen = set()
            cur = lbl
            while cur in parent_map and cur not in seen:
                seen.add(cur)
                cur = parent_map[cur]
                d += 1
            return d

        best = max(
            parents_in_labels,
            key=lambda label: (depth(label), -freq_map.get(label, 0), -label),
        )
        return best
    # Otherwise expose highest-frequency label
    return max(labels, key=lambda label: (freq_map.get(label, 0), -label))


def select_exposed_random(labels, rng):
    return rng.choice(list(labels))


def select_exposed_fine_or_tail(labels, parent_map, freq_map):
    """Expose the finest/most-tail label. Opposite of head_or_coarse."""
    # If mapped relationship exists, expose the child (finest)
    for lbl in labels:
        if lbl in parent_map and parent_map[lbl] in labels:
            return lbl
    # Otherwise expose lowest-frequency label (tail)
    return min(labels, key=lambda label: (freq_map.get(label, 999999), label))


def build_manifest(split, image_ids, rel_data, parent_map, policy, rng, freq_map):
    pairs_data = []
    multi_count = 0
    hidden_fine_count = 0
    total_pairs = 0

    for image_id in image_ids:
        key = str(image_id)
        if key not in rel_data:
            key = image_id
        if key not in rel_data:
            continue
        relations = rel_data[key]
        if not relations:
            continue

        pair_entries = build_pairs(relations, parent_map)
        for entry in pair_entries:
            total_pairs += 1
            labels = entry["all_labels"]
            if len(labels) >= 2:
                multi_count += 1
                # Select exposed label
                if policy == "head_or_coarse_one":
                    exposed = select_exposed_head_or_coarse(
                        labels, parent_map, freq_map
                    )
                elif policy == "random_one":
                    exposed = select_exposed_random(labels, rng)
                elif policy == "fine_or_tail_one":
                    exposed = select_exposed_fine_or_tail(labels, parent_map, freq_map)
                else:
                    raise ValueError(f"Unknown policy: {policy}")
                hidden = [label for label in labels if label != exposed]
                # Count hidden fine labels
                for h in hidden:
                    if h in parent_map:
                        hidden_fine_count += 1
            else:
                exposed = labels[0]
                hidden = []

            pairs_data.append(
                {
                    "image_id": image_id,
                    "sub_idx": entry["sub_idx"],
                    "obj_idx": entry["obj_idx"],
                    "all_labels": labels,
                    "exposed_label": exposed,
                    "hidden_labels": hidden,
                    "is_mapped_fine": entry.get("has_mapped_fine", False),
                }
            )

    manifest = {
        "schema_version": 2,
        "split": split,
        "policy": policy,
        "seed": rng.randint(0, 2**31) if policy == "random_one" else 0,
        "parent_map_sha": file_sha256(
            _PROJECT_ROOT / "configs" / "pob_strong_mappings.json"
        ),
        "source_annotation_sha": file_sha256(
            _PROJECT_ROOT / "data" / "VisualGenome" / "rel.json"
        ),
        "train_predicate_frequencies": {str(k): v for k, v in freq_map.items()},
        "num_pairs": total_pairs,
        "num_multi_label_pairs": multi_count,
        "hidden_fine_count": hidden_fine_count,
        "pairs": pairs_data,
    }
    return manifest


def compute_train_frequencies(train_image_ids, rel_data):
    """Compute predicate frequencies from training set."""
    freq = Counter()
    for image_id in train_image_ids:
        key = str(image_id)
        if key not in rel_data:
            key = image_id
        if key not in rel_data:
            continue
        for rel in rel_data[key]:
            pred_idx = int(rel[2]) - 1
            freq[pred_idx] += 1
    return dict(freq)


def main():
    parser = argparse.ArgumentParser(
        description="Build EXP-009 hidden-positive manifests"
    )
    parser.add_argument(
        "--policy",
        choices=["head_or_coarse_one", "random_one", "fine_or_tail_one"],
        default="head_or_coarse_one",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random_one policy and for selecting which seed's manifests to build",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Build manifests for seeds 42, 123, 2027",
    )
    args = parser.parse_args()

    # Load data
    rel_path = _PROJECT_ROOT / "data" / "VisualGenome" / "rel.json"
    with open(rel_path) as f:
        rel_data = json.load(f)

    train_rels = rel_data["train"]
    val_rels = rel_data["val"]

    # Load canonical 5K manifests
    manifest_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "manifests"
    out_dir = manifest_dir  # same output directory

    seeds = [42, 123, 2027] if args.all_seeds else [args.seed]
    parent_map = load_parent_map(_PROJECT_ROOT / "configs" / "pob_strong_mappings.json")

    for seed in seeds:
        rng = random.Random(seed)
        train_manifest_path = manifest_dir / f"gen_sgg_exp001_train5k_seed{seed}.json"
        val_manifest_path = manifest_dir / f"gen_sgg_exp001_val5k_seed{seed}.json"

        if not train_manifest_path.exists():
            print(
                f"SKIP seed {seed}: canonical manifest not found at {train_manifest_path}"
            )
            continue

        train_ids = load_manifest_image_ids(train_manifest_path)
        val_ids = load_manifest_image_ids(val_manifest_path)

        # Compute train-only frequencies
        freq_map = compute_train_frequencies(train_ids, train_rels)

        # Build train manifest
        train_manifest = build_manifest(
            "train", train_ids, train_rels, parent_map, args.policy, rng, freq_map
        )
        train_out = out_dir / f"exp009_train_hidden_{args.policy}_seed{seed}.json"
        with open(train_out, "w") as f:
            json.dump(train_manifest, f, indent=2)
        print(
            f"Train {args.policy} seed={seed}: {train_manifest['num_multi_label_pairs']} multi-label pairs, "
            f"{train_manifest['hidden_fine_count']} hidden fine, "
            f"total {train_manifest['num_pairs']} pairs -> {train_out}"
        )

        # Build val manifest
        val_manifest = build_manifest(
            "val", val_ids, val_rels, parent_map, args.policy, rng, freq_map
        )
        val_out = out_dir / f"exp009_val_hidden_{args.policy}_seed{seed}.json"
        with open(val_out, "w") as f:
            json.dump(val_manifest, f, indent=2)
        print(
            f"Val   {args.policy} seed={seed}: {val_manifest['num_multi_label_pairs']} multi-label pairs, "
            f"{val_manifest['hidden_fine_count']} hidden fine, "
            f"total {val_manifest['num_pairs']} pairs -> {val_out}"
        )

    # Integrity checks for last-built val manifest
    pairs = val_manifest["pairs"]
    keys = [(p["image_id"], p["sub_idx"], p["obj_idx"]) for p in pairs]
    assert len(keys) == len(set(keys)), "DUPLICATE PAIR KEYS!"
    for p in pairs:
        if p["hidden_labels"]:
            assert p["exposed_label"] not in p["hidden_labels"], (
                f"Exposed label {p['exposed_label']} found in hidden {p['hidden_labels']}"
            )
            assert p["exposed_label"] in p["all_labels"], (
                f"Exposed label {p['exposed_label']} not in all_labels {p['all_labels']}"
            )
    print("INTEGRITY CHECKS PASSED")


if __name__ == "__main__":
    main()
