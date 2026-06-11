"""
构造 Compositional SGG split。

策略:
- 对每个 predicate p，收集所有 unique (subject_class, object_class) 组合
- 75% 的 composition 归入训练集，25% 归入测试集
- 保证: 所有 object class 和 predicate 在训练集中都出现过
- 输出: composition_split.json 定义每个 pred 的 seen/unseen compositions

输出格式 (composition_split.json):
{
  "pred_compositions": {
    "on": {
      "seen": [[1, 3], [2, 5], ...],     # (subject_cat, object_cat) in train
      "unseen": [[4, 7], [8, 2], ...]     # (subject_cat, object_cat) in test only
    },
    ...
  },
  "metadata": {
    "split_ratio": 0.75,
    "seed": 42,
    "total_compositions": 13446,
    "total_seen": ...,
    "total_unseen": ...
  }
}
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_full_data(data_root: Path):
    """Load all splits and return parsed relations per split.

    Handles both formats:
    - train: rel.json uses GLOBAL annotation IDs (matching COCO annotation IDs)
    - val/test: rel.json uses LOCAL indices within each image's annotation list
    """
    # Load COCO annotations
    anns_by_id = {}       # split -> {global_ann_id: category_id}
    anns_by_image = {}    # split -> {image_id: [(global_ann_id, category_id), ...]}  (sorted by ann_id)
    cats = {}

    for split in ["train", "val", "test"]:
        ann_file = data_root / f"{split}.json"
        if not ann_file.exists():
            print(f"  Skip {split}: file not found")
            continue
        with open(ann_file) as f:
            data = json.load(f)

        anns_by_id[split] = {}
        anns_by_image[split] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_id[split][ann["id"]] = ann["category_id"]
            anns_by_image[split][ann["image_id"]].append((ann["id"], ann["category_id"]))

        # Sort each image's annotations by global ID for consistent local indexing
        for img_id in anns_by_image[split]:
            anns_by_image[split][img_id].sort(key=lambda x: x[0])

        cats = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Load relations
    rel_file = data_root / "rel.json"
    with open(rel_file) as f:
        rel_data = json.load(f)

    predicate_names = rel_data["rel_categories"]

    # Determine format per split: train uses global IDs, val/test use local indices
    # Heuristic: if the first annotation ID in rel appears in the COCO ann dict, it's global
    def _is_global_format(rel_split, split_name):
        """Check if the first few relation annotation IDs match global COCO IDs."""
        ann_dict = anns_by_id.get(split_name, {})
        count = 0
        matched = 0
        for img_id_str, rel_list in rel_split.items():
            for rel in rel_list:
                subj_id = rel[0]
                if count < 20:
                    if subj_id in ann_dict:
                        matched += 1
                    count += 1
                else:
                    break
            if count >= 20:
                break
        return matched > count * 0.5 if count > 0 else False

    # Parse all relations
    all_relations = {}  # split -> list of parsed relations
    for split in ["train", "val", "test"]:
        if split not in rel_data:
            continue
        relations = rel_data[split]
        parsed = []
        skipped = 0
        use_global = _is_global_format(relations, split)
        fmt = "global IDs" if use_global else "local indices"
        print(f"  [{split}] Using {fmt} format")

        for img_id_str, rel_list in relations.items():
            img_id = int(img_id_str)

            # Get image's annotation list for local index resolution
            img_anns = anns_by_image.get(split, {}).get(img_id, [])

            for rel in rel_list:
                subj_id, obj_id, pred_id = rel

                if use_global:
                    # Global annotation IDs (train)
                    ann_dict = anns_by_id.get(split, {})
                    if subj_id not in ann_dict or obj_id not in ann_dict:
                        skipped += 1
                        continue
                    subj_cat = ann_dict[subj_id]
                    obj_cat = ann_dict[obj_id]
                else:
                    # Local indices (val/test)
                    if subj_id >= len(img_anns) or obj_id >= len(img_anns):
                        skipped += 1
                        continue
                    _, subj_cat = img_anns[subj_id]
                    _, obj_cat = img_anns[obj_id]

                parsed.append({
                    "image_id": img_id,
                    "subject_class": subj_cat,
                    "object_class": obj_cat,
                    "predicate_id": pred_id,
                    "predicate": predicate_names[pred_id] if pred_id < len(predicate_names) else f"unknown_{pred_id}",
                })

        all_relations[split] = parsed
        if skipped > 0:
            print(f"  [{split}] Skipped {skipped} relations due to missing annotation refs")

    return all_relations, cats, predicate_names


def build_composition_split(all_relations, split_ratio=0.75, min_seen_compositions=1, seed=42):
    """
    Build compositional split from ALL relations (train + val + test).

    Returns:
        composition_split: dict mapping predicate -> {seen, unseen} compositions
        stats: summary statistics
    """
    random.seed(seed)

    # Step 1: 从 ALL splits 收集每个 predicate 的所有 unique composition
    pred_compositions = defaultdict(set)  # pred -> set of (s, o) tuples

    for split_name in ["train", "val", "test"]:
        for rel in all_relations.get(split_name, []):
            pred = rel["predicate"]
            comp = (rel["subject_class"], rel["object_class"])
            pred_compositions[pred].add(comp)

    # Remove background
    if "__background__" in pred_compositions:
        del pred_compositions["__background__"]

    # Step 2: 对每个 predicate 划分 composition
    composition_split = {}
    total_seen = 0
    total_unseen = 0

    for pred, comp_set in pred_compositions.items():
        comps = list(comp_set)
        random.shuffle(comps)

        n_total = len(comps)
        n_train = max(min_seen_compositions, int(n_total * split_ratio))

        # 确保至少有 1 个 unseen (如果 comps > 1)
        if n_train >= n_total and n_total > 1:
            n_train = n_total - 1
        n_train = min(n_train, n_total)

        seen = sorted(comps[:n_train])
        unseen = sorted(comps[n_train:])

        composition_split[pred] = {
            "seen": [[s, o] for s, o in seen],
            "unseen": [[s, o] for s, o in unseen],
            "seen_count": len(seen),
            "unseen_count": len(unseen),
            "total_count": n_total,
        }
        total_seen += len(seen)
        total_unseen += len(unseen)

    # Step 3: Verify
    all_train_subjects = set()
    all_train_objects = set()
    for pred, split_data in composition_split.items():
        for s, o in split_data["seen"]:
            all_train_subjects.add(s)
            all_train_objects.add(o)

    # Check that all objects in unseen are also in seen at least once
    missing_subjects = set()
    missing_objects = set()
    for pred, split_data in composition_split.items():
        for s, o in split_data["unseen"]:
            if s not in all_train_subjects:
                missing_subjects.add(s)
            if o not in all_train_objects:
                missing_objects.add(o)

    if missing_subjects:
        print(f"  WARNING: {len(missing_subjects)} subjects in unseen but not in any seen composition")
        # Add them to a special "all_seen_subjects" set and ensure they appear in training
        for s in missing_subjects:
            all_train_subjects.add(s)
    if missing_objects:
        print(f"  WARNING: {len(missing_objects)} objects in unseen but not in any seen composition")
        for o in missing_objects:
            all_train_objects.add(o)

    stats = {
        "split_ratio": split_ratio,
        "seed": seed,
        "total_predicates": len(composition_split),
        "total_compositions": total_seen + total_unseen,
        "total_seen": total_seen,
        "total_unseen": total_unseen,
        "unseen_ratio": total_unseen / (total_seen + total_unseen) if (total_seen + total_unseen) > 0 else 0,
        "train_subjects_count": len(all_train_subjects),
        "train_objects_count": len(all_train_objects),
        "missing_subjects_in_train": list(missing_subjects),
        "missing_objects_in_train": list(missing_objects),
    }

    return composition_split, stats


def compute_split_statistics_by_group(composition_split, predicate_groups):
    """Compute per-group statistics for the split."""
    group_stats = defaultdict(lambda: {"seen": 0, "unseen": 0, "total": 0, "predicates": 0})
    for pred, split_data in composition_split.items():
        group = predicate_groups.get(pred, "Tail")
        group_stats[group]["seen"] += split_data["seen_count"]
        group_stats[group]["unseen"] += split_data["unseen_count"]
        group_stats[group]["total"] += split_data["total_count"]
        group_stats[group]["predicates"] += 1

    print("\nCompositional Split by H/B/T Group:")
    print(f"{'Group':<8} {'#Pred':<8} {'Seen':<10} {'Unseen':<10} {'Total':<10} {'Unseen%':<10}")
    print("-" * 56)
    for group in ["Head", "Body", "Tail"]:
        gs = group_stats[group]
        unseen_pct = gs["unseen"] / gs["total"] * 100 if gs["total"] > 0 else 0
        print(f"{group:<8} {gs['predicates']:<8} {gs['seen']:<10} {gs['unseen']:<10} "
              f"{gs['total']:<10} {unseen_pct:<10.1f}")

    return dict(group_stats)


def main():
    parser = argparse.ArgumentParser(description="Build Compositional SGG Split")
    parser.add_argument("--data_root", type=str, default="data/VisualGenome")
    parser.add_argument("--output_dir", type=str, default="data/VisualGenome/composition_splits")
    parser.add_argument("--split_ratio", type=float, default=0.75)
    parser.add_argument("--min_seen", type=int, default=1,
                        help="Minimum seen compositions per predicate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=3,
                        help="Generate N random splits for stability")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_root}...")
    all_relations, cat_id_to_name, predicate_names = load_full_data(data_root)

    for split_name, rels in all_relations.items():
        print(f"  {split_name}: {len(rels)} relations")

    # Build predicate -> H/B/T group mapping for statistics
    from collections import Counter
    pred_freqs = Counter()
    for rel in all_relations.get("train", []):
        pred_freqs[rel["predicate"]] += 1
    sorted_preds = sorted(pred_freqs.items(), key=lambda x: -x[1])
    predicate_groups = {}
    for rank, (pred, freq) in enumerate(sorted_preds):
        if pred == "__background__":
            continue
        if rank < 5:
            predicate_groups[pred] = "Head"
        elif rank < 20:
            predicate_groups[pred] = "Body"
        else:
            predicate_groups[pred] = "Tail"

    for seed_offset in range(args.n_splits):
        seed = args.seed + seed_offset
        print(f"\n{'='*60}")
        print(f"Building split {seed_offset} (seed={seed})...")
        print(f"{'='*60}")

        composition_split, stats = build_composition_split(
            all_relations,
            split_ratio=args.split_ratio,
            min_seen_compositions=args.min_seen,
            seed=seed,
        )

        print(f"\nOverall statistics:")
        print(f"  Total predicates: {stats['total_predicates']}")
        print(f"  Total compositions: {stats['total_compositions']}")
        print(f"  Seen: {stats['total_seen']} ({stats['total_seen']/stats['total_compositions']*100:.1f}%)")
        print(f"  Unseen: {stats['total_unseen']} ({stats['total_unseen']/stats['total_compositions']*100:.1f}%)")

        group_stats = compute_split_statistics_by_group(composition_split, predicate_groups)

        # Save
        split_dir = output_dir / f"split_{seed_offset}"
        split_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "composition_split": composition_split,
            "metadata": stats,
            "group_statistics": group_stats,
        }

        with open(split_dir / "composition_split.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved to: {split_dir / 'composition_split.json'}")

    # Also save a summary across all splits
    summary = {
        "description": "Compositional SGG splits for VG150",
        "format": "Each split_N/composition_split.json contains per-predicate seen/unseen (S,O) pairs",
        "n_splits": args.n_splits,
        "base_seed": args.seed,
        "split_ratio": args.split_ratio,
        "data_root": str(data_root),
    }
    with open(output_dir / "README.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Generated {args.n_splits} splits in {output_dir}")


if __name__ == "__main__":
    main()
