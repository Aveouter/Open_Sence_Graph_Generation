"""
Composition filter for SGG evaluation.

给定一个 composition_split.json，对测试集中每个 relation 判断是 "seen" 还是 "unseen"。
用于在评估时分别报告 seen/unseen 的性能。
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class CompositionFilter:
    """Determines whether a (subject_class, object_class, predicate) triple is seen or unseen."""

    def __init__(self, split_path: str):
        """
        Args:
            split_path: Path to composition_split.json
        """
        with open(split_path) as f:
            data = json.load(f)

        self.seen_compositions: Dict[str, Set[Tuple[int, int]]] = {}
        self.unseen_compositions: Dict[str, Set[Tuple[int, int]]] = {}

        for pred, split_data in data["composition_split"].items():
            self.seen_compositions[pred] = set()
            self.unseen_compositions[pred] = set()
            for s, o in split_data["seen"]:
                self.seen_compositions[pred].add((s, o))
            for s, o in split_data["unseen"]:
                self.unseen_compositions[pred].add((s, o))

        self.metadata = data.get("metadata", {})
        self.group_stats = data.get("group_statistics", {})

    def is_seen(self, subject_class: int, object_class: int, predicate: str) -> Optional[bool]:
        """Returns True if seen, False if unseen, None if unknown predicate."""
        if predicate not in self.seen_compositions:
            return None
        comp = (subject_class, object_class)
        if comp in self.seen_compositions[predicate]:
            return True
        if comp in self.unseen_compositions[predicate]:
            return False
        # Sometimes a composition might not be in either set
        # (e.g., composition that only appears in val/test but not in train)
        return None

    def get_seen_compositions(self, predicate: str) -> Set[Tuple[int, int]]:
        return self.seen_compositions.get(predicate, set())

    def get_unseen_compositions(self, predicate: str) -> Set[Tuple[int, int]]:
        return self.unseen_compositions.get(predicate, set())

    def get_all_predicates(self) -> List[str]:
        return list(self.seen_compositions.keys())


def compute_composition_labels_for_dataset(data_root: str, split_path: str, eval_split: str = "test"):
    """
    为数据集的每个 relation 计算 seen/unseen label。

    Returns:
        List of dicts: [{"image_id": ..., "subject_class": ..., "object_class": ...,
                          "predicate": ..., "is_seen": True/False/None, ...}]
    """
    from make_compositional_split import load_full_data

    data_root = Path(data_root)
    all_relations, cat_id_to_name, predicate_names = load_full_data(data_root)

    composition_filter = CompositionFilter(split_path)

    labeled_relations = []
    stats = {"seen": 0, "unseen": 0, "unknown": 0}

    for rel in all_relations.get(eval_split, []):
        is_seen = composition_filter.is_seen(
            rel["subject_class"], rel["object_class"], rel["predicate"]
        )
        entry = {**rel, "is_seen": is_seen}
        labeled_relations.append(entry)

        if is_seen is True:
            stats["seen"] += 1
        elif is_seen is False:
            stats["unseen"] += 1
        else:
            stats["unknown"] += 1

    print(f"Composition labels for {eval_split}: "
          f"seen={stats['seen']}, unseen={stats['unseen']}, unknown={stats['unknown']}")

    return labeled_relations, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label dataset relations with seen/unseen compositions")
    parser.add_argument("--data_root", type=str, default="data/VisualGenome")
    parser.add_argument("--split_path", type=str,
                        default="data/VisualGenome/composition_splits/split_0/composition_split.json")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    labeled, stats = compute_composition_labels_for_dataset(
        args.data_root, args.split_path, args.eval_split
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"stats": stats, "relations": labeled}, f)
        print(f"Saved to {args.output}")
