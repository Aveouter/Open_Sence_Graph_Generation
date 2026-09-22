"""Core VG150 annotation access for the ontology-probe diagnostic.

Relation triplets in ``rel.json`` are ``[sub_idx, obj_idx, pred_id]`` where the
two indices are **0-based positional indices into that image's annotation list
in file order** -- *not* global COCO annotation ids.  ``pred_id`` is 1..50, with
``rel_categories[0] == "__background__"``.

Every downstream artifact inherits that convention, so it is asserted on load
against the same semantics ``data/dataloaders/coco.py`` relies on.  Stdlib only,
so the tests for this module run in the dependency-free CI job.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.ontology_probe.common import (
    BACKGROUND_PREDICATE_ID,
    NUM_VG_OBJECT_SLOTS,
    NUM_VG_PREDICATES,
    SPLITS,
    read_json,
    status_block,
)

__all__ = [
    "VGIndex",
    "RelationTable",
    "load_vg_index",
    "validate_index_semantics",
    "build_relation_table",
    "sub_object_boxes",
    "union_box",
    "directed_pair_key",
    "undirected_pair_key",
]


def directed_pair_key(c_s: int, c_o: int) -> int:
    """Encode an ordered (subject, object) category pair as one integer."""
    return c_s * NUM_VG_OBJECT_SLOTS + c_o


def undirected_pair_key(c_s: int, c_o: int) -> int:
    """Encode an unordered category pair, so (a,b) and (b,a) share a key."""
    lo, hi = (c_s, c_o) if c_s <= c_o else (c_o, c_s)
    return lo * NUM_VG_OBJECT_SLOTS + hi


@dataclass
class VGIndex:
    """Per-image annotations and relations for one VG150 split."""

    data_root: Path
    split: str
    object_names: dict[int, str]
    rel_categories: list[str]
    image_ids: list[int]
    image_file_names: dict[int, str]
    image_sizes: dict[int, tuple[int, int]]
    ann_labels_by_image: dict[int, list[int]]
    ann_boxes_by_image: dict[int, list[list[float]]] = field(default_factory=dict)
    rels_by_image: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def predicate_names(self) -> list[str]:
        """51 names, index 0 == __background__."""
        return list(self.rel_categories)

    def predicate_name(self, pred_id: int) -> str:
        return self.rel_categories[pred_id]

    def n_objects(self) -> int:
        return sum(len(v) for v in self.ann_labels_by_image.values())

    def n_relations(self) -> int:
        return sum(len(v) for v in self.rels_by_image.values())


def load_vg_index(
    data_root: Path | str,
    split: str,
    with_boxes: bool = False,
) -> VGIndex:
    """Load one split's COCO annotations plus ``rel.json`` relationships.

    ``with_boxes`` additionally materialises absolute xyxy boxes, which the
    geometry features need but the statistics and split builders do not.
    """
    data_root = Path(data_root)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    split_path = data_root / f"{split}.json"
    rel_path = data_root / "rel.json"
    for path in (split_path, rel_path):
        if not path.exists():
            raise FileNotFoundError(
                f"VG150 input missing: {path}. Expected train/val/test.json + "
                f"rel.json under --data-root (see data/README.md)."
            )

    coco = read_json(split_path)
    rel_payload = read_json(rel_path)

    rel_categories = list(rel_payload["rel_categories"])
    if len(rel_categories) != NUM_VG_PREDICATES + 1:
        raise ValueError(
            f"rel_categories has {len(rel_categories)} entries, expected "
            f"{NUM_VG_PREDICATES + 1} (background + 50 predicates)"
        )
    if rel_categories[BACKGROUND_PREDICATE_ID] != "__background__":
        raise ValueError(
            "rel_categories[0] must be '__background__', got "
            f"{rel_categories[BACKGROUND_PREDICATE_ID]!r}"
        )

    object_names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}

    image_ids: list[int] = []
    image_file_names: dict[int, str] = {}
    image_sizes: dict[int, tuple[int, int]] = {}
    for image in coco["images"]:
        image_id = int(image["id"])
        image_ids.append(image_id)
        image_file_names[image_id] = str(image["file_name"])
        image_sizes[image_id] = (int(image["height"]), int(image["width"]))

    # Annotations must be grouped in file order: rel.json's indices are
    # positional into that order, so ordering is part of the contract.
    labels_by_image: dict[int, list[int]] = collections.defaultdict(list)
    boxes_by_image: dict[int, list[list[float]]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        labels_by_image[image_id].append(int(ann["category_id"]))
        if with_boxes:
            boxes_by_image[image_id].append([float(v) for v in ann["bbox"]])

    raw_rels = rel_payload.get(split)
    if raw_rels is None:
        raise KeyError(f"rel.json has no {split!r} section")

    rels_by_image: dict[int, list[tuple[int, int, int]]] = {}
    for raw_key, triplets in raw_rels.items():
        image_id = int(raw_key)
        rels_by_image[image_id] = [
            (int(sub), int(obj), int(pred)) for sub, obj, pred in triplets
        ]

    return VGIndex(
        data_root=data_root,
        split=split,
        object_names=object_names,
        rel_categories=rel_categories,
        image_ids=image_ids,
        image_file_names=image_file_names,
        image_sizes=image_sizes,
        ann_labels_by_image=dict(labels_by_image),
        ann_boxes_by_image={k: v for k, v in boxes_by_image.items()} if with_boxes else {},
        rels_by_image=rels_by_image,
    )


def _bbox_to_xyxy(bbox: Sequence[float]) -> list[float]:
    """COCO [x, y, w, h] -> absolute [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def validate_index_semantics(
    index: VGIndex,
    n_images: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Assert the relation-index convention holds for a random image sample.

    This is the foundation the whole diagnostic rests on: if positional indices
    were off, every split and every prior table downstream would be silently
    wrong rather than loudly broken.
    """
    rng = random.Random(seed)
    pool = [i for i in index.image_ids if index.rels_by_image.get(i)]
    if len(pool) > n_images:
        pool = rng.sample(pool, n_images)

    problems: list[dict[str, Any]] = []
    checked_relations = 0
    for image_id in pool:
        n_ann = len(index.ann_labels_by_image.get(image_id, []))
        for sub_idx, obj_idx, pred_id in index.rels_by_image.get(image_id, []):
            checked_relations += 1
            if not (0 <= sub_idx < n_ann and 0 <= obj_idx < n_ann):
                problems.append(
                    {
                        "image_id": image_id,
                        "n_annotations": n_ann,
                        "sub_idx": sub_idx,
                        "obj_idx": obj_idx,
                        "reason": "index outside annotation list",
                    }
                )
            if not 1 <= pred_id <= NUM_VG_PREDICATES:
                problems.append(
                    {
                        "image_id": image_id,
                        "pred_id": pred_id,
                        "reason": "predicate id outside 1..50",
                    }
                )

    # Every relation-bearing image must also exist in the COCO image list, and
    # vice versa -- a mismatch means rel.json and <split>.json disagree.
    rel_images = set(index.rels_by_image)
    coco_images = set(index.image_ids)
    rel_only = sorted(rel_images - coco_images)
    coco_only = sorted(coco_images - rel_images)

    report: dict[str, Any] = status_block(
        split=index.split,
        data_root=str(index.data_root),
        images_sampled=len(pool),
        relations_checked=checked_relations,
        n_images_coco=len(coco_images),
        n_images_rel=len(rel_images),
        rel_images_missing_from_coco=rel_only[:20],
        coco_images_missing_from_rel=coco_only[:20],
        n_problems=len(problems),
        problems_sample=problems[:20],
        index_convention=(
            "rel.json triplets are [sub_idx, obj_idx, pred_id] with sub_idx/obj_idx "
            "0-based positional into that image's annotation list in file order"
        ),
        passed=not problems and not rel_only,
    )
    return report


@dataclass
class RelationTable:
    """Flat relation rows in a single label space (default: raw VG50).

    Parallel lists rather than a list of row dicts: VG150 train has 315k
    relations and this module is stdlib-only.
    """

    image_id: list[int] = field(default_factory=list)
    sub_idx: list[int] = field(default_factory=list)
    obj_idx: list[int] = field(default_factory=list)
    pred_id: list[int] = field(default_factory=list)
    c_s: list[int] = field(default_factory=list)
    c_o: list[int] = field(default_factory=list)
    _directed_cache: tuple[int, list[int]] | None = field(
        default=None, repr=False, compare=False
    )
    _undirected_cache: tuple[int, list[int]] | None = field(
        default=None, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.pred_id)

    def relation_key(self, i: int) -> tuple[int, int, int, int]:
        """Stable identity of row ``i``; used for leakage audits."""
        return (self.image_id[i], self.sub_idx[i], self.obj_idx[i], self.pred_id[i])

    def relation_keys(self) -> list[tuple[int, int, int, int]]:
        return [self.relation_key(i) for i in range(len(self))]

    def directed_pairs(self) -> list[int]:
        """Ordered (c_s, c_o) pairs, memoised.

        Cached because it is O(n) over 315k train relations: callers that index
        it inside a loop would otherwise be accidentally quadratic.  The cache is
        keyed on the row count so that appending rows cannot serve stale values.

        Build tables through :func:`build_relation_table` or :meth:`subset`.
        Editing ``c_s``/``c_o`` in place without changing the row count is not
        supported -- it would bypass the key and silently keep the old pairs.
        """
        if self._directed_cache is None or self._directed_cache[0] != len(self):
            self._directed_cache = (
                len(self),
                [
                    directed_pair_key(self.c_s[i], self.c_o[i])
                    for i in range(len(self))
                ],
            )
        return self._directed_cache[1]

    def undirected_pairs(self) -> list[int]:
        """Unordered category pairs (memoised), so (a,b) and (b,a) share a key."""
        if self._undirected_cache is None or self._undirected_cache[0] != len(self):
            self._undirected_cache = (
                len(self),
                [
                    undirected_pair_key(self.c_s[i], self.c_o[i])
                    for i in range(len(self))
                ],
            )
        return self._undirected_cache[1]

    def subset(self, indices: Iterable[int]) -> RelationTable:
        idx = list(indices)
        return RelationTable(
            image_id=[self.image_id[i] for i in idx],
            sub_idx=[self.sub_idx[i] for i in idx],
            obj_idx=[self.obj_idx[i] for i in idx],
            pred_id=[self.pred_id[i] for i in idx],
            c_s=[self.c_s[i] for i in idx],
            c_o=[self.c_o[i] for i in idx],
        )

    def pair_to_predicates(self) -> dict[int, set[int]]:
        """Undirected pair key -> set of predicates observed on it."""
        mapping: dict[int, set[int]] = collections.defaultdict(set)
        for i in range(len(self)):
            mapping[undirected_pair_key(self.c_s[i], self.c_o[i])].add(
                self.pred_id[i]
            )
        return dict(mapping)

    def predicate_counts(self) -> collections.Counter:
        return collections.Counter(self.pred_id)


def build_relation_table(index: VGIndex) -> RelationTable:
    """Flatten every relation in ``index`` into a :class:`RelationTable`."""
    table = RelationTable()
    labels = index.ann_labels_by_image
    for image_id, triplets in index.rels_by_image.items():
        image_labels = labels.get(image_id)
        if image_labels is None:
            raise KeyError(
                f"rel.json references image {image_id} absent from "
                f"{index.split}.json"
            )
        for sub_idx, obj_idx, pred_id in triplets:
            table.image_id.append(image_id)
            table.sub_idx.append(sub_idx)
            table.obj_idx.append(obj_idx)
            table.pred_id.append(pred_id)
            table.c_s.append(image_labels[sub_idx])
            table.c_o.append(image_labels[obj_idx])
    return table


def sub_object_boxes(
    index: VGIndex,
    table: RelationTable,
    i: int,
) -> tuple[list[float], list[float], tuple[int, int]]:
    """Absolute xyxy boxes for relation ``i`` plus the image's (height, width).

    Requires ``index`` to have been loaded with ``with_boxes=True``.  Models in
    ``src/models`` expect normalized cxcywh, so consumers convert using the
    returned size; keeping absolute xyxy here is what makes that conversion
    testable.
    """
    if not index.ann_boxes_by_image:
        raise ValueError("sub_object_boxes needs load_vg_index(with_boxes=True)")
    image_id = table.image_id[i]
    boxes = index.ann_boxes_by_image[image_id]
    return (
        _bbox_to_xyxy(boxes[table.sub_idx[i]]),
        _bbox_to_xyxy(boxes[table.obj_idx[i]]),
        index.image_sizes[image_id],
    )


def union_box(sub_xyxy: Sequence[float], obj_xyxy: Sequence[float]) -> list[float]:
    """Smallest axis-aligned box containing both boxes."""
    return [
        min(sub_xyxy[0], obj_xyxy[0]),
        min(sub_xyxy[1], obj_xyxy[1]),
        max(sub_xyxy[2], obj_xyxy[2]),
        max(sub_xyxy[3], obj_xyxy[3]),
    ]
