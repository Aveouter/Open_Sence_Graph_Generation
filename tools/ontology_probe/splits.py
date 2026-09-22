"""Load split artifacts and verify they still align with the relation table.

Split files store *indices* into the deterministic relation table rather than
repeating 315k tuples.  That is compact but fragile if the table is ever rebuilt
in a different order, so every load re-verifies ``relation_order_sha256`` and
refuses to proceed on a mismatch -- a silent misalignment here would corrupt
every downstream number while looking perfectly healthy.

Stdlib only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import DEFAULT_OUTPUT_ROOT, read_json
from tools.ontology_probe.vg_annotations import RelationTable

PARTS = ("train", "dev", "eval")

__all__ = ["LoadedSplit", "load_split", "verify_relation_order"]


class SplitAlignmentError(AssertionError):
    """Raised when a split file no longer matches the relation table."""


def verify_relation_order(table: RelationTable, payload: dict[str, Any]) -> None:
    """Raise unless the table's relation order matches the recorded hash."""
    from tools.ontology_probe.build_splits import relation_order_sha256

    expected = payload["_meta"]["relation_order_sha256"]
    actual = relation_order_sha256(table)
    if actual != expected:
        raise SplitAlignmentError(
            "relation table order does not match the split artifact "
            f"(recorded {expected[:12]}..., actual {actual[:12]}...); rebuild "
            "the splits with build_splits.py rather than reusing these indices"
        )


@dataclass
class LoadedSplit:
    """A split artifact plus the table it indexes into."""

    split_id: str
    payload: dict[str, Any]
    table: RelationTable

    def rows(self, part: str) -> list[int]:
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        return list(self.payload[part])

    def subset(self, part: str) -> RelationTable:
        return self.table.subset(self.rows(part))

    def unseen_subset(self) -> list[int]:
        """``C_unseen``: eval rows whose pair has no training support."""
        return list(self.payload.get("unseen_subset", []))

    @property
    def meta(self) -> dict[str, Any]:
        return self.payload["_meta"]

    @property
    def counts(self) -> dict[str, Any]:
        return self.payload["counts"]

    @property
    def pair_policy(self) -> dict[str, Any]:
        return self.payload["pair_policy"]

    def label_space_support(self, level: str) -> dict[str, Any]:
        return self.payload["label_spaces"][level]

    def usable_classes(self, level: str) -> list[str]:
        support = self.label_space_support(level)
        return sorted(
            name
            for name in support["eval_class_counts"]
            if name not in support["excluded_classes"]
        )

    def labels_for(self, part: str, cmap: Any) -> list[int]:
        """Canonical class indices for one part, in table order."""
        return [cmap.remap(p) for p in self.table.subset(part).pred_id]


def load_split(
    split_id: str,
    splits_dir: Path | str = DEFAULT_OUTPUT_ROOT / "splits",
    table: RelationTable | None = None,
    verify: bool = True,
) -> LoadedSplit:
    """Load a split artifact, optionally verifying alignment against ``table``."""
    path = Path(splits_dir) / f"split_{split_id}.json"
    payload = read_json(path)
    if verify:
        if table is None:
            raise ValueError("verify=True requires the relation table")
        verify_relation_order(table, payload)
    return LoadedSplit(
        split_id=split_id,
        payload=payload,
        table=table if table is not None else RelationTable(),
    )


def available_splits(splits_dir: Path | str = DEFAULT_OUTPUT_ROOT / "splits") -> list[str]:
    """Split ids present on disk, derived from ``splits_index.json`` when present."""
    splits_dir = Path(splits_dir)
    index_path = splits_dir / "splits_index.json"
    if index_path.exists():
        return sorted(read_json(index_path)["files"])
    return sorted(
        p.stem[len("split_"):] for p in splits_dir.glob("split_*.json")
        if p.name not in {"split_audit.json"}
    )


def pairwise_comparison_classes(
    loaded: LoadedSplit,
    coarse_level: str,
    fine_level: str,
) -> list[str]:
    """Class set to macro-average over when comparing two label spaces.

    Always the coarser space's usable classes.  The finer space's probabilities
    must be pooled onto them first; comparing each space's own class set would
    compare different partitions -- at L2 the on-family children are one class,
    at VG50 they are eight, and the usable-class filters then differ too.
    """
    if fine_level not in loaded.payload["label_spaces"]:
        raise ValueError(f"unknown label space {fine_level!r}")
    return loaded.usable_classes(coarse_level)
