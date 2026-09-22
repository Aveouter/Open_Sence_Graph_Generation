"""Sharded on-disk cache for frozen visual features.

Layout::

    <root>/<encoder>/<split>/shards/shard_00000.h5
    <root>/<encoder>/<split>/shards/shard_00000.done
    <root>/<encoder>/<split>/manifest.json

Per shard, covering a contiguous block of the split's images:

===================  ==========================================
``obj_image_id``     [N_obj] image id of each object
``obj_idx``          [N_obj] positional index into that image's annotations
``obj_feats``        [N_obj, D] float16, context-padded crop
``rel_row``          [M] index into the split's RelationTable
``rel_sub_obj``      [M, 2] local object rows holding subject and object
``rel_union_feats``  [M, D] float16, crop of the subject/object union box
===================  ==========================================

Object features are stored **once per object**, not once per relation: a VG150
train object participates in 315,642/670,591 ~ 0.47 relations on average but is
identical in every one, so per-relation storage would triple the cache and slow
extraction for no information gain.

Sharding makes the run resumable (a killed job reuses finished shards) and lets
each worker write its own file -- h5py is not safe for concurrent writers.

Stdlib-only at import time; h5py/numpy are imported lazily so this module stays
importable in the dependency-free CI job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

CACHE_VERSION = "cache_v1"
DEFAULT_SHARD_IMAGES = 1000

__all__ = [
    "CACHE_VERSION",
    "shard_dir",
    "shard_path",
    "write_shard",
    "is_shard_done",
    "read_manifest",
    "build_manifest",
    "plan_shards",
    "load_visual_features",
    "cache_stats",
]


def shard_dir(root: Path | str, encoder: str, split: str) -> Path:
    return Path(root) / CACHE_VERSION / encoder / split


def shard_path(directory: Path, shard_id: int) -> Path:
    return directory / "shards" / f"shard_{shard_id:05d}.h5"


def done_path(directory: Path, shard_id: int) -> Path:
    return directory / "shards" / f"shard_{shard_id:05d}.done"


def plan_shards(
    image_ids: Sequence[int],
    shard_images: int = DEFAULT_SHARD_IMAGES,
) -> list[list[int]]:
    """Partition images into contiguous shards, preserving split order."""
    return [
        list(image_ids[start : start + shard_images])
        for start in range(0, len(image_ids), shard_images)
    ]


def write_shard(
    directory: Path,
    shard_id: int,
    obj_image_id,
    obj_idx,
    obj_feats,
    rel_row,
    rel_sub_obj,
    rel_union_feats,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write one shard and its completion marker.

    The ``.done`` marker is written last via a rename, so an interrupted shard
    is never mistaken for a finished one.
    """
    import h5py

    directory = Path(directory)
    (directory / "shards").mkdir(parents=True, exist_ok=True)
    path = shard_path(directory, shard_id)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as handle:
        handle.create_dataset("obj_image_id", data=obj_image_id)
        handle.create_dataset("obj_idx", data=obj_idx)
        handle.create_dataset("obj_feats", data=obj_feats)
        handle.create_dataset("rel_row", data=rel_row)
        handle.create_dataset("rel_sub_obj", data=rel_sub_obj)
        handle.create_dataset("rel_union_feats", data=rel_union_feats)
        if meta:
            for key, value in meta.items():
                handle.attrs[key] = value
    os.replace(tmp, path)
    done_path(directory, shard_id).write_text("ok\n", encoding="utf-8")
    return path


def is_shard_done(directory: Path, shard_id: int) -> bool:
    return done_path(Path(directory), shard_id).exists()


def read_manifest(directory: Path | str) -> dict[str, Any] | None:
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(
    directory: Path | str,
    encoder: str,
    split: str,
    n_images: int,
    n_shards: int,
    n_objects: int,
    n_relations: int,
    feature_dim: int,
    crop: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write the cache manifest, including the crop parameters it was built with.

    Crop parameters are recorded because a cache extracted with a different
    context scale is not comparable to one built without it, and nothing in the
    shard files themselves would reveal the difference.
    """
    directory = Path(directory)
    manifest = {
        "cache_version": CACHE_VERSION,
        "encoder": encoder,
        "split": split,
        "n_images": n_images,
        "n_shards": n_shards,
        "n_objects": n_objects,
        "n_relations": n_relations,
        "feature_dim": feature_dim,
        "crop": crop,
        "provenance": provenance,
        "status": "diagnostic_experiment",
        "not_a_reproduction": True,
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def _iter_present_shards(directory: Path) -> Iterator[Path]:
    for path in sorted((Path(directory) / "shards").glob("shard_*.h5")):
        shard_id = int(path.stem.split("_")[1])
        if is_shard_done(directory, shard_id):
            yield path


def load_visual_features(
    directory: Path | str,
    rows: Sequence[int],
) -> dict[str, Any]:
    """Assemble ``{visual_s, visual_o, visual_u}`` for the given relation rows.

    ``rows`` index the split's relation table, so results come back in the order
    asked for -- the caller must not have to re-sort, and a mismatch there would
    silently misalign every feature with its label.
    """
    import h5py
    import numpy as np

    directory = Path(directory)
    wanted = np.asarray(list(rows), dtype=np.int64)
    position = {int(row): i for i, row in enumerate(wanted)}

    visual_s: list[Any] = [None] * len(wanted)
    visual_o: list[Any] = [None] * len(wanted)
    visual_u: list[Any] = [None] * len(wanted)
    found = 0

    for path in _iter_present_shards(directory):
        with h5py.File(path, "r") as handle:
            rel_row = handle["rel_row"][:]
            mask = np.isin(rel_row, wanted, assume_unique=False)
            if not mask.any():
                continue
            local = np.nonzero(mask)[0]
            union = handle["rel_union_feats"][local]
            sub_obj = handle["rel_sub_obj"][local]
            # Read only the object rows this request actually references. A
            # shard's object block is ~300 MB fp16 and a single probe split
            # touches a small fraction of it, so reading the whole block would
            # pull ~18 GB across the train cache for a few hundred MB of use.
            needed = np.unique(sub_obj)
            obj_feats = handle["obj_feats"][needed]
            dense_index = {int(old): new for new, old in enumerate(needed)}
            for k, local_pos in enumerate(local):
                out_index = position[int(rel_row[local_pos])]
                sub_local = dense_index[int(sub_obj[k, 0])]
                obj_local = dense_index[int(sub_obj[k, 1])]
                # Sub/object features come from the object block, so the same
                # object is never encoded twice even across different relations.
                visual_s[out_index] = obj_feats[sub_local]
                visual_o[out_index] = obj_feats[obj_local]
                visual_u[out_index] = union[k]
                found += 1

    missing = [int(r) for i, r in enumerate(wanted) if visual_u[i] is None]
    if missing:
        raise KeyError(
            f"{len(missing)} relation rows are absent from the cache at "
            f"{directory} (e.g. {missing[:5]}); the cache is incomplete or was "
            "built from a different split"
        )

    import torch

    def _stack(items: list[Any]):
        if not items:
            return torch.zeros(0, 0, dtype=torch.float16)
        return torch.from_numpy(np.stack(items)).to(torch.float16)

    return {
        "visual_s": _stack(visual_s),
        "visual_o": _stack(visual_o),
        "visual_u": _stack(visual_u),
    }


def cache_stats(directory: Path | str) -> dict[str, Any]:
    """Summarise an on-disk cache without loading features."""
    import h5py

    directory = Path(directory)
    manifest = read_manifest(directory)
    n_shards = 0
    n_objects = 0
    n_relations = 0
    bytes_total = 0
    for path in _iter_present_shards(directory):
        n_shards += 1
        bytes_total += path.stat().st_size
        with h5py.File(path, "r") as handle:
            n_objects += int(handle["obj_image_id"].shape[0])
            n_relations += int(handle["rel_row"].shape[0])
    return {
        "manifest": manifest,
        "n_shards_present": n_shards,
        "n_objects": n_objects,
        "n_relations": n_relations,
        "bytes": bytes_total,
    }
