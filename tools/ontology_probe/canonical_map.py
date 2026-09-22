"""Canonical predicate label spaces and fine->canonical probability pooling.

Two levels are defined, deliberately kept separate so that the ontology
experiment can tell *annotation noise* apart from *semantic entailment*:

``L1_noise``
    Merges only within-lemma surface variants -- ``lying on``/``laying on``
    (spelling) and ``wearing``/``wears`` (tense).  No entailment claim is made,
    so a gain here can only be annotation noise being absorbed.

``L2_entail``
    ``L1_noise`` plus the eight frozen strong mappings in
    ``configs/pob_strong_mappings.json`` (seven on-family children -> ``on``,
    ``covered in`` -> ``in``).  A gain here may be semantic generalisation.

If ``Δ_ontology(L1) ≈ 0`` while ``Δ_ontology(L2) > 0``, canonicalisation is
absorbing semantics rather than noise, and the claim that the ontology itself
blocks generalisation is not supported.

Stdlib only.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.ontology_probe.common import REPO_ROOT, read_json

DEFAULT_CANONICAL_MAP_PATH = REPO_ROOT / "configs" / "predicate_canonical_map_vg150.json"

#: Merge kinds, most conservative first.
KIND_SINGLETON = "singleton"
KIND_NOISE = "noise_only"
KIND_ENTAILMENT = "semantic_entailment"

__all__ = [
    "CanonicalMap",
    "CanonicalMapError",
    "identity_map",
    "level_map",
    "load_canonical_map",
    "DEFAULT_CANONICAL_MAP_PATH",
    "KIND_SINGLETON",
    "KIND_NOISE",
    "KIND_ENTAILMENT",
]


class CanonicalMapError(ValueError):
    """Raised when a canonical map is malformed or inconsistent."""


@dataclass(frozen=True)
class CanonicalMap:
    """A mapping from the 50 fine VG predicates onto canonical classes."""

    level: str
    class_names: tuple[str, ...]
    #: fine predicate id (1..50) -> canonical class index (0..C-1)
    id_map: dict[int, int]
    #: fine predicate id -> merge kind of the class it lands in
    kinds: dict[int, str]
    members: tuple[tuple[int, ...], ...]
    background_name: str = "__background__"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def fine_ids(self) -> list[int]:
        return sorted(self.id_map)

    def remap(self, pred_id: int) -> int:
        """Map one fine predicate id to its canonical class index."""
        try:
            return self.id_map[pred_id]
        except KeyError as exc:
            raise CanonicalMapError(f"predicate id {pred_id} not in map") from exc

    def remap_all(self, pred_ids: Iterable[int]) -> list[int]:
        return [self.remap(p) for p in pred_ids]

    def class_of_predicate(self, pred_id: int) -> tuple[int, ...]:
        return self.members[self.remap(pred_id)]

    def is_merged(self, pred_id: int) -> bool:
        """True when the predicate shares its canonical class with another."""
        return len(self.class_of_predicate(pred_id)) > 1

    def merged_fine_ids(self) -> set[int]:
        return {p for p in self.id_map if self.is_merged(p)}

    def untouched_fine_ids(self) -> set[int]:
        """Predicates whose canonical class is a singleton.

        For these the *label mapping* is the identity.  Note this does not make
        their predicted *accuracy* identical across label spaces: a pooled
        distribution can lose an argmax to a merged class that outvotes a single
        predicate, so native and pooled accuracy are still reported separately.
        """
        return {p for p in self.id_map if not self.is_merged(p)}

    def assignment_matrix(self) -> list[list[float]]:
        """``[num_fine, n_classes]`` one-hot matrix; rows are fine predicate ids.

        Row index is ``pred_id - 1`` (fine predicates are 1..50).  Multiply a
        fine probability matrix by this to pool into canonical classes.
        """
        n_fine = max(self.id_map) - min(self.id_map) + 1
        matrix = [[0.0] * self.n_classes for _ in range(n_fine)]
        for pred_id, class_idx in self.id_map.items():
            matrix[pred_id - min(self.id_map)][class_idx] = 1.0
        return matrix

    def pool_probs(self, fine_probs: Sequence[float]) -> list[float]:
        """Pool one fine probability row (index 0 == predicate id 1) by summing.

        Summing is exact for both argmax and NLL, unlike re-normalising a
        coarsened distribution, and it is the only direction that is well
        defined (fine -> coarse, never the reverse).
        """
        if len(fine_probs) != len(self.id_map):
            raise CanonicalMapError(
                f"expected {len(self.id_map)} fine probabilities, got {len(fine_probs)}"
            )
        pooled = [0.0] * self.n_classes
        for pred_id, class_idx in self.id_map.items():
            pooled[class_idx] += float(fine_probs[pred_id - 1])
        return pooled

    def class_sizes(self) -> dict[str, int]:
        return {
            name: len(self.members[i]) for i, name in enumerate(self.class_names)
        }


def identity_map(predicate_names: Sequence[str], level: str = "vg50") -> CanonicalMap:
    """The raw VG50 space as a :class:`CanonicalMap` with all-singleton classes.

    Lets every downstream stage treat the uncanonicalised baseline as just
    another label space instead of special-casing it, which is what keeps the
    Δ_ontology comparison from quietly using two different code paths.
    """
    fine_ids = [i for i in range(1, len(predicate_names))]
    return CanonicalMap(
        level=level,
        class_names=tuple(predicate_names[i] for i in fine_ids),
        id_map={pred_id: class_idx for class_idx, pred_id in enumerate(fine_ids)},
        kinds={pred_id: KIND_SINGLETON for pred_id in fine_ids},
        members=tuple((pred_id,) for pred_id in fine_ids),
        background_name=predicate_names[0],
    )


def load_canonical_map(
    path: Path | str = DEFAULT_CANONICAL_MAP_PATH,
    level: str = "L2_entail",
) -> CanonicalMap:
    """Load one level of the frozen canonical map."""
    payload = read_json(Path(path))
    levels = list(payload.get("levels", []))
    if level not in levels:
        raise CanonicalMapError(f"level {level!r} not in {levels}")

    class_names = payload["class_names"][level]
    raw_id_map = payload["id_maps"][level]
    entries = {int(e["class_id"]): e for e in payload["classes"][level]}

    id_map = {int(k): int(v) for k, v in raw_id_map.items()}
    members: list[tuple[int, ...]] = []
    kinds: dict[int, str] = {}
    for class_idx in range(len(class_names)):
        entry = entries.get(class_idx)
        if entry is None:
            raise CanonicalMapError(f"{level}: missing class entry {class_idx}")
        member_ids = tuple(int(v) for v in entry["vg_ids"])
        members.append(member_ids)
        for pred_id in member_ids:
            kinds[pred_id] = str(entry["kind"])

    return CanonicalMap(
        level=level,
        class_names=tuple(class_names),
        id_map=id_map,
        kinds=kinds,
        members=tuple(members),
    )


def level_map(
    level: str,
    map_path: Path | str = DEFAULT_CANONICAL_MAP_PATH,
) -> CanonicalMap:
    """Resolve a label-space name to a :class:`CanonicalMap`.

    ``vg50`` is not stored in the config -- it is the identity space -- so
    without this helper every call site has to special-case it, which is how the
    two spaces end up going through different code paths.
    """
    if level == "vg50":
        payload = read_json(Path(map_path))
        return identity_map(payload["_meta"]["predicate_names"])
    return load_canonical_map(map_path, level)


def validate_canonical_map(
    cmap: CanonicalMap,
    predicate_names: Sequence[str],
    expected_class_count: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a loaded map.

    ``predicate_names`` is indexed by predicate id, so it includes
    ``__background__`` at index 0 and is 51 long.
    """
    errors: list[str] = []
    warnings: list[str] = []

    n_fine = len(predicate_names) - 1
    if sorted(cmap.id_map) != list(range(1, n_fine + 1)):
        missing = set(range(1, n_fine + 1)) - set(cmap.id_map)
        extra = set(cmap.id_map) - set(range(1, n_fine + 1))
        errors.append(f"id_map must cover 1..{n_fine}; missing={sorted(missing)} extra={sorted(extra)}")

    # Checked first: every later loop indexes members/class_names by class_idx,
    # and a validator must report malformed input rather than raise on it.
    if len(cmap.members) != cmap.n_classes:
        errors.append(
            f"members has {len(cmap.members)} entries but there are "
            f"{cmap.n_classes} class names"
        )

    # Every class index must be in range.
    for pred_id, class_idx in cmap.id_map.items():
        if not 0 <= class_idx < cmap.n_classes:
            errors.append(f"predicate {pred_id} maps to out-of-range class {class_idx}")

    for class_idx, member_ids in enumerate(cmap.members):
        label = (
            repr(cmap.class_names[class_idx])
            if class_idx < len(cmap.class_names)
            else "<no name>"
        )
        if not member_ids:
            errors.append(f"class {class_idx} ({label}) is empty")
        unknown = [p for p in member_ids if p not in cmap.id_map]
        if unknown:
            errors.append(f"class {class_idx} lists unknown predicate ids {unknown}")
        elif any(cmap.id_map[p] != class_idx for p in member_ids):
            errors.append(f"class {class_idx} disagrees with id_map for {member_ids}")

    # Classes must partition the fine predicates exactly once.
    flattened = [p for group in cmap.members for p in group]
    if sorted(flattened) != list(range(1, n_fine + 1)):
        duplicates = [p for p, c in collections.Counter(flattened).items() if c > 1]
        if duplicates:
            errors.append(f"predicate ids appear in more than one class: {sorted(duplicates)}")
        else:
            errors.append("classes do not cover every predicate id exactly once")

    if len(set(cmap.class_names)) != cmap.n_classes:
        errors.append("canonical class names are not unique")

    if expected_class_count is not None and cmap.n_classes != expected_class_count:
        errors.append(
            f"{cmap.level} has {cmap.n_classes} classes, expected {expected_class_count}"
        )

    # Sanity: canonical names should look like one of their members, or the map
    # is inventing vocabulary rather than merging existing predicates.
    for class_idx, name in enumerate(cmap.class_names):
        if class_idx >= len(cmap.members):
            break
        member_names = {
            predicate_names[p]
            for p in cmap.members[class_idx]
            if 0 < p < len(predicate_names)
        }
        if member_names and name not in member_names:
            warnings.append(
                f"class {class_idx} name {name!r} is not one of its members "
                f"{sorted(member_names)}"
            )

    return errors, warnings


def merge_kind_of(cmap: CanonicalMap, pred_id: int) -> str:
    return cmap.kinds[pred_id]


def summarise(cmap: CanonicalMap) -> dict[str, Any]:
    """Report-ready summary of one level."""
    sizes = cmap.class_sizes()
    merged = {name: size for name, size in sizes.items() if size > 1}
    return {
        "level": cmap.level,
        "n_classes": cmap.n_classes,
        "n_merged_classes": len(merged),
        "merged_classes": merged,
        "n_predicates_merged": len(cmap.merged_fine_ids()),
        "n_predicates_untouched": len(cmap.untouched_fine_ids()),
        "kinds": {
            kind: sum(
                1 for p in cmap.id_map if cmap.kinds[p] == kind
            )
            for kind in sorted(set(cmap.kinds.values()))
        },
    }
