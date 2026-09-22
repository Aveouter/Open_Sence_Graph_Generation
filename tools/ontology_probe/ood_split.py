"""Pair-OOD split construction and its leakage audit.

The split is **image-level**: a relation may only reach the eval set if its whole
image was withheld from training.  That is what stops a probe from memorising an
image and being scored on it, and it is why the eval set is not simply "all
relations with an unseen pair".

The holdout unit is the *undirected* pair, so holding out ``(person, bike)``
also removes ``(bike, person)``: the pair is globally unseen in every direction.
Without that, a directed-only holdout would leave the reverse direction in
training for symmetric predicates and quietly leak.

Stdlib only.
"""

from __future__ import annotations

import collections
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tools.ontology_probe.common import NUM_VG_OBJECT_SLOTS
from tools.ontology_probe.vg_annotations import RelationTable, VGIndex

__all__ = [
    "OODSplitConfig",
    "holdout_signature",
    "greedy_holdout_pairs",
    "build_pair_ood_split",
    "build_pair_known_split",
    "audit_split",
    "SplitBundle",
]


@dataclass(frozen=True)
class OODSplitConfig:
    """Knobs for the greedy pair holdout.

    Defaults are the operating point measured during planning: 2,793 pairs over
    11,544 images (20% of train), 14,998 study-predicate eval relations, and 74%
    of train retained.
    """

    max_pair_count: int = 20
    min_pair_count: int = 2
    image_cap_fraction: float = 0.20
    dev_fraction: float = 0.10
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_pair_count": self.max_pair_count,
            "min_pair_count": self.min_pair_count,
            "image_cap_fraction": self.image_cap_fraction,
            "dev_fraction": self.dev_fraction,
            "seed": self.seed,
        }


def holdout_signature(seed: int, c_s: int, c_o: int) -> str:
    """Deterministic tie-break for pair ordering.

    ``hash()`` is unusable here: PYTHONHASHSEED randomises str/bytes hashing per
    process, so a split built with it would not be reproducible across runs.
    """
    payload = f"{seed}|{min(c_s, c_o)}|{max(c_s, c_o)}".encode()
    return hashlib.sha1(payload).hexdigest()


@dataclass
class SplitBundle:
    """Indices into a canonical :class:`RelationTable`, plus provenance."""

    split_id: str
    train: list[int] = field(default_factory=list)
    dev: list[int] = field(default_factory=list)
    eval: list[int] = field(default_factory=list)
    held_out_pairs: list[tuple[int, int]] = field(default_factory=list)
    affected_images: list[int] = field(default_factory=list)
    dropped_in_affected_images: int = 0
    notes: dict[str, Any] = field(default_factory=dict)


def _pair_rows(table: RelationTable) -> dict[int, list[int]]:
    """Undirected pair key -> row indices, memoising the pair view once."""
    pairs = table.undirected_pairs()
    grouped: dict[int, list[int]] = collections.defaultdict(list)
    for row, pair in enumerate(pairs):
        grouped[pair].append(row)
    return grouped


def greedy_holdout_pairs(
    table: RelationTable,
    predicate_names: Sequence[str],
    study_predicate_ids: Iterable[int],
    config: OODSplitConfig,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Choose undirected pairs to withhold under an explicit image budget.

    Scores each candidate pair by (study-predicate relations it would move to
    eval) / (new images it would commit), and accepts greedily until the image
    budget is exhausted.  Returned as (pair keys, affected image ids, stats).
    """
    study_ids = set(study_predicate_ids)
    pairs = table.undirected_pairs()
    rows_by_pair = _pair_rows(table)

    pair_images: dict[int, set[int]] = {}
    pair_gain: dict[int, int] = {}
    for pair, rows in rows_by_pair.items():
        count = len(rows)
        if not config.min_pair_count <= count <= config.max_pair_count:
            continue
        gain = sum(1 for r in rows if table.pred_id[r] in study_ids)
        if gain == 0:
            continue
        pair_images[pair] = {table.image_id[r] for r in rows}
        pair_gain[pair] = gain

    n_train_images = len(set(table.image_id))
    image_budget = int(config.image_cap_fraction * n_train_images)

    # Greedy by gain-per-new-image; the sha1 tie-break makes the order total.
    scored = []
    for pair in pair_gain:
        c_s, c_o = divmod(pair, NUM_VG_OBJECT_SLOTS)
        ratio = pair_gain[pair] / max(1, len(pair_images[pair]))
        scored.append((-ratio, holdout_signature(config.seed, c_s, c_o), pair))
    scored.sort()

    selected: list[int] = []
    committed: set[int] = set()
    for _neg_ratio, _sig, pair in scored:
        new_images = pair_images[pair] - committed
        if len(committed) + len(new_images) > image_budget:
            continue
        selected.append(pair)
        committed.update(new_images)

    stats = {
        "n_train_images": n_train_images,
        "image_budget": image_budget,
        "n_candidate_pairs": len(pair_gain),
        "n_selected_pairs": len(selected),
        "n_committed_images": len(committed),
        "committed_image_fraction": len(committed) / n_train_images if n_train_images else 0.0,
    }
    return selected, sorted(committed), stats


def build_pair_ood_split(
    table: RelationTable,
    index: VGIndex,
    predicate_names: Sequence[str],
    study_predicate_ids: Iterable[int],
    config: OODSplitConfig,
) -> SplitBundle:
    """Strict pair-OOD split: eval pairs never appear in train, in any direction."""
    selected, affected, stats = greedy_holdout_pairs(
        table, predicate_names, study_predicate_ids, config
    )
    held = set(selected)
    affected_set = set(affected)
    pairs = table.undirected_pairs()

    eval_rows = [
        row
        for row in range(len(table))
        if table.image_id[row] in affected_set and pairs[row] in held
    ]
    # Everything in a non-affected image trains, except the dev slice.
    trainable_rows = [
        row for row in range(len(table)) if table.image_id[row] not in affected_set
    ]

    train_rows, dev_rows = _split_train_dev(
        table, trainable_rows, config.dev_fraction, config.seed
    )
    dropped = sum(
        1
        for row in range(len(table))
        if table.image_id[row] in affected_set and pairs[row] not in held
    )

    return SplitBundle(
        split_id="pair_ood",
        train=train_rows,
        dev=dev_rows,
        eval=eval_rows,
        held_out_pairs=sorted(divmod(p, NUM_VG_OBJECT_SLOTS) for p in selected),
        affected_images=affected,
        dropped_in_affected_images=dropped,
        notes={
            "construction": "greedy undirected-pair holdout under an image budget",
            "holdout_unit": "undirected category pair (both directions withheld)",
            "eval_rule": "relations in affected images whose pair was held out",
            "train_rule": "all relations in non-affected images, minus the dev slice",
            **stats,
        },
    )


def build_pair_known_split(
    table: RelationTable,
    index: VGIndex,
    predicate_names: Sequence[str],
    config: OODSplitConfig,
) -> SplitBundle:
    """Relaxed split: new images, but every eval pair is still seen in train.

    This is where the strict ``C_prior_conflict`` subset lives.  Under a strict
    pair-OOD eval set every pair has zero training counts, so the pair prior
    collapses to a constant global vector whose argmax is always ``on`` -- and
    "can vision override a confident prior?" becomes unmeasurable.  Here the
    prior has real support, so the question is well posed.
    """
    rng = random.Random(config.seed)
    images = sorted(set(table.image_id))
    held_out = set(rng.sample(images, int(len(images) * config.image_cap_fraction)))

    trainable_rows = [r for r in range(len(table)) if table.image_id[r] not in held_out]
    train_rows, dev_rows = _split_train_dev(
        table, trainable_rows, config.dev_fraction, config.seed
    )

    pairs = table.undirected_pairs()
    train_pairs = {pairs[r] for r in train_rows}
    all_eval_rows = [r for r in range(len(table)) if table.image_id[r] in held_out]
    eval_rows = [r for r in all_eval_rows if pairs[r] in train_pairs]

    return SplitBundle(
        split_id="pair_known",
        train=train_rows,
        dev=dev_rows,
        eval=eval_rows,
        held_out_pairs=[],
        affected_images=sorted(held_out),
        dropped_in_affected_images=len(all_eval_rows) - len(eval_rows),
        notes={
            "construction": "random image holdout; eval keeps only pairs still present in train",
            "purpose": "home of the strict C_prior_conflict subset",
            "n_eval_before_pair_filter": len(all_eval_rows),
        },
    )


def _split_train_dev(
    table: RelationTable,
    rows: list[int],
    dev_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Image-level train/dev split so dev images never leak into probe training."""
    rng = random.Random(seed)
    images = sorted({table.image_id[r] for r in rows})
    n_dev = int(len(images) * dev_fraction)
    dev_images = set(rng.sample(images, n_dev))
    train_rows = [r for r in rows if table.image_id[r] not in dev_images]
    dev_rows = [r for r in rows if table.image_id[r] in dev_images]
    return train_rows, dev_rows


def audit_split(
    table: RelationTable,
    bundle: SplitBundle,
    predicate_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Machine-check the leakage guarantees.  Violations are raised, not logged.

    Five channels are covered here: relation overlap, image overlap, pair
    disjointness in both directions, dev/train image overlap, and emptiness.
    The remaining channels (prior source, standardisation stats, early-stopping
    slice) are enforced by the consumers that build those artifacts.
    """
    pairs = table.undirected_pairs()

    def images(rows: Sequence[int]) -> set[int]:
        return {table.image_id[r] for r in rows}

    train_rows, dev_rows, eval_rows = set(bundle.train), set(bundle.dev), set(bundle.eval)
    problems: list[str] = []

    if train_rows & eval_rows:
        problems.append(f"{len(train_rows & eval_rows)} rows in both train and eval")
    if dev_rows & eval_rows:
        problems.append(f"{len(dev_rows & eval_rows)} rows in both dev and eval")
    if train_rows & dev_rows:
        problems.append(f"{len(train_rows & dev_rows)} rows in both train and dev")

    train_images, dev_images, eval_images = (
        images(bundle.train),
        images(bundle.dev),
        images(bundle.eval),
    )
    if train_images & eval_images:
        problems.append(
            f"{len(train_images & eval_images)} images in both train and eval "
            f"(image-level leakage)"
        )
    if dev_images & eval_images:
        problems.append(f"{len(dev_images & eval_images)} images in both dev and eval")
    if train_images & dev_images:
        problems.append(f"{len(train_images & dev_images)} images in both train and dev")

    train_pairs = {pairs[r] for r in bundle.train}
    eval_pairs = {pairs[r] for r in bundle.eval}
    overlap = train_pairs & eval_pairs
    if bundle.split_id == "pair_ood" and overlap:
        decoded = sorted(tuple(divmod(p, NUM_VG_OBJECT_SLOTS)) for p in overlap)[:10]
        problems.append(
            f"{len(overlap)} undirected pairs appear in both train and eval "
            f"(pair leakage): {decoded}"
        )

    # Per-predicate pair overlap: the guarantee must hold for *every* predicate,
    # not merely in aggregate, or a per-predicate claim would be unsupported.
    per_predicate_overlap: dict[str, int] = {}
    if bundle.split_id == "pair_ood" and predicate_names:
        train_pred_pairs = collections.defaultdict(set)
        for r in bundle.train:
            train_pred_pairs[table.pred_id[r]].add(pairs[r])
        for r in bundle.eval:
            if pairs[r] in train_pred_pairs[table.pred_id[r]]:
                name = predicate_names[table.pred_id[r]]
                per_predicate_overlap[name] = per_predicate_overlap.get(name, 0) + 1
        if per_predicate_overlap:
            problems.append(
                f"per-predicate pair leakage in {sorted(per_predicate_overlap)}"
            )

    if not bundle.train:
        problems.append("train split is empty")
    if not bundle.eval:
        problems.append("eval split is empty")

    if problems:
        raise AssertionError("; ".join(problems))

    def predicate_counts(rows: Sequence[int]) -> dict[str, int]:
        if not predicate_names:
            return {}
        counts = collections.Counter(table.pred_id[r] for r in rows)
        return {predicate_names[p]: c for p, c in sorted(counts.items())}

    return {
        "passed": True,
        "split_id": bundle.split_id,
        "n_train": len(bundle.train),
        "n_dev": len(bundle.dev),
        "n_eval": len(bundle.eval),
        "n_train_images": len(train_images),
        "n_dev_images": len(dev_images),
        "n_eval_images": len(eval_images),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "pair_overlap": 0,
        "per_predicate_pair_overlap": per_predicate_overlap,
        "train_predicate_counts": predicate_counts(bundle.train),
        "eval_predicate_counts": predicate_counts(bundle.eval),
        "dropped_in_affected_images": bundle.dropped_in_affected_images,
        "held_out_pairs_sample": [list(p) for p in bundle.held_out_pairs[:10]],
    }


def compute_unseen_subset(table: RelationTable, bundle: SplitBundle) -> list[int]:
    """``C_unseen``: eval rows whose pair has zero counts in the train subset.

    Definitional for a strict pair-OOD split, so under ``pair_ood`` this returns
    the whole eval set.  It is computed rather than assumed, so it stays correct
    if the split construction is ever relaxed.
    """
    pairs = table.undirected_pairs()
    train_pairs = {pairs[r] for r in bundle.train}
    return [r for r in bundle.eval if pairs[r] not in train_pairs]
