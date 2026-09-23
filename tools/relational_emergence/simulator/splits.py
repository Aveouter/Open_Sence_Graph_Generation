"""Dataset splits over base contexts.

All four of the plan's split types. The split unit is the **base context**, never
the episode row: a context's twins and its three intervention regimes are
near-duplicates of one another, so splitting by row would put a context on both
sides of the boundary and inflate every number measured across it.

``iid``
    contexts partitioned at random.
``unseen_filler``
    a subset of filler identities is held out; every context using one goes to
    test, and no filler appears on both sides.
``unseen_family``
    appearance families are held out instead of identities. This is the headline
    axis: ADR 0005 puts the family strictly on the nuisance side, so a model that
    generalizes across it is invariant to appearance rather than extrapolating
    physics.

``pair_recombination``
    every object *identity* is seen in training, but the identity *pair* is not.
    An identity is a bucket over a structural parameter, with bucket edges fixed
    by the declared parameter range rather than by the sample so the classes do
    not move between runs. The held-out pairs are then chosen so that no identity
    disappears from training entirely — otherwise the split would silently become
    an unseen-identity split and measure something easier than it claims.

    An earlier version of this file recorded this split as needing a change to
    the data-generating process, on the grounds that one `Filler` supplies both
    objects. That was wrong: `_structural` draws the probe's and the supporter's
    attributes from *independent* uniforms, so the pair is already a pair — it
    was the split that was missing, not the data.

Every split is asserted disjoint on the axis it claims and on the context id,
because a split that leaks is worse than no split: it reports a generalization
number that measures memorization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .counterfactuals import CounterfactualGroup

SPLIT_KINDS: tuple[str, ...] = (
    "iid",
    "unseen_filler",
    "unseen_family",
    "pair_recombination",
)

# Identity buckets for the pair-recombination split. The edges are the declared
# parameter ranges from `fillers._structural`, not quantiles of a particular
# pool, so a class means the same thing in every run and a held-out pair stays
# held out if the pool is resampled.
PROBE_IDENTITY_RANGE = (0.25, 0.35)
PROBE_IDENTITY_BUCKETS = 2
SUPPORTER_IDENTITY_RANGE = (0.90, 1.10)
SUPPORTER_IDENTITY_BUCKETS = 3

DEFAULT_TEST_FRACTION = 0.3

# Frozen here rather than in Phase1AConfig on purpose. The split seed decides
# which contexts are held out; it does not decide the dataset, and putting it in
# the config would change the provenance digest that every artifact carries --
# invalidating measurements whose data is byte-identical. It is fixed so a split
# cannot be drawn after seeing the number it produces, and moves into the frozen
# config when Phase IB starts generating against it.
DEFAULT_SPLIT_SEED = 0


@dataclass(frozen=True)
class GroupSplit:
    """Which base contexts belong to train and which to test."""

    kind: str
    seed: int
    train: tuple[str, ...]
    test: tuple[str, ...]
    held_out: tuple[str, ...]

    def assert_disjoint(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"{self.kind}: contexts on both sides: {sorted(overlap)!r}")
        if not self.train or not self.test:
            raise ValueError(f"{self.kind}: a side is empty")


def _holdout_count(total: int, fraction: float) -> int:
    """How many classes to hold out, rounded *up* and never all of them.

    int() truncates, so a 30% target over six classes holds out one and a 30%
    target over four holds out one -- both well under the fraction asked for. For
    the pair split that is the difference between a usable test set and a thin
    one. At least one class is always held out, and at least one is always kept.
    """
    return max(1, min(total - 1, math.ceil(total * fraction)))


def _families_of(groups: list[CounterfactualGroup]) -> dict[str, str]:
    return {group.spec.key(): group.filler.nuisance.appearance_family for group in groups}


def _fillers_of(groups: list[CounterfactualGroup]) -> dict[str, int]:
    return {group.spec.key(): group.filler.index for group in groups}


def _bucket(value: float, bounds: tuple[float, float], count: int) -> int:
    low, high = bounds
    index = int((value - low) / (high - low) * count)
    return min(count - 1, max(0, index))


def probe_identity(structural: object) -> str:
    """The probe object's identity class, from a structural attribute."""
    return f"i{_bucket(structural.i_half_u, PROBE_IDENTITY_RANGE, PROBE_IDENTITY_BUCKETS)}"


def supporter_identity(structural: object) -> str:
    """The supporter object's identity class, from a structural attribute."""
    return f"j{_bucket(structural.j_half_u, SUPPORTER_IDENTITY_RANGE, SUPPORTER_IDENTITY_BUCKETS)}"


def identity_pair(group: CounterfactualGroup) -> tuple[str, str]:
    structural = group.filler.structural
    return probe_identity(structural), supporter_identity(structural)


def build_split(
    groups: list[CounterfactualGroup],
    kind: str,
    seed: int = 0,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> GroupSplit:
    """Partition base contexts, holding out whole fillers or families where asked."""
    if kind not in SPLIT_KINDS:
        raise ValueError(f"unknown split {kind!r}, not in {SPLIT_KINDS!r}")
    if not groups:
        raise ValueError("cannot split an empty group set")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")

    keys = sorted(group.spec.key() for group in groups)
    rng = random.Random(seed)
    held_out: tuple[str, ...] = ()

    if kind == "iid":
        shuffled = keys[:]
        rng.shuffle(shuffled)
        cut = _holdout_count(len(shuffled), test_fraction)
        test = tuple(sorted(shuffled[:cut]))
        train = tuple(sorted(shuffled[cut:]))
    elif kind == "unseen_filler":
        by_filler = _fillers_of(groups)
        fillers = sorted({by_filler[key] for key in keys})
        rng.shuffle(fillers)
        cut = _holdout_count(len(fillers), test_fraction)
        held = set(fillers[:cut])
        held_out = tuple(f"filler_{index}" for index in sorted(held))
        test = tuple(sorted(key for key in keys if by_filler[key] in held))
        train = tuple(sorted(key for key in keys if by_filler[key] not in held))
    elif kind == "unseen_family":
        by_family = _families_of(groups)
        families = sorted({by_family[key] for key in keys})
        rng.shuffle(families)
        cut = _holdout_count(len(families), test_fraction)
        held = set(families[:cut])
        held_out = tuple(sorted(held))
        test = tuple(sorted(key for key in keys if by_family[key] in held))
        train = tuple(sorted(key for key in keys if by_family[key] not in held))
    else:
        by_key = {group.spec.key(): identity_pair(group) for group in groups}
        pairs = sorted({by_key[key] for key in keys})
        if len(pairs) < 3:
            raise ValueError(
                f"pair_recombination needs at least three identity pairs, found "
                f"{len(pairs)}: with two, holding one out leaves a single training "
                f"pair and the split degenerates"
            )
        rng.shuffle(pairs)
        cut = _holdout_count(len(pairs), test_fraction)
        held = set(pairs[:cut])
        # Holding out every pair containing an identity would remove that
        # identity from training altogether, quietly turning this into an
        # unseen-identity split. Drop such pairs from the holdout greedily.
        for probe, supporter in sorted(held):
            remaining = [pair for pair in pairs if pair not in held]
            if not any(pair[0] == probe for pair in remaining) or not any(
                pair[1] == supporter for pair in remaining
            ):
                held.discard((probe, supporter))
        if not held:
            raise ValueError("pair_recombination: every candidate holdout would orphan an identity")
        held_out = tuple(f"{probe}/{supporter}" for probe, supporter in sorted(held))
        test = tuple(sorted(key for key in keys if by_key[key] in held))
        train = tuple(sorted(key for key in keys if by_key[key] not in held))

    split = GroupSplit(kind=kind, seed=seed, train=train, test=test, held_out=held_out)
    split.assert_disjoint()
    _assert_axis_held_out(groups, split)
    return split


def _assert_axis_held_out(groups: list[CounterfactualGroup], split: GroupSplit) -> None:
    """The held-out axis must not appear in train. Asserted, never warned."""
    if split.kind == "unseen_filler":
        by_filler = _fillers_of(groups)
        train_fillers = {by_filler[key] for key in split.train}
        test_fillers = {by_filler[key] for key in split.test}
        leaked = train_fillers & test_fillers
        if leaked:
            raise ValueError(f"unseen_filler split leaks fillers {sorted(leaked)!r}")
    elif split.kind == "unseen_family":
        by_family = _families_of(groups)
        train_families = {by_family[key] for key in split.train}
        test_families = {by_family[key] for key in split.test}
        leaked = train_families & test_families
        if leaked:
            raise ValueError(f"unseen_family split leaks families {sorted(leaked)!r}")
    elif split.kind == "pair_recombination":
        by_key = {group.spec.key(): identity_pair(group) for group in groups}
        train_pairs = {by_key[key] for key in split.train}
        test_pairs = {by_key[key] for key in split.test}
        leaked = train_pairs & test_pairs
        if leaked:
            raise ValueError(f"pair_recombination leaks pairs {sorted(leaked)!r}")
        # Both halves of the claim: the pair is novel, but neither identity is.
        train_probes = {pair[0] for pair in train_pairs}
        train_supporters = {pair[1] for pair in train_pairs}
        for probe, supporter in test_pairs:
            if probe not in train_probes:
                raise ValueError(f"pair_recombination holds out identity {probe!r} entirely")
            if supporter not in train_supporters:
                raise ValueError(
                    f"pair_recombination holds out identity {supporter!r} entirely"
                )
