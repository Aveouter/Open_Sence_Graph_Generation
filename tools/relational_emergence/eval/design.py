"""The partitions Phase IB evaluates across, kept apart from the training loop.

Which rows the encoder saw, which conditions the readout is fit on, and which
rows the endpoints are scored on are three *different* partitions, and collapsing
them is the easiest way to publish a generalization number that measures
memorization. They are separated here so each can be asserted independently:

``context partition``
    which base contexts the encoder trained on. A context's twins and its three
    regimes are near-duplicates, so the unit is the context -- splitting by row
    would put a context on both sides and inflate everything measured across it.
``condition partition``
    which (family, regime) combinations the readout was fit on. Disjoint from the
    ones it is tested on, always, whatever the context split is doing.
``validation partition``
    a slice of the *training* contexts held back for checkpoint selection, so an
    epoch is never chosen on data the model was fitted on.

Pure stdlib and free of torch, so the design can be unit-tested without the model
layer, and so a mistake here shows up as a failing test rather than as a
suspiciously good number.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..simulator.splits import _holdout_count

DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_CONDITION_FRACTION = 0.3


@dataclass(frozen=True)
class ContextPartition:
    """Which contexts train, which select the checkpoint, which are scored."""

    fit: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        for left, right in (("fit", "validation"), ("fit", "test"), ("validation", "test")):
            overlap = set(getattr(self, left)) & set(getattr(self, right))
            if overlap:
                raise ValueError(f"{left} and {right} share contexts: {sorted(overlap)!r}")
        if not self.fit or not self.test:
            raise ValueError("a context partition needs a non-empty fit and test side")

    @property
    def seen(self) -> tuple[str, ...]:
        """Every context the encoder is allowed to learn from."""
        return tuple(sorted((*self.fit, *self.validation)))


def partition_contexts(
    train_keys: tuple[str, ...],
    test_keys: tuple[str, ...],
    seed: int = 0,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> ContextPartition:
    """Split the training side into a fit set and a checkpoint-selection set.

    The validation slice comes out of the *training* side, never the test side:
    selecting a checkpoint on test contexts would make the reported test number a
    validation number, and the whole point of the split is that the test side is
    untouched by every choice the run makes.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    keys = sorted(train_keys)
    if len(keys) < 2:
        raise ValueError("cannot partition fewer than two training contexts")
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * validation_fraction)))
    return ContextPartition(
        fit=tuple(sorted(shuffled[cut:])),
        validation=tuple(sorted(shuffled[:cut])),
        test=tuple(sorted(test_keys)),
    )


@dataclass(frozen=True)
class ConditionPartition:
    """The (family, regime) combinations the readout is fit on and tested on."""

    train: frozenset[str]
    test: frozenset[str]

    def __post_init__(self) -> None:
        overlap = self.train & self.test
        if overlap:
            raise ValueError(f"condition partitions overlap: {sorted(overlap)!r}")
        if not self.train or not self.test:
            raise ValueError("a condition partition needs both sides non-empty")


def condition_of(row: dict) -> str:
    """The CCGP condition. Family crossed with regime, as the plan defines it."""
    return f"{row['appearance_family']}|{row['regime']}"


def family_of_condition(condition: str) -> str:
    return condition.split("|", 1)[0]


def family_condition_partition(
    rows: list[dict], seed: int = 0, fraction: float = DEFAULT_CONDITION_FRACTION
) -> ConditionPartition:
    """Hold out whole appearance families as the readout's test conditions.

    By family rather than by regime or by arbitrary condition, because the family
    is the axis the whole phase is about: ADR 0005 puts it strictly on the
    nuisance side, so a readout that transfers across held-out families is
    transferring something other than appearance. Regimes stay on both sides on
    purpose -- holding one out would make the intervention-richness comparison
    unmeasurable, since every arm would then be tested on a regime it never read.

    The partition is drawn from the families actually present, so a split that
    already holds them out and this partition agree rather than fight.
    """
    families = sorted({row["appearance_family"] for row in rows})
    if len(families) < 2:
        raise ValueError("a family-held-out condition split needs at least two families")
    rng = random.Random(seed)
    shuffled = families[:]
    rng.shuffle(shuffled)
    cut = _holdout_count(len(shuffled), fraction)
    test_families = set(shuffled[:cut])
    return ConditionPartition(
        train=frozenset(
            condition_of(row) for row in rows if row["appearance_family"] not in test_families
        ),
        test=frozenset(
            condition_of(row) for row in rows if row["appearance_family"] in test_families
        ),
    )


def chance_level_by_condition(rows: list[dict]) -> dict[str, float]:
    """The compatibility-aware chance level per condition, as a probability.

    A condition whose rows admit four mechanisms has a chance level of 1/4; one
    that admits three, of 1/3. A condition holds several ``S_0`` tuples with
    different admissible counts, and the level is then the **mean of the
    reciprocals**, not the reciprocal of the mean count.

    That distinction is not pedantic. An earlier version returned a rounded mean
    count, which maps any mix averaging between 3.5 and 4.5 onto 4 and reports
    0.250 where the true level is 0.273 -- a systematic understatement of chance
    on every mixed condition, and so a systematic overstatement of every result
    measured against it.

    Counted from the mechanisms that actually *appear* in the condition's rows
    rather than from a compatibility table. The two agree when the dataset is
    intact, and when they do not, the materialised data is what the readout was
    scored against -- so a disagreement shows up as a shifted baseline instead of
    hiding behind a table that says what should have been generated.
    """
    seen: dict[str, list[int]] = {}
    for row in rows:
        seen.setdefault(condition_of(row), []).append(max(1, len(row["compatible"])))
    return {
        condition: sum(1.0 / count for count in counts) / len(counts)
        for condition, counts in seen.items()
    }


@dataclass(frozen=True)
class TransplantTrial:
    """One target context evaluated against one transplanted source.

    ``wrong_mechanisms`` is the control's source: the *same* source context,
    under every other law it admits. Holding the source fixed is the whole design
    -- if the wrong arm drew a different source too, a gain would be as consistent
    with "some contexts are easier to decode" as with "the code carries the
    mechanism".

    All of the other laws rather than a single arbitrary one, because one fixed
    alternative makes the control a constant: the wrong arm's error would then be
    the same number for every trial sharing a source, and the contrast would
    measure how much the correct latents *vary* rather than whether they are
    better. Averaging over the alternatives gives a control with the same power as
    the treatment and no arbitrary choice in it.
    """

    target_key: str
    source_key: str
    regime: str
    target_mechanism: str
    wrong_mechanisms: tuple[str, ...]

    def key(self) -> str:
        return f"{self.target_key}|{self.regime}|{self.target_mechanism}"


def transplant_plan(
    targets: list[str],
    sources: list[str],
    regimes_by_key: dict[str, list[str]],
    mechanisms_by_key: dict[str, list[str]],
    tuple_by_key: dict[str, str],
    filler_by_key: dict[str, int],
    source_offset: int = 0,
) -> list[TransplantTrial]:
    """Pair every target context with a cross-filler source from the same tuple.

    The source must share the target's ``S_0`` tuple, because compatibility is a
    property of the tuple and a source that does not admit the target's law cannot
    supply a latent for it. It must also have a *different* filler, which is what
    makes the reading a cross-filler one; a same-filler source would let a gain
    come from the decoder recognising the object rather than from the code.

    ``source_offset`` rotates which eligible source is chosen, so repeating the
    whole evaluation with a different offset gives an independent draw without
    changing any trained object -- a cheap robustness check on a pairing that is
    otherwise arbitrary.
    """
    known = set(tuple_by_key)
    unknown = sorted((set(targets) | set(sources)) - known)
    if unknown:
        # A context with no materialised episodes cannot participate, and the
        # caller should have derived its lists from the rows rather than from the
        # split, so this is a mismatch worth stopping on rather than skipping.
        raise ValueError(
            f"{len(unknown)} contexts have no episodes, e.g. {unknown[0]!r}; the "
            "target and source lists must come from the materialised dataset"
        )

    by_tuple: dict[str, list[str]] = {}
    for key in sorted(sources):
        by_tuple.setdefault(tuple_by_key[key], []).append(key)

    trials: list[TransplantTrial] = []
    for target in sorted(targets):
        eligible = [
            key
            for key in by_tuple.get(tuple_by_key[target], [])
            if key != target and filler_by_key[key] != filler_by_key[target]
        ]
        if not eligible:
            continue
        # Rotating the offset is the robustness check on a pairing that is
        # otherwise arbitrary: it changes which context supplies the code and
        # nothing else about the design.
        source = eligible[source_offset % len(eligible)]
        for regime in sorted(regimes_by_key[target]):
            for target_mechanism in sorted(mechanisms_by_key[target]):
                wrong = tuple(
                    name for name in sorted(mechanisms_by_key[target]) if name != target_mechanism
                )
                if not wrong:
                    continue
                trials.append(
                    TransplantTrial(
                        target_key=target,
                        source_key=source,
                        regime=regime,
                        target_mechanism=target_mechanism,
                        wrong_mechanisms=wrong,
                    )
                )
    return trials


def subsample_labels(
    indices: list[int], labels: list[int], per_class: int, seed: int
) -> list[int]:
    """At most ``per_class`` rows per class, drawn without replacement.

    The label-efficiency curve's x-axis. Drawn by class rather than by row so a
    rare mechanism is not crowded out by a common one, which would make the curve
    a statement about class balance rather than about label count.
    """
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if len(indices) != len(labels):
        raise ValueError("indices and labels must be aligned")
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = {}
    for index, label in zip(indices, labels, strict=True):
        buckets.setdefault(label, []).append(index)
    chosen: list[int] = []
    for label in sorted(buckets):
        pool = buckets[label][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    return sorted(chosen)
