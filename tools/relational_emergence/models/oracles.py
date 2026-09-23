"""Level-2 identifiability: the three capacity-identical oracles.

Level-1 asks whether the mechanism ``M`` changes the future at all, with no model
in the loop. Level-2 asks the model-side question: **is a wrong-but-legal ``M``
worse than a known-unknown ``M``?** Three predictors are trained on the same
frozen episodes, with the same trunk, the same depth, the same head, the same
input width, the same split, the same initial weights and the same batch order.
The *only* thing that differs between them is what sits in the four-unit ``M``
slot of the input:

``Base``
    a zero vector -- "there is a mechanism variable and its value is unknown".
``TrueM``
    the one-hot of the mechanism that actually generated the episode.
``ShuffledM``
    the one-hot of a *different* mechanism drawn from the same group's compatible
    set, as a derangement: a permutation with no fixed points. Never a globally
    random mechanism. An out-of-support one-hot would not be a wrong label, it
    would be an impossible one -- the model could reject it out of hand, the
    control would collapse into outlier detection, and ``delta_M`` would measure
    the detection of an impossible input rather than the cost of a wrong belief.

The primary statistic is ``delta_M = L_ShuffledM - L_TrueM``: positive means the
true mechanism is worth something to the predictor. The guard statistic is
``shuffled_vs_base = L_Base - L_ShuffledM``: the shuffled arm must not
*systematically beat* the arm that was told nothing, because a wrong-but-legal
label that is more useful than no label at all would mean the shuffle is
carrying information it should not (a broken derangement, an out-of-support
one-hot, or a label correlated with the context through the pairing).

Reading the two together is what makes the comparison interpretable. ``TrueM``
beating ``ShuffledM`` alone would be consistent with the slot acting as a
lookup table keyed on an arbitrary-but-consistent code; ``TrueM`` and
``ShuffledM`` both being indistinguishable from ``Base`` says the slot is being
ignored. Neither reading is available from one number.

**Checkpoint selection uses the validation prediction loss and nothing else.**
Selecting a checkpoint by any relation-facing quantity -- mechanism recovery, a
linear probe, a ``J_ab`` -- is forbidden by ADR 0007: the control arm would be
tuned toward the outcome the gate is testing, and the gate would be a search
over checkpoints rather than a measurement.

The loss is MSE on targets standardised with **train-only** statistics, frozen
once per seed and shared by all three arms of that seed. The statistics are
re-derived per seed, so a loss is a within-seed quantity: the per-seed losses
are aggregated as *paired* differences, never pooled.

Torch is imported lazily, inside the functions that need it, so this module
imports cleanly in the dependency-free CI job. The dataset layer -- loading, the
impulse block, the derangement, the split, the statistics -- is pure stdlib and
is exercisable there without torch.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..simulator.actions import ActionSchedule, Impulse
from ..simulator.compatibility import MECHANISMS
from ..simulator.geometry import GravityFrame

BASE_MODE = "base"
TRUE_MODE = "true"
SHUFFLED_MODE = "shuffled"
MODES: tuple[str, ...] = (BASE_MODE, TRUE_MODE, SHUFFLED_MODE)

#: The one-hot order of the ``M`` slot. Taken from the compatibility table
#: rather than restated, so the oracle cannot drift from the ontology.
MECHANISM_INDEX: dict[str, int] = {name: index for index, name in enumerate(MECHANISMS)}
M_SLOT_WIDTH = len(MECHANISMS)

#: ``(i_u, i_t, j_u, j_t)`` per step: the world-space impulse on each body.
IMPULSE_CHANNELS = 4

#: The pre-registered floor on replicates. See :class:`OracleConfig`.
MIN_SEEDS = 3

#: Rejection-sampling budget before falling back to a rotation, which is always
#: fixed-point-free. The fallback exists so termination never depends on luck.
DERANGEMENT_ATTEMPTS = 200

#: A column that never moves in the training split has zero standard deviation,
#: and dividing by it yields ``inf``/``nan`` rather than a no-op. The floor keeps
#: a constant column at exactly zero after standardisation. It is not a rare
#: case: the passive regime applies no impulse at all, so every impulse channel
#: in a passive-only dataset is identically zero.
STD_FLOOR = 1e-6

#: Two-sided 95% Student-t quantiles by degrees of freedom. A paired comparison
#: over three to five seeds is a small-sample estimate, and the normal
#: approximation used by the row-level convention in
#: ``tools/ontology_probe/probe_metrics.py`` is stated for 62k-row sets: at
#: ``n = 5`` it would put the interval at 1.96 standard errors where the correct
#: factor is 2.776, and that error is in the direction of passing the gate.
_T95: tuple[float, ...] = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)
_T95_ASYMPTOTIC = 1.960

__all__ = [
    "BASE_MODE",
    "DERANGEMENT_ATTEMPTS",
    "IMPULSE_CHANNELS",
    "MECHANISM_INDEX",
    "MIN_SEEDS",
    "MODES",
    "M_SLOT_WIDTH",
    "SHUFFLED_MODE",
    "STD_FLOOR",
    "TRUE_MODE",
    "ArmReport",
    "Estimate",
    "Level2Result",
    "OracleConfig",
    "OracleDataset",
    "SeedReport",
    "Split",
    "Standardiser",
    "build_oracle",
    "derangement",
    "estimate",
    "group_split",
    "impulse_block",
    "load_oracle_dataset",
    "oracle_dataset_sha256",
    "run_level2",
    "run_seed",
    "shuffled_mechanisms",
    "standardiser",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleDataset:
    """A materialised view of the frozen JSONL, before any tensor exists.

    Python tuples rather than tensors, so the whole data layer stays torch-free
    and can be exercised in the dependency-free CI job. One row here is one
    episode: one base context, one regime, one mechanism.
    """

    horizon: int
    w_x0: int
    w_f: int
    w_a: int
    w_y: int
    x0: tuple[tuple[float, ...], ...]
    f: tuple[tuple[float, ...], ...]
    a: tuple[tuple[float, ...], ...]
    y: tuple[tuple[float, ...], ...]
    group: tuple[str, ...]
    tuple_key: tuple[str, ...]
    regime: tuple[str, ...]
    mechanism: tuple[str, ...]
    compatible: tuple[tuple[str, ...], ...]

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def input_width(self) -> int:
        """``x0 ++ F ++ A ++ M``. Identical for all three arms, by construction."""
        return self.w_x0 + self.w_f + self.w_a + M_SLOT_WIDTH

    @property
    def output_width(self) -> int:
        """``H * 8``: the whole horizon predicted in one shot by the direct head.

        The head emits the flattened future ``X_1 .. X_H``, not one state at a
        time, so the width is the horizon times the outcome-vector width.
        """
        return self.horizon * self.w_y

    def groups(self) -> tuple[str, ...]:
        """Every group key, sorted. The unit of the train/validation split."""
        return tuple(sorted(set(self.group)))


def oracle_dataset_sha256(path: Path | str) -> str:
    """Hash the frozen episodes, so a report can cite the exact input it read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_oracle_dataset(path: Path | str) -> OracleDataset:
    """Read ``oracle_dataset.jsonl`` into tuples, validating its shape.

    The widths are read off the file rather than assumed, and every row is
    required to agree: a dataset with two horizons cannot go into one tensor, and
    discovering that inside the training loop would be a confusing failure far
    from its cause. Nothing here re-samples -- the file is the interface, so the
    split, the compatibility table and the pairing all come from the audited run.
    """
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"oracle dataset {path} is empty")

    first = rows[0]
    horizon = len(first["trajectory"])
    w_x0 = len(first["x0"])
    w_f = len(first["filler_structural"]) + len(first["filler_nuisance"])
    w_a = horizon * IMPULSE_CHANNELS
    w_y = len(first["trajectory"][0])

    x0: list[tuple[float, ...]] = []
    f: list[tuple[float, ...]] = []
    a: list[tuple[float, ...]] = []
    y: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        if len(row["trajectory"]) != horizon:
            raise ValueError(
                f"row {index} has horizon {len(row['trajectory'])}, expected {horizon}"
            )
        if len(row["x0"]) != w_x0:
            raise ValueError(f"row {index} has a {len(row['x0'])}-wide x0")
        features = tuple(row["filler_structural"]) + tuple(row["filler_nuisance"])
        if len(features) != w_f:
            raise ValueError(f"row {index} has a {len(features)}-wide filler block")
        flat_target = tuple(value for state in row["trajectory"] for value in state)
        if len(flat_target) != horizon * w_y:
            raise ValueError(f"row {index} has an inconsistently shaped trajectory")
        named = (row["mechanism"], *row["compatible"])
        unknown = sorted({name for name in named if name not in MECHANISM_INDEX})
        if unknown:
            raise ValueError(f"row {index} names unknown mechanisms: {unknown}")
        x0.append(tuple(row["x0"]))
        f.append(features)
        a.append(tuple(impulse_block(row["impulses"], row["frame_angle_rad"], horizon)))
        y.append(flat_target)

    groups = tuple(row["group"] for row in rows)
    compatible = tuple(tuple(row["compatible"]) for row in rows)
    _check_group_sets(groups, compatible)
    return OracleDataset(
        horizon=horizon,
        w_x0=w_x0,
        w_f=w_f,
        w_a=w_a,
        w_y=w_y,
        x0=tuple(x0),
        f=tuple(f),
        a=tuple(a),
        y=tuple(y),
        group=groups,
        tuple_key=tuple(row["tuple"] for row in rows),
        regime=tuple(row["regime"] for row in rows),
        mechanism=tuple(row["mechanism"] for row in rows),
        compatible=compatible,
    )


def _check_group_sets(
    groups: tuple[str, ...], compatible: tuple[tuple[str, ...], ...]
) -> None:
    """Rows from one group must agree about that group's compatible set.

    The set is what the derangement permutes inside, and the shuffle is drawn
    per group. If two rows of a group disagreed, the drawn permutation would be
    applied against a set one of them does not have, and the resulting label
    could be out of support for that row without anything raising.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for group, legal in zip(groups, compatible, strict=True):
        existing = seen.setdefault(group, legal)
        if existing != legal:
            raise ValueError(
                f"group {group!r} disagrees with itself about its compatible set: "
                f"{sorted(set(existing))!r} vs {sorted(set(legal))!r}"
            )


def impulse_block(
    impulses: Sequence[dict[str, Any]], frame_angle_rad: float, horizon: int
) -> list[float]:
    """``(H, 4)`` flattened: the world-space impulse applied to ``i`` and to ``j``.

    Rebuilt through :class:`~tools.relational_emergence.simulator.actions.ActionSchedule`
    rather than by re-deriving the direction table here, so the block fed to the
    oracle is computed by the same call the simulator's own integration makes.
    That table is the reason ``"up"`` means the same thing in every gravity
    frame, and a private copy here would be the one place a drift could hide.

    No division by mass: the structural features already carry the masses, and
    handing the model ``impulse / mass`` would perform part of the physics for it.
    """
    frame = GravityFrame(frame_angle_rad)
    schedule = ActionSchedule(
        impulses=tuple(
            Impulse(
                target=item["target"],
                direction=item["direction"],
                magnitude=item["magnitude"],
                start=item["start"],
                steps=item["steps"],
            )
            for item in impulses
        )
    )
    block: list[float] = []
    for step in range(horizon):
        impulse_i = schedule.impulse_vector("i", step, frame)
        impulse_j = schedule.impulse_vector("j", step, frame)
        block.extend((impulse_i[0], impulse_i[1], impulse_j[0], impulse_j[1]))
    return block


# ---------------------------------------------------------------------------
# The shuffled arm
# ---------------------------------------------------------------------------


def derangement(labels: Sequence[str], rng: random.Random) -> dict[str, str]:
    """A permutation of ``labels`` with no fixed point.

    Rejection sampling over shuffles, with a rotation as the fallback: a rotation
    by a non-zero offset never fixes a point, so this always returns a
    derangement rather than raising once its retry budget is spent.
    """
    ordered = list(labels)
    if len(ordered) < 2:
        raise ValueError(f"a derangement needs at least two labels, got {ordered!r}")
    shuffles = list(ordered)
    for _ in range(DERANGEMENT_ATTEMPTS):
        rng.shuffle(shuffles)
        if all(a != b for a, b in zip(ordered, shuffles, strict=True)):
            return dict(zip(ordered, shuffles, strict=True))
    offset = rng.randrange(1, len(ordered))
    return {
        label: ordered[(index + offset) % len(ordered)]
        for index, label in enumerate(ordered)
    }


def shuffled_mechanisms(dataset: OracleDataset, seed: int) -> tuple[str, ...]:
    """``M_shuffled`` per row: a derangement *within the row's own group*.

    The permutation is drawn once per group and applied to every row of that
    group, so the shuffled label does not depend on the regime or on anything
    else that varies inside the group. This arm's label distribution is therefore
    identical to ``TrueM``'s and its contexts are identical; the two arms differ
    in the label's truth and in nothing else.

    Drawing a mechanism globally would put labels in the slot that the row's
    ``S_0`` tuple cannot support -- an impossible input, detectable as such.
    Drawing a fresh permutation per row would let the permutation correlate with
    the regime. Both are excluded, and the in-support condition is checked here
    rather than argued.
    """
    rng = random.Random(f"level2-shuffle-{seed}")
    mapping_by_group: dict[str, dict[str, str]] = {}
    for group, legal in zip(dataset.group, dataset.compatible, strict=True):
        if group not in mapping_by_group:
            mapping_by_group[group] = derangement(sorted(set(legal)), rng)

    shuffled: list[str] = []
    for index, mechanism in enumerate(dataset.mechanism):
        legal = set(dataset.compatible[index])
        if mechanism not in legal:
            raise ValueError(
                f"row {index}: mechanism {mechanism!r} is not in the row's own "
                f"compatible set {sorted(legal)!r}"
            )
        drawn = mapping_by_group[dataset.group[index]][mechanism]
        if drawn == mechanism:
            raise ValueError(f"row {index}: the shuffle left {mechanism!r} fixed")
        if drawn not in legal:
            raise ValueError(
                f"row {index}: the shuffle produced {drawn!r}, which is out of "
                f"support for this row ({sorted(legal)!r})"
            )
        shuffled.append(drawn)
    return tuple(shuffled)


# ---------------------------------------------------------------------------
# Split and standardisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A split by counterfactual group, never by row."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]

    def overlaps(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.train_groups) & set(self.val_groups)))


def group_split(dataset: OracleDataset, val_fraction: float, seed: int) -> Split:
    """Split by ``row["group"]``, so a group's twins never straddle the split.

    Twins share ``X_0``, fillers, frame and action schedule and differ only in
    ``M``, so they are the most informative rows in the dataset about each other.
    Splitting by row would put a group's twins on both sides and let validation
    measure memorisation of the context rather than prediction of the outcome --
    by an amount nobody could quantify afterwards.
    """
    groups = list(dataset.groups())
    if len(groups) < 2:
        raise ValueError(
            f"a group split needs at least two groups, the dataset has {len(groups)}"
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    random.Random(f"level2-split-{seed}").shuffle(groups)
    n_val = min(len(groups) - 1, max(1, round(val_fraction * len(groups))))
    val_groups = tuple(sorted(groups[:n_val]))
    train_groups = tuple(sorted(groups[n_val:]))
    val_set = set(val_groups)
    train = tuple(index for index, group in enumerate(dataset.group) if group not in val_set)
    val = tuple(index for index, group in enumerate(dataset.group) if group in val_set)
    return Split(train=train, val=val, train_groups=train_groups, val_groups=val_groups)


@dataclass(frozen=True)
class Standardiser:
    """Train-only mean and standard deviation, frozen before the arms are fit.

    Stored for provenance rather than for reuse: a report citing a loss has to be
    able to say which units that loss is in. ``signature`` is the digest the
    artifact carries, so the numbers stay small and the statistics stay checkable.
    """

    input_mean: tuple[float, ...]
    input_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]

    def signature(self) -> str:
        payload = json.dumps(
            {
                "input_mean": [round(value, 12) for value in self.input_mean],
                "input_std": [round(value, 12) for value in self.input_std],
                "target_mean": [round(value, 12) for value in self.target_mean],
                "target_std": [round(value, 12) for value in self.target_std],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _continuous_row(dataset: OracleDataset, index: int) -> tuple[float, ...]:
    return dataset.x0[index] + dataset.f[index] + dataset.a[index]


def _column_stats(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-column mean and standard deviation, with the degenerate case floored."""
    if not rows:
        raise ValueError("statistics need at least one row")
    width = len(rows[0])
    means: list[float] = []
    deviations: list[float] = []
    for column in range(width):
        values = [row[column] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        deviations.append(max(math.sqrt(variance), STD_FLOOR))
    return tuple(means), tuple(deviations)


def standardiser(dataset: OracleDataset, train: Sequence[int]) -> Standardiser:
    """Freeze the standardisation on the training split's rows alone.

    Validation rows are never consulted here. Deriving the statistics on the
    whole dataset would let a validation row's own scale into the transform
    applied to it -- small, but exactly the kind of favourable bias a gate should
    not be able to buy.
    """
    if not train:
        raise ValueError("cannot standardise on an empty training split")
    input_mean, input_std = _column_stats([_continuous_row(dataset, index) for index in train])
    target_mean, target_std = _column_stats([dataset.y[index] for index in train])
    return Standardiser(
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleConfig:
    """The frozen settings all three arms share.

    ``depth`` hidden layers of ``hidden`` units with ReLU, and a **direct**
    multi-step head producing ``H * 8`` values in one shot. Nothing
    autoregresses: with a rollout, one arm's small early error is fed back as
    input and compounds, and ``delta_M`` would then mix "who knows more about
    ``M``" with "who accumulated error faster". The comparison is about
    information, so the head is direct.
    """

    hidden: int = 128
    depth: int = 3
    epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 64
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "cpu"
    # Rollout mode only, and ignored by the direct head. The rollout is evaluated
    # over this many steps rather than the full horizon: an autoregressive
    # rollout of 60 steps compounds its own error until the loss is dominated by
    # divergence, which measures stability rather than information. Ten is the
    # horizon the Level-1 gate selected from the J_ab saturation curve, so the
    # oracle is scored over the same interval the gate treats as covering the
    # interaction consequence.
    eval_horizon: int = 10
    # The smallest `delta_M` this run must be able to detect, in standardised
    # MSE. Losses are in units where the training-set mean predictor scores 1.0,
    # so 0.05 is a five-percent reduction in prediction error -- large enough
    # that a predictive-utility claim resting on it would mean something, and
    # small enough to be reachable: at five seeds the paired interval's standard
    # error has to be under about 0.014, which this comparison clears.
    #
    # Declared here rather than chosen after the fact, because "delta_M is not
    # significantly positive" and "this run could not have seen an effect this
    # size" look identical in a results table and only the second explains a
    # null. ADR 0007 requires the distinction; M0 closes it the same way. A
    # target of 0.01 was tried first and is not reachable at any feasible seed
    # count -- resolving it would need roughly 260 seeds -- which is itself a
    # reason to state the target rather than assume one.
    delta_m_mde_target: float = 0.05

    def __post_init__(self) -> None:
        # Three is the pre-registered floor, not a convenience: a paired
        # comparison over two replicates has one degree of freedom, and its
        # interval is so wide that it could not distinguish a real effect from
        # noise in either direction. Enforced here so a two-seed run cannot be
        # reported as a gate result at all.
        if len(self.seeds) < MIN_SEEDS:
            raise ValueError(
                f"the Level-2 comparison needs at least {MIN_SEEDS} seeds, "
                f"got {len(self.seeds)}: {list(self.seeds)}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.depth < 1 or self.hidden < 1:
            raise ValueError("the trunk needs at least one hidden layer of at least one unit")

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "depth": self.depth,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "val_fraction": self.val_fraction,
            "weight_decay": self.weight_decay,
            "seeds": list(self.seeds),
            "device": self.device,
        }

    def config_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


@functools.lru_cache(maxsize=1)
def _oracle_class() -> Any:
    """The ``nn.Module`` subclass, defined on first use so torch stays lazy.

    One class object per process, so a state dict written here still describes
    the same architecture when it is read back. The ``M`` slot is part of the
    input vector rather than a separate embedding: three architectures that
    differed in *where* ``M`` entered would not have comparable capacity, and
    being able to say they do is the whole point of the control.
    """

    from torch import nn

    class Oracle(nn.Module):
        """A ReLU MLP over ``x0 ++ F ++ A ++ M`` with a direct ``H * 8`` head."""

        def __init__(self, in_features: int, out_features: int, settings: OracleConfig) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            width = in_features
            for _ in range(settings.depth):
                layers.append(nn.Linear(width, settings.hidden))
                layers.append(nn.ReLU())
                width = settings.hidden
            layers.append(nn.Linear(width, out_features))
            self.trunk = nn.Sequential(*layers)

        def forward(self, features):
            return self.trunk(features)

    return Oracle


def build_oracle(in_features: int, out_features: int, config: OracleConfig) -> Any:
    """Instantiate one arm. Importing torch is the caller's business."""
    oracle = _oracle_class()
    return oracle(in_features, out_features, config)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmReport:
    """One arm, one seed: the selected checkpoint's two losses and where it sat."""

    seed: int
    mode: str
    train_loss: float
    val_loss: float
    best_epoch: int
    epochs: int
    n_parameters: int


@dataclass(frozen=True)
class SeedReport:
    """One seed's three arms, which differ only in the ``M`` slot."""

    seed: int
    n_train: int
    n_val: int
    val_groups: tuple[str, ...]
    standardiser_signature: str
    arms: dict[str, ArmReport]

    def loss(self, mode: str) -> float:
        return self.arms[mode].val_loss


def _continuous_block(dataset: OracleDataset, split: Sequence[int], stats: Standardiser) -> Any:
    """``x0 ++ F ++ A``, standardised per column. Pure input; no ``M`` here."""
    import torch

    return torch.tensor(
        [
            [
                (value - mean) / std
                for value, mean, std in zip(
                    _continuous_row(dataset, index),
                    stats.input_mean,
                    stats.input_std,
                    strict=True,
                )
            ]
            for index in split
        ],
        dtype=torch.float32,
    )


def _attach_m(
    continuous: Any, mode: str, labels: Sequence[str], rows: Sequence[int]
) -> Any:
    """Append the four-unit ``M`` slot to a standardised continuous block.

    ``labels`` is indexed by *dataset* row and ``rows`` says which rows the block
    holds, so the same label sequence serves the training and the validation
    block without either being re-derived -- a validation row carries its own
    label, never a training row's.

    The slot is deliberately outside the standardisation. A one-hot that had been
    rescaled would no longer be the code the ontology defines, and ``Base``'s
    zero vector would stop meaning "unknown" and start meaning "the column mean".
    """
    import torch

    slot = torch.zeros((continuous.shape[0], M_SLOT_WIDTH), dtype=torch.float32)
    if mode != BASE_MODE:
        for position, row in enumerate(rows):
            slot[position, MECHANISM_INDEX[labels[row]]] = 1.0
    return torch.cat([continuous, slot], dim=1)


def _targets(dataset: OracleDataset, split: Sequence[int], stats: Standardiser) -> Any:
    import torch

    return torch.tensor(
        [
            [
                (value - mean) / std
                for value, mean, std in zip(
                    dataset.y[index], stats.target_mean, stats.target_std, strict=True
                )
            ]
            for index in split
        ],
        dtype=torch.float32,
    )


def _mean_mse(prediction: Any, target: Any, batch_size: int) -> float:
    """MSE over standardised targets, in minibatches to bound peak memory."""
    import torch

    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, prediction.shape[0], batch_size):
            stop = min(start + batch_size, prediction.shape[0])
            batch_prediction = prediction[start:stop]
            batch_target = target[start:stop]
            total += float(
                torch.nn.functional.mse_loss(batch_prediction, batch_target, reduction="sum")
            )
            seen += batch_prediction.numel()
    return total / max(seen, 1)


def _fit_arm(
    dataset: OracleDataset,
    split: Split,
    stats: Standardiser,
    train_continuous: Any,
    val_continuous: Any,
    labels: Sequence[str],
    mode: str,
    config: OracleConfig,
    seed: int,
) -> ArmReport:
    import torch

    # Identical weights and identical batch order in every arm of a seed. With
    # only the M slot differing, an arm-to-arm gap cannot be an initialisation or
    # a shuffling artefact -- stronger than merely equal capacity, and cheap.
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(config.device)
    model = build_oracle(dataset.input_width, dataset.output_width, config).to(device)
    train_features = _attach_m(train_continuous, mode, labels, split.train).to(device)
    train_targets = _targets(dataset, split.train, stats).to(device)
    val_features = _attach_m(val_continuous, mode, labels, split.val).to(device)
    val_targets = _targets(dataset, split.val, stats).to(device)
    n_train = train_features.shape[0]
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val = math.inf
    best_state = None
    best_epoch = -1
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(train_features[batch]), train_targets[batch])
            loss.backward()
            optimiser.step()

        # Checkpoint selection: minimum validation *prediction* loss, and nothing
        # else. Selecting on any relation-facing metric -- mechanism recovery, a
        # probe, a J_ab -- is forbidden by ADR 0007, because it would tune each
        # arm toward the outcome the gate is testing and turn the comparison into
        # a search over checkpoints.
        model.eval()
        with torch.no_grad():
            val_loss = _mean_mse(model(val_features), val_targets, config.batch_size)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint was selected; the training loop did not run")
    model.load_state_dict(best_state)
    model.eval()
    return ArmReport(
        seed=seed,
        mode=mode,
        train_loss=_mean_mse(model(train_features), train_targets, config.batch_size),
        val_loss=best_val,
        best_epoch=best_epoch,
        epochs=config.epochs,
        n_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def run_seed(dataset: OracleDataset, config: OracleConfig, seed: int) -> SeedReport:
    """Train all three arms for one seed and report their selected losses.

    The split and the shuffle are drawn once per seed and shared by the arms, so
    the only thing that changes between them is the label. That is what makes
    ``delta_M`` a paired comparison rather than a difference between two models
    that also disagree about the data.
    """
    split = group_split(dataset, config.val_fraction, seed)
    if split.overlaps():
        raise ValueError(f"seed {seed}: the split leaked groups {split.overlaps()!r}")
    shuffled = shuffled_mechanisms(dataset, seed)
    stats = standardiser(dataset, split.train)
    train_continuous = _continuous_block(dataset, split.train, stats)
    val_continuous = _continuous_block(dataset, split.val, stats)

    arms: dict[str, ArmReport] = {}
    for mode in MODES:
        labels = dataset.mechanism if mode == TRUE_MODE else shuffled
        arms[mode] = _fit_arm(
            dataset=dataset,
            split=split,
            stats=stats,
            train_continuous=train_continuous,
            val_continuous=val_continuous,
            labels=labels,
            mode=mode,
            config=config,
            seed=seed,
        )
        report = arms[mode]
        print(
            f"[level2] seed={seed} arm={mode:8s} val={report.val_loss:.6f} "
            f"train={report.train_loss:.6f} best_epoch={report.best_epoch}"
        )
    parameter_counts = {report.n_parameters for report in arms.values()}
    if len(parameter_counts) != 1:
        raise ValueError(
            f"seed {seed}: the arms are not capacity-identical, parameter counts "
            f"{sorted(parameter_counts)!r}"
        )
    return SeedReport(
        seed=seed,
        n_train=len(split.train),
        n_val=len(split.val),
        val_groups=split.val_groups,
        standardiser_signature=stats.signature(),
        arms=arms,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A paired mean over seeds, with a Student-t interval."""

    mean: float
    lo: float
    hi: float
    se: float
    n: int
    ci_method: str = "student_t_se"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "lo": self.lo,
            "hi": self.hi,
            "se": self.se,
            "n": self.n,
            "ci_method": self.ci_method,
        }


def minimum_detectable_effect(
    estimate: Estimate, alpha: float = 0.05, power: float = 0.8
) -> float:
    """The smallest true effect this many seeds could resolve, at the given power.

    ``(t_alpha + z_power) * SE`` on the paired differences. With five seeds the t
    quantile is 2.132 rather than 1.96, and using the normal one would understate
    the floor by about 9% -- in the direction that makes an underpowered run look
    adequate.
    """
    del alpha  # the quantile table is the two-sided 95% one, as ``estimate`` uses
    if estimate.n < 2:
        raise ValueError("a detectable effect needs at least two replicates")
    quantile = _T95[estimate.n - 2] if estimate.n - 1 <= len(_T95) else _T95_ASYMPTOTIC
    z_power = 0.8416212335729143 if power == 0.8 else 1.2815515655446004
    return (quantile + z_power) * estimate.se


def estimate(values: Sequence[float], alpha: float = 0.05) -> Estimate:
    """Mean and two-sided ``1 - alpha`` t interval for a paired difference."""
    n = len(values)
    if n < 2:
        raise ValueError(f"an interval needs at least two seeds, got {n}")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    se = math.sqrt(variance / n)
    quantile = _T95[n - 2] if n - 1 <= len(_T95) else _T95_ASYMPTOTIC
    return Estimate(mean=mean, lo=mean - quantile * se, hi=mean + quantile * se, se=se, n=n)


@dataclass(frozen=True)
class Level2Result:
    """The Level-2 verdict and every number behind it.

    ``passed`` requires both halves of the ADR 0007 criterion:

    * ``TrueM`` significantly better than ``ShuffledM`` -- the lower end of the
      ``delta_M`` interval is above zero, on a two-sided 95% interval, so a
      significant result in the *wrong* direction cannot pass either;
    * ``ShuffledM`` not systematically better than ``Base`` -- the interval for
      ``shuffled_vs_base`` does not lie wholly above zero. "Better" is a *lower*
      loss, so an interval above zero would mean the wrong-but-legal label was
      worth more than no label, which is a leak in the control rather than a
      result about ``M``.

    The reverse comparison -- ``Base`` significantly better than ``ShuffledM`` --
    is reported as a diagnostic and is deliberately **not** gated on. A wrong
    label costing more than no label is a plausible outcome, not a failure, and
    requiring it would make the gate depend on a direction the ADR never asks
    about.
    """

    passed: bool
    delta_m: Estimate
    shuffled_vs_base: Estimate
    mean_losses: dict[str, float]
    per_seed_losses: dict[str, tuple[float, ...]]
    per_seed_delta_m: tuple[float, ...]
    per_seed_shuffled_vs_base: tuple[float, ...]
    seeds: tuple[int, ...]
    n_train_by_seed: tuple[int, ...]
    n_val_by_seed: tuple[int, ...]
    val_groups_by_seed: tuple[tuple[str, ...], ...]
    standardiser_signatures: tuple[str, ...]
    best_epochs: dict[str, tuple[int, ...]]
    n_parameters: int
    base_below_shuffled: bool
    blocked_by: tuple[str, ...]
    mde: float
    mde_target: float
    adequately_powered: bool

    def as_dict(self) -> dict[str, Any]:
        """The artifact payload. The gate's keys are fixed by
        :func:`tools.relational_emergence.audits.protocol.gate_decision`."""
        return {
            "passed": self.passed,
            "delta_M": self.delta_m.mean,
            "delta_M_ci": [self.delta_m.lo, self.delta_m.hi],
            "shuffled_vs_base": self.shuffled_vs_base.mean,
            "trueM_loss": self.mean_losses[TRUE_MODE],
            "shuffledM_loss": self.mean_losses[SHUFFLED_MODE],
            "base_loss": self.mean_losses[BASE_MODE],
            "seeds": list(self.seeds),
            "n_train": self.n_train_by_seed[0],
            "n_val": self.n_val_by_seed[0],
            "delta_M_detail": self.delta_m.as_dict(),
            "shuffled_vs_base_detail": self.shuffled_vs_base.as_dict(),
            "per_seed": {
                "seeds": list(self.seeds),
                "delta_M": list(self.per_seed_delta_m),
                "shuffled_vs_base": list(self.per_seed_shuffled_vs_base),
                "losses": {
                    mode: list(values) for mode, values in sorted(self.per_seed_losses.items())
                },
                "best_epochs": {
                    mode: list(epochs) for mode, epochs in sorted(self.best_epochs.items())
                },
                "n_train": list(self.n_train_by_seed),
                "n_val": list(self.n_val_by_seed),
                "val_groups": [list(groups) for groups in self.val_groups_by_seed],
                "standardiser_sha256": list(self.standardiser_signatures),
            },
            "n_parameters": self.n_parameters,
            "base_significantly_below_shuffled": self.base_below_shuffled,
            "blocked_by": list(self.blocked_by),
            "mde": self.mde,
            "mde_target": self.mde_target,
            "adequately_powered": self.adequately_powered,
        }


DIRECT = "direct"
ROLLOUT = "rollout"
ORACLE_MODES: tuple[str, ...] = (DIRECT, ROLLOUT)


def run_level2(
    dataset: OracleDataset, config: OracleConfig, oracle: str = DIRECT
) -> Level2Result:
    """Train every arm for every seed and reduce to the gate's two numbers.

    ``oracle`` selects the parameterisation, not the comparison: both train the
    same three arms from the same split with the same checkpoint rule and reduce
    to the same paired ``delta_M``. ``direct`` predicts the whole horizon in one
    shot; ``rollout`` predicts one step and is scored over a rolled-out horizon.
    They are not interchangeable measurements, so whichever one produced a
    verdict has to be named in the artifact and in any report citing it.
    """
    if dataset.n_rows < 2:
        raise ValueError("the Level-2 comparison needs more than one episode")
    if oracle not in ORACLE_MODES:
        raise ValueError(f"unknown oracle {oracle!r}, not in {ORACLE_MODES!r}")
    print(
        f"[level2] oracle={oracle} rows={dataset.n_rows} groups={len(dataset.groups())} "
        f"tuples={len(set(dataset.tuple_key))} horizon={dataset.horizon} "
        f"input_width={dataset.input_width} output_width={dataset.output_width}"
    )
    if oracle == ROLLOUT:
        from .rollout_oracle import run_rollout_seed

        reports = tuple(run_rollout_seed(dataset, config, seed) for seed in config.seeds)
    else:
        reports = tuple(run_seed(dataset, config, seed) for seed in config.seeds)

    per_seed_delta = tuple(
        report.loss(SHUFFLED_MODE) - report.loss(TRUE_MODE) for report in reports
    )
    per_seed_vs_base = tuple(
        report.loss(BASE_MODE) - report.loss(SHUFFLED_MODE) for report in reports
    )
    delta_m = estimate(per_seed_delta)
    shuffled_vs_base = estimate(per_seed_vs_base)

    # Minimum detectable effect for the paired comparison: the smallest true
    # delta_M this many seeds could have resolved, at 80% power and one-sided
    # 5%. Reported so a null can be read as "absent" or as "undetectable" rather
    # than leaving the two indistinguishable (ADR 0007).
    mde = minimum_detectable_effect(delta_m)
    adequately_powered = mde <= config.delta_m_mde_target

    blocked: list[str] = []
    if not delta_m.lo > 0.0:
        blocked.append(
            "delta_M_not_significantly_positive"
            if adequately_powered
            else "delta_M_underpowered"
        )
    if shuffled_vs_base.lo > 0.0:
        blocked.append("shuffled_significantly_better_than_base")

    return Level2Result(
        passed=not blocked,
        delta_m=delta_m,
        shuffled_vs_base=shuffled_vs_base,
        mean_losses={
            mode: sum(report.loss(mode) for report in reports) / len(reports) for mode in MODES
        },
        per_seed_losses={
            mode: tuple(report.loss(mode) for report in reports) for mode in MODES
        },
        per_seed_delta_m=per_seed_delta,
        per_seed_shuffled_vs_base=per_seed_vs_base,
        seeds=config.seeds,
        n_train_by_seed=tuple(report.n_train for report in reports),
        n_val_by_seed=tuple(report.n_val for report in reports),
        val_groups_by_seed=tuple(report.val_groups for report in reports),
        standardiser_signatures=tuple(report.standardiser_signature for report in reports),
        best_epochs={
            mode: tuple(report.arms[mode].best_epoch for report in reports) for mode in MODES
        },
        n_parameters=reports[0].arms[TRUE_MODE].n_parameters,
        base_below_shuffled=shuffled_vs_base.hi < 0.0,
        blocked_by=tuple(blocked),
        mde=mde,
        mde_target=config.delta_m_mde_target,
        adequately_powered=adequately_powered,
    )
