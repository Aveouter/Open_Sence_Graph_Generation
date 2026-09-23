"""External probing actions and the three intervention regimes.

An action is a low-dimensional impulse: a target body, a direction expressed in
the gravity frame rather than in world coordinates, a momentum magnitude, and a
duration in steps. Directions are frame-relative so that the same policy is
meaningful under a rotated gravity frame, and the frame is a world property
rather than a mechanism property, so this stays mechanism-independent.

The policies also carry the requirement that they never consult the mechanism.
Nothing here receives one, and there are no legality gates: an action that a
constraint blocks produces a contact response, and that response *is* the
signal. Gating actions on the mechanism would put ``P(A | M) != P(A)`` and
invalidate every leakage probe built on top.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .geometry import GravityFrame, Vec2

PASSIVE = "passive"
WEAK = "weak"
RICH = "rich"
REGIME_NAMES: tuple[str, ...] = (PASSIVE, WEAK, RICH)

LATERAL_DIRECTIONS: tuple[str, ...] = ("lateral_pos", "lateral_neg")
ALL_DIRECTIONS: tuple[str, ...] = (
    "up",
    "down",
    "lateral_pos",
    "lateral_neg",
    "diagonal_up_pos",
    "diagonal_up_neg",
)


@dataclass(frozen=True)
class Impulse:
    target: str
    direction: str
    magnitude: float
    start: int
    steps: int

    def __post_init__(self) -> None:
        if self.target not in ("i", "j"):
            raise ValueError(f"unknown impulse target {self.target!r}")
        if self.direction not in ALL_DIRECTIONS:
            raise ValueError(f"unknown impulse direction {self.direction!r}")
        if self.magnitude < 0.0:
            raise ValueError("impulse magnitude must be non-negative")
        if self.steps < 1:
            raise ValueError("impulse must last at least one step")

    def covers(self, step: int) -> bool:
        return self.start <= step < self.start + self.steps


@dataclass(frozen=True)
class ActionSchedule:
    """The per-episode action sequence. Shared verbatim across a group's twins."""

    impulses: tuple[Impulse, ...]

    def impulse_vector(self, target: str, step: int, frame: GravityFrame) -> Vec2:
        """Total impulse applied to ``target`` at ``step``, in world coordinates."""
        total_u = 0.0
        total_t = 0.0
        for impulse in self.impulses:
            if impulse.target != target or not impulse.covers(step):
                continue
            unit = direction_vector(impulse.direction, frame)
            total_u += unit[0] * impulse.magnitude
            total_t += unit[1] * impulse.magnitude
        return (total_u, total_t)


def direction_vector(direction: str, frame: GravityFrame) -> Vec2:
    """A frame-relative direction name as a world-space unit vector."""
    up = frame.up
    lateral = frame.lateral
    table: dict[str, Vec2] = {
        "up": up,
        "down": (-up[0], -up[1]),
        "lateral_pos": lateral,
        "lateral_neg": (-lateral[0], -lateral[1]),
        "diagonal_up_pos": _normalize((up[0] + lateral[0], up[1] + lateral[1])),
        "diagonal_up_neg": _normalize((up[0] - lateral[0], up[1] - lateral[1])),
    }
    if direction not in table:
        raise ValueError(f"unknown direction {direction!r}, not in {ALL_DIRECTIONS!r}")
    return table[direction]


def _normalize(vector: Vec2) -> Vec2:
    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (vector[0] / norm, vector[1] / norm)


@dataclass(frozen=True)
class RegimePolicy:
    """A regime is a point in one policy family, not a different policy family.

    Passive, weak and rich differ only in how much excitation the same kind of
    impulse sequence supplies. They never differ in what they know.
    """

    name: str
    n_impulses: int
    magnitude_range: tuple[float, float]
    directions: tuple[str, ...]
    targets: tuple[str, ...]
    duration_range: tuple[int, int]
    min_start: int
    max_start: int


REGIMES: dict[str, RegimePolicy] = {
    PASSIVE: RegimePolicy(
        name=PASSIVE,
        n_impulses=0,
        magnitude_range=(0.0, 0.0),
        directions=(),
        targets=(),
        duration_range=(1, 1),
        min_start=0,
        max_start=0,
    ),
    WEAK: RegimePolicy(
        name=WEAK,
        n_impulses=2,
        magnitude_range=(0.3, 0.7),
        directions=LATERAL_DIRECTIONS,
        targets=("i",),
        duration_range=(1, 2),
        min_start=2,
        max_start=40,
    ),
    RICH: RegimePolicy(
        name=RICH,
        n_impulses=8,
        magnitude_range=(0.8, 2.5),
        directions=ALL_DIRECTIONS,
        targets=("i", "j"),
        duration_range=(1, 3),
        min_start=1,
        max_start=45,
    ),
}


def sample_schedule(policy: RegimePolicy, horizon: int, rng: random.Random) -> ActionSchedule:
    """Draw an action schedule. Consumes no mechanism, filler or tuple information."""
    if policy.n_impulses == 0:
        return ActionSchedule(impulses=())
    if policy.max_start < policy.min_start:
        raise ValueError(f"regime {policy.name!r} has an empty start window")
    impulses: list[Impulse] = []
    for _ in range(policy.n_impulses):
        start = rng.randint(policy.min_start, min(policy.max_start, horizon - 1))
        impulses.append(
            Impulse(
                target=rng.choice(policy.targets),
                direction=rng.choice(policy.directions),
                magnitude=rng.uniform(*policy.magnitude_range),
                start=start,
                steps=rng.randint(*policy.duration_range),
            )
        )
    return ActionSchedule(impulses=tuple(sorted(impulses, key=lambda imp: (imp.start, imp.target, imp.direction))))


def total_impulse_magnitude(schedule: ActionSchedule) -> float:
    """A scalar used to check that the regimes are ordered as intended."""
    return sum(impulse.magnitude * impulse.steps for impulse in schedule.impulses)
