"""Level-1 identifiability: the architecture-free gate.

The measurement is the shared-noise matched-counterfactual trajectory distance

    J_ab(h) = E_c[ d_norm(X_h^{M_a}, X_h^{M_b}) ]

averaged over base contexts ``c``, with the horizon mean ``J_ab^AUC`` as the
primary scalar and the full curve kept alongside it. Because twins share
``X_0``, fillers, actions and noise, the two futures are deterministic single
trajectories rather than distributions, so the distance is exact and needs no
estimator.

Two floors are reported separately, and the separation matters (ADR 0007):

``floor_num``
    the same mechanism under the same seed, re-run through a numerically
    different computation path. This is the floating-point floor and should be
    at machine precision. If it is not, the integration is doing something
    order-dependent that should be understood before any J is believed.

``floor_null``
    the same mechanism in the same context with the initial velocity
    perturbation re-drawn. This is the statistical floor, and it doubles as the
    precondition test: if distances under *one* mechanism are comparable to
    distances *between* mechanisms, the dynamics are noise-dominated, the
    deterministic/shared-noise premise fails, and the honest verdict is
    ``STOP_DATA`` rather than a J divided by a floor that swallowed it.

``d_norm`` is computed on the physical outcome state only -- positions and
velocities -- never on the mechanism, an activation flag or simulator metadata,
and each dimension is divided by a frozen robust scale estimated on the
development world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..simulator.counterfactuals import CounterfactualGroup, build_group
from ..simulator.state import DEFAULT_CONSTANTS, WorldConstants, WorldState
from ..simulator.world import rollout

MAD_TO_SIGMA = 1.4826
RELATIVE_SPREAD_FLOOR = 1e-2
ABSOLUTE_SCALE_FLOOR = 1e-9


def robust_scale(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Per-dimension ``1.4826 * MAD``, floored at a fraction of that dimension's spread.

    A bare MAD is degenerate here and the degeneracy is not subtle. Velocities
    sit at zero whenever both bodies are resting, which is most of every horizon,
    so a velocity dimension's MAD is floating-point dust -- around ``1e-6`` --
    while the same dimension's range is tens of units during the excited phase.
    Dividing by that dust makes every distance the velocity difference times
    ``1e7`` and buries the mechanism signal entirely.

    The floor is therefore relative to each dimension's own observed range rather
    than absolute: a dimension may never be weighted more than a hundred times
    more heavily than its own full swing. That keeps the estimate robust to
    outliers, which is what MAD is for, without letting a mostly-resting
    dimension become infinitely sensitive.
    """
    if not vectors:
        raise ValueError("robust scale needs at least one outcome vector")
    width = len(vectors[0])
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("outcome vectors must share a width")
    scales: list[float] = []
    for index in range(width):
        column = sorted(vector[index] for vector in vectors)
        median = _median_sorted(column)
        deviations = sorted(abs(value - median) for value in column)
        mad = _median_sorted(deviations)
        spread = column[-1] - column[0]
        scales.append(
            max(
                MAD_TO_SIGMA * mad,
                RELATIVE_SPREAD_FLOOR * spread,
                ABSOLUTE_SCALE_FLOOR,
            )
        )
    return tuple(scales)


def _median_sorted(values: list[float]) -> float:
    count = len(values)
    if count == 0:
        raise ValueError("median of an empty sequence")
    middle = count // 2
    if count % 2 == 1:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def normalized_distance(a: WorldState, b: WorldState, scale: tuple[float, ...]) -> float:
    vector_a = a.outcome_vector()
    vector_b = b.outcome_vector()
    if len(vector_a) != len(scale) or len(vector_b) != len(scale):
        raise ValueError("outcome vector width does not match the frozen scale")
    return math.sqrt(
        sum(
            ((x - y) / s) ** 2
            for x, y, s in zip(vector_a, vector_b, scale, strict=True)
        )
    )


@dataclass(frozen=True)
class PairCurves:
    """Distances for one mechanism pair, one regime, one tuple."""

    tuple_key: str
    regime: str
    pair: str
    per_group: tuple[tuple[float, ...], ...]

    def mean_curve(self) -> tuple[float, ...]:
        if not self.per_group:
            return ()
        horizon = len(self.per_group[0])
        return tuple(
            sum(curve[h] for curve in self.per_group) / len(self.per_group)
            for h in range(horizon)
        )

    def auc(self) -> float:
        curve = self.mean_curve()
        if not curve:
            return 0.0
        return sum(curve) / len(curve)

    def per_group_auc(self) -> tuple[float, ...]:
        return tuple(sum(curve) / len(curve) for curve in self.per_group)


def _trajectories(
    group: CounterfactualGroup,
    regime: str,
    horizon: int,
    constants: WorldConstants,
    numerical_variant: int = 0,
) -> dict[str, tuple[WorldState, ...]]:
    schedule = group.schedules[regime]
    return {
        mechanism: rollout(
            group.x0,
            schedule,
            group.filler.structural,
            mechanism,
            group.frame,
            group.link_offset(mechanism),
            horizon,
            constants,
            numerical_variant=numerical_variant,
        )
        for mechanism in group.mechanisms
    }


def _pair_key(a: str, b: str) -> str:
    return "/".join(sorted((a, b)))


def collect_calibration_vectors(
    groups: list[CounterfactualGroup], horizon: int, constants: WorldConstants
) -> list[tuple[float, ...]]:
    """Outcome vectors used to freeze the scale, drawn across every regime and law."""
    vectors: list[tuple[float, ...]] = []
    for group in groups:
        for regime in sorted(group.schedules):
            for trajectory in _trajectories(group, regime, horizon, constants).values():
                vectors.extend(state.outcome_vector() for state in trajectory)
    return vectors


def pair_curves(
    groups: list[CounterfactualGroup],
    scale: tuple[float, ...],
    horizon: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
) -> dict[str, list[PairCurves]]:
    """``J_ab(h)`` per tuple, regime and pair, with one curve per base context."""
    collected: dict[str, dict[str, dict[str, list[tuple[float, ...]]]]] = {}
    for group in groups:
        for regime in sorted(group.schedules):
            trajectories = _trajectories(group, regime, horizon, constants)
            mechanisms = sorted(trajectories)
            for index, first in enumerate(mechanisms):
                for second in mechanisms[index + 1 :]:
                    key = _pair_key(first, second)
                    curve = tuple(
                        normalized_distance(trajectories[first][h], trajectories[second][h], scale)
                        for h in range(horizon)
                    )
                    (
                        collected.setdefault(group.s0.key(), {})
                        .setdefault(regime, {})
                        .setdefault(key, [])
                        .append(curve)
                    )
    return {
        tuple_key: [
            PairCurves(
                tuple_key=tuple_key,
                regime=regime,
                pair=pair,
                per_group=tuple(per_group),
            )
            for regime, pairs in sorted(regimes.items())
            for pair, per_group in sorted(pairs.items())
        ]
        for tuple_key, regimes in sorted(collected.items())
    }


def numerical_floor(
    groups: list[CounterfactualGroup],
    scale: tuple[float, ...],
    horizon: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
) -> dict:
    """Same-seed distances when only the arithmetic grouping changes.

    Reported as a distribution rather than a single worst case, so it can be
    read against the null floor's distribution on equal terms. The maximum is
    kept because that is what a threshold would have to clear, but a max compared
    against a mean would flatter the check.
    """
    worst = 0.0
    total = 0.0
    count = 0
    largest: list[float] = []
    for group in groups:
        for regime in sorted(group.schedules):
            schedule = group.schedules[regime]
            for mechanism in group.mechanisms:
                straight = rollout(
                    group.x0,
                    schedule,
                    group.filler.structural,
                    mechanism,
                    group.frame,
                    group.link_offset(mechanism),
                    horizon,
                    constants,
                )
                variant = rollout(
                    group.x0,
                    schedule,
                    group.filler.structural,
                    mechanism,
                    group.frame,
                    group.link_offset(mechanism),
                    horizon,
                    constants,
                    numerical_variant=1,
                )
                for state_a, state_b in zip(straight, variant, strict=True):
                    distance = normalized_distance(state_a, state_b, scale)
                    worst = max(worst, distance)
                    total += distance
                    count += 1
                    largest.append(distance)
    largest.sort()
    return {
        "max": worst,
        "mean": total / count if count else 0.0,
        "quantile_95": largest[min(len(largest) - 1, int(0.95 * len(largest)))] if largest else 0.0,
        "n_samples": count,
    }


def null_floor_curves(
    groups: list[CounterfactualGroup],
    scale: tuple[float, ...],
    horizon: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
) -> dict[str, list[PairCurves]]:
    """Same-mechanism, re-drawn-noise distances; the null floor and its precondition role."""
    collected: dict[str, dict[str, dict[str, list[tuple[float, ...]]]]] = {}
    for group in groups:
        twin = build_group(
            group.s0,
            group.spec.group_index,
            group.filler,
            horizon,
            constants,
            noise_offset=0x5EED,
        )
        for regime in sorted(group.schedules):
            primary = _trajectories(group, regime, horizon, constants)
            secondary = _trajectories(twin, regime, horizon, constants)
            for mechanism in sorted(primary):
                curve = tuple(
                    normalized_distance(primary[mechanism][h], secondary[mechanism][h], scale)
                    for h in range(horizon)
                )
                (
                    collected.setdefault(group.s0.key(), {})
                    .setdefault(regime, {})
                    .setdefault(mechanism, [])
                    .append(curve)
                )
    return {
        tuple_key: [
            PairCurves(
                tuple_key=tuple_key,
                regime=regime,
                pair=mechanism,
                per_group=tuple(per_group),
            )
            for regime, mechanisms in sorted(regimes.items())
            for mechanism, per_group in sorted(mechanisms.items())
        ]
        for tuple_key, regimes in sorted(collected.items())
    }
