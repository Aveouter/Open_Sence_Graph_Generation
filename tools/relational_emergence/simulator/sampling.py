"""Initial-state sampling, conditioned on the target ``S_0`` tuple.

``X_0`` is drawn from ``P(X_0 | S_0)`` and is independent of the mechanism and of
the intervention regime. That independence is what lets a counterfactual group
reuse one ``X_0`` across every compatible law, and it is why the mechanism is
defined as a dispositional law: it constrains the response to excitation without
changing what ``X_0`` physically is.

The sampler does not trust itself. Every sample is classified by the same pure
predicates a probe would use, and the caller is expected to check the round trip;
:func:`sample_for_tuple` raises when it fails, which is what turns a mis-placed
object into a loud failure instead of a mislabelled row.
"""

from __future__ import annotations

import random

from .fillers import StructuralAttributes
from .factors import S0, enumerate_realizable
from .geometry import GravityFrame, Vec2
from .labeling import classify_s0
from .state import DEFAULT_CONSTANTS, Body, WorldConstants, WorldState

GAP = 0.15

# Zero-centred initial velocity, and deliberately tiny. Over a 40-step horizon a
# velocity of v accumulates about 34 * v of displacement once damping is counted,
# so v = 0.02 drifts roughly 0.7 units -- further than the cavity's lateral slack,
# which would make `containment` and `support` separate under passive observation
# simply because one has walls and the other does not. That would destroy the
# only cell where the three main laws are supposed to be passively
# indistinguishable and separable only under excitation. At 0.002 the drift is
# about 0.07, comfortably inside the slack and comfortably below the non-touching
# gap, so it perturbs the ensemble without deciding any comparison.
VELOCITY_SCALE = 0.002


def tuple_offsets(s0: S0, structural: StructuralAttributes) -> tuple[float, float]:
    """``(du, dt)`` for ``i`` relative to ``j``, in frame coordinates.

    Each branch places ``i`` exactly at the boundary the corresponding predicate
    tests: resting on the notch floor for ``inside``, on a wall's top face for
    ``on_top``, touching a side wall for the lateral case. ``non_touching`` then
    lifts the object by :data:`GAP`, which is comfortably above the predicate
    tolerance so the two cases cannot be confused by rounding.
    """
    shape = structural.shape()
    if s0.containment == "inside":
        du = -shape.floor_offset + structural.i_half_u
        dt = 0.0
    elif s0.pose == "on_top":
        du = shape.half_u + structural.i_half_u
        dt = (shape.cavity_half_t + shape.half_t) / 2.0
    else:
        du = 0.0
        dt = shape.half_t + structural.i_half_t
    if s0.contact == "non_touching":
        if s0.containment == "inside" or s0.pose == "on_top":
            du += GAP
        else:
            dt += GAP
    return du, dt


def _world_offset(du: float, dt: float, frame: GravityFrame) -> Vec2:
    up = frame.up
    lateral = frame.lateral
    return (up[0] * du + lateral[0] * dt, up[1] * du + lateral[1] * dt)


def rest_position_of_supporter(
    structural: StructuralAttributes, frame: GravityFrame, constants: WorldConstants
) -> Vec2:
    """Where ``j`` sits once the ground holds it."""
    height = constants.ground_u + structural.j_half_u
    up = frame.up
    return (up[0] * height, up[1] * height)


def sample_for_tuple(
    s0: S0,
    structural: StructuralAttributes,
    frame: GravityFrame,
    rng: random.Random,
    constants: WorldConstants = DEFAULT_CONSTANTS,
    strict: bool = True,
) -> WorldState:
    """Draw ``X_0`` for ``s0`` and verify it classifies back to ``s0``."""
    j_pos = rest_position_of_supporter(structural, frame, constants)
    du, dt = tuple_offsets(s0, structural)
    offset = _world_offset(du, dt, frame)
    state = WorldState(
        i=Body(
            pos=(j_pos[0] + offset[0], j_pos[1] + offset[1]),
            vel=(
                rng.uniform(-VELOCITY_SCALE, VELOCITY_SCALE),
                rng.uniform(-VELOCITY_SCALE, VELOCITY_SCALE),
            ),
            half_extent=structural.i_world_half_extent(frame),
            mass=structural.mass_i,
        ),
        j=Body(
            pos=j_pos,
            vel=(
                rng.uniform(-VELOCITY_SCALE, VELOCITY_SCALE),
                rng.uniform(-VELOCITY_SCALE, VELOCITY_SCALE),
            ),
            half_extent=structural.j_world_half_extent(frame),
            mass=structural.mass_j,
        ),
    )
    if strict:
        observed = classify_s0(state, structural, frame, tol=constants_rounding_tol())
        if observed != s0:
            raise AssertionError(
                f"sampled X_0 classifies as {observed.key()}, expected {s0.key()}; "
                "the sampler and the labelling predicates disagree"
            )
    return state


def constants_rounding_tol() -> float:
    """The predicate tolerance used for the round-trip check.

    Tight, because the sampler places objects *exactly* on a predicate boundary
    and a loose tolerance would let a placement that is off by a hair still pass.
    """
    return 1e-9


ALL_TUPLES: tuple[S0, ...] = enumerate_realizable()
