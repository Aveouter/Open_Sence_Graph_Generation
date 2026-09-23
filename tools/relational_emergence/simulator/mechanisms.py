"""The four dispositional interaction laws.

A law states how the pair is constrained *if* contact or excitation occurs. It
may be inactive at ``X_0``, and being inactive is not the same as being
incompatible: ``compatibility.compatible`` says whether the law is well defined
at a tuple at all, and it is the only thing that constrains sampling.

Law semantics, frozen here:

``free``
    No interaction. The two bodies interpenetrate freely.

``support``
    ``j``'s material resists penetration along the **contact normal**. The normal
    is the minimum-translation axis of the deepest overlap, so an object resting
    on the notch floor is held vertically while one leaning on a side wall is
    held laterally. A law that pushed along a fixed axis instead would eject
    objects sideways through walls, and would make ``support`` and
    ``containment`` agree wherever they should differ.

``attachment``
    A rigid link: ``i``'s offset from ``j`` is frozen at its value at ``X_0``,
    and ``i`` shares ``j``'s velocity.

``containment``
    ``i`` is confined to the cavity: floor, ceiling and both side walls. It is
    compatible only with the ``inside`` tuples, where ``X_0`` already lies within
    the region the law projects into -- so satisfying it never requires dragging
    an object through the enclosure's own wall.

Each law is enforced twice per step, which is what keeps resting objects resting:
once at the velocity level *before* integration, so a body descending onto a
surface has its approach velocity removed, and once as a position projection
*after* integration as a safety net. Without the velocity pass an object at rest
sinks by ``g * dt^2`` every step and is pushed back out; that jitter is identical
across twins sharing a seed but different across seeds, so it would inflate the
same-mechanism null floor and corrupt the precondition gate.
"""

from __future__ import annotations

import math

from .compatibility import MECHANISMS
from .fillers import StructuralAttributes
from .geometry import (
    GravityFrame,
    MaterialShape,
    Vec2,
    contact_is_touching,
    frame_coords,
    half_extents_in_frame,
    minimum_translation,
    rect_gap,
)
from .state import Body, WorldState

FREE = "free"
SUPPORT = "support"
ATTACHMENT = "attachment"
CONTAINMENT = "containment"

CONTACT_SLOP = 0.02

# Fraction of a violation corrected per step. Below 1 so the relaxation is a
# contraction rather than a projection: several overlapping rectangles then
# converge instead of fighting each other.
GAIN_POS = 0.6


def ramp(penetration_proximity: float, softness: float) -> float:
    """A continuous contact weight: 1 when touching, 0 at ``softness`` away.

    This is what replaces the switch. ``max(0, ...)`` keeps it continuous at the
    outer end and ``min(1, ...)`` at the inner end, so the response never jumps.
    """
    if softness <= 0.0:
        return 1.0 if penetration_proximity > 0.0 else 0.0
    return min(1.0, max(0.0, penetration_proximity) / softness)


def contact_states(
    du: float,
    i_half_u: float,
    dt: float,
    i_half_t: float,
    shape: MaterialShape,
    softness: float,
) -> tuple[tuple[str, float, float, float], ...]:
    """One ``(axis, sign, weight, depth)`` per material rectangle.

    Every rectangle contributes, and each contribution is continuous in the
    state. That is the point: choosing *the* contact -- by deepest overlap, or by
    nearest gap -- is an argmax, and an argmax is a discrete branch. A residual
    one here was measurable: `support` was the only law whose numerical floor was
    not at machine precision (21.4 against 1e-12 for the other three), because a
    body crossing between two rectangles flipped which one it resolved against.
    Summing over a fixed set of rectangles removes the choice rather than
    smoothing it.

    Overlap and proximity share one formula. The separation is signed -- negative
    with magnitude equal to the minimum-translation depth when the box
    interpenetrates, positive when it is clear -- and the weight is
    ``ramp(softness - separation, softness)``, which is 1 at contact and 0 at
    ``softness`` away. A branch here would be the same switch in another costume.
    """
    contacts: list[tuple[float, float, float, float]] = []
    for rect in shape.material_rects():
        mtv = minimum_translation(du, i_half_u, dt, i_half_t, rect)
        if mtv is not None:
            axis, depth = mtv
            sign = 1.0 if depth > 0.0 else -1.0
            normal_u, normal_t = (sign, 0.0) if axis == "u" else (0.0, sign)
            contacts.append(
                (normal_u, normal_t, ramp(softness + abs(depth), softness), abs(depth))
            )
            continue
        gap_u, gap_t = rect_gap(du, i_half_u, dt, i_half_t, rect)
        distance = math.hypot(gap_u, gap_t)
        if distance >= softness or distance == 0.0:
            continue
        # The gradient of the distance function, not a choice between two axes.
        # Picking "whichever gap is larger" makes the normal flip at a corner,
        # and because the weight is near 1 there, the flip is a full-strength
        # switch -- the last discrete branch `support` had left.
        contacts.append(
            (gap_u / distance, gap_t / distance, ramp(softness - distance, softness), 0.0)
        )
    return tuple(contacts)


def frame_normal(axis: str, sign: float, frame: GravityFrame) -> Vec2:
    """A frame-basis ``(axis, sign)`` normal as a world-space direction."""
    base = frame.up if axis == "u" else frame.lateral
    return (base[0] * sign, base[1] * sign)


def frame_basis(normal_u: float, normal_t: float, frame: GravityFrame) -> Vec2:
    """A normal given on the ``(up, lateral)`` basis, expressed on world axes."""
    up = frame.up
    side = frame.lateral
    return (
        normal_u * up[0] + normal_t * side[0],
        normal_u * up[1] + normal_t * side[1],
    )


def _kill_approach(state: WorldState, normal: Vec2, weight: float) -> WorldState:
    """Scale ``i``'s approaching normal velocity down, so the pair stops closing.

    This treats ``j`` as infinitely massive. Splitting the correction by mass
    instead -- the textbook inelastic impulse -- looks more principled and is
    worse here: under gravity both bodies acquire the same downward velocity, so
    the *relative* approach is zero and the constraint does nothing at all, while
    the mass-split reaction nudges the supporter downward each step and the pair
    sinks together at a few hundredths per step. With a supporter fifteen times
    heavier than the probe, matching the normal velocity is the same physics
    without that failure mode.

    ``weight`` is the contact ramp. Multiplying by it is what keeps the map
    continuous: without it, contact is a switch that zeroes the whole velocity
    the moment two boxes touch, and the resulting branch flips turn arithmetic
    noise into macroscopic divergence.
    """
    if weight <= 0.0:
        return state
    approach = (state.i.vel[0] - state.j.vel[0]) * normal[0] + (
        state.i.vel[1] - state.j.vel[1]
    ) * normal[1]
    if approach >= 0.0:
        return state
    return WorldState(
        i=Body(
            pos=state.i.pos,
            vel=(
                state.i.vel[0] - normal[0] * approach * weight,
                state.i.vel[1] - normal[1] * approach * weight,
            ),
            half_extent=state.i.half_extent,
            mass=state.i.mass,
        ),
        j=state.j,
    )


def _cavity_limits(shape: MaterialShape, i_half_u: float, i_half_t: float) -> tuple[float, float, float]:
    """``(floor, ceiling, lateral limit)`` for a box of these half-extents."""
    floor = -shape.floor_offset + i_half_u
    ceiling = max(floor, shape.half_u - i_half_u)
    lateral = max(0.0, shape.cavity_half_t - i_half_t)
    return floor, ceiling, lateral


def apply_free(
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    softness: float = CONTACT_SLOP,
) -> WorldState:
    return state


def apply_support(
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    softness: float = CONTACT_SLOP,
) -> WorldState:
    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)
    corrected = state
    for normal_u, normal_t, weight, _ in contact_states(du, i_half_u, dt, i_half_t, shape, softness):
        if weight > 0.0:
            corrected = _kill_approach(
                corrected, frame_basis(normal_u, normal_t, frame), weight
            )
    return corrected


def apply_containment(
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    softness: float = CONTACT_SLOP,
) -> WorldState:
    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)
    floor, ceiling, lateral = _cavity_limits(shape, i_half_u, i_half_t)

    corrected = state
    # The sign is the direction the body is *allowed* to move -- `_kill_approach`
    # removes velocity pointing along its negation. So a body below the floor is
    # pushed up (sign +1) and one above the ceiling is pushed down (sign -1).
    # These two were inverted, which made containment unable to hold anything up:
    # the floor response measured the body's velocity as already escaping and
    # did nothing, and the body sank through its own enclosure.
    for overshoot, axis, sign in (
        (dt - lateral, "t", -1.0),
        (-lateral - dt, "t", 1.0),
        (floor - du, "u", 1.0),
        (du - ceiling, "u", -1.0),
    ):
        # The ramp is shifted by `softness` so that resting exactly on the floor
        # or a wall is contact: with `ramp(overshoot)` a body at zero overshoot
        # would receive no response, accumulate a step of gravity, and sag
        # through its own enclosure before the projection pushed it back.
        if overshoot > -softness:
            corrected = _kill_approach(
                corrected, frame_normal(axis, sign, frame), ramp(overshoot + softness, softness)
            )
    return corrected


def apply_attachment(
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    softness: float = CONTACT_SLOP,
) -> WorldState:
    if link_offset is None:
        raise ValueError("attachment requires the link offset recorded at X_0")
    pos = (state.j.pos[0] + link_offset[0], state.j.pos[1] + link_offset[1])
    return WorldState(
        i=Body(pos=pos, vel=state.j.vel, half_extent=state.i.half_extent, mass=state.i.mass),
        j=state.j,
    )


LAWS = {
    FREE: apply_free,
    SUPPORT: apply_support,
    ATTACHMENT: apply_attachment,
    CONTAINMENT: apply_containment,
}

assert set(LAWS) == set(MECHANISMS), "every mechanism must have exactly one law"


def apply_law(
    mechanism: str,
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    softness: float = CONTACT_SLOP,
) -> WorldState:
    """The pre-integration velocity constraint of ``mechanism``'s law."""
    if mechanism not in LAWS:
        raise ValueError(f"unknown mechanism {mechanism!r}, not in {MECHANISMS!r}")
    if mechanism == ATTACHMENT:
        return state
    return LAWS[mechanism](state, structural, frame, link_offset, softness)


def project_positions(
    mechanism: str,
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
) -> WorldState:
    """The post-integration position pass.

    All three constrained laws need one. ``support`` needs it as a backstop: the
    velocity pass stops a body descending onto a surface, but a body that starts
    the step already overlapping -- because a lateral impulse drove it into a
    wall -- would otherwise stay embedded.
    """
    if mechanism == ATTACHMENT:
        return apply_attachment(state, structural, frame, link_offset)
    if mechanism == CONTAINMENT:
        return _project_into_cavity(state, structural, frame)
    if mechanism == SUPPORT:
        return _project_out_of_material(state, structural, frame)
    return state


def _project_out_of_material(
    state: WorldState, structural: StructuralAttributes, frame: GravityFrame
) -> WorldState:
    """Relax ``i`` out of any overlap, by a fraction of the minimum-translation depth.

    Moving by ``GAIN_POS * depth`` rather than the whole depth keeps the
    correction continuous in the state and avoids the overshoot that a full
    projection produces when several rectangles overlap at once.
    """
    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)

    # Every overlapping rectangle contributes its own push, so a body wedged
    # between two of them is resolved along both axes instead of whichever the
    # argmax happened to pick.
    offset_x = 0.0
    offset_y = 0.0
    for normal_u, normal_t, weight, depth in contact_states(
        du, i_half_u, dt, i_half_t, shape, CONTACT_SLOP
    ):
        if depth <= 0.0 or weight <= 0.0:
            continue
        normal = frame_basis(normal_u, normal_t, frame)
        offset_x += normal[0] * GAIN_POS * depth
        offset_y += normal[1] * GAIN_POS * depth
    if offset_x == 0.0 and offset_y == 0.0:
        return state
    return WorldState(
        i=Body(
            pos=(state.i.pos[0] + offset_x, state.i.pos[1] + offset_y),
            vel=state.i.vel,
            half_extent=state.i.half_extent,
            mass=state.i.mass,
        ),
        j=state.j,
    )


def _project_into_cavity(
    state: WorldState, structural: StructuralAttributes, frame: GravityFrame
) -> WorldState:
    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)
    floor, ceiling, lateral = _cavity_limits(shape, i_half_u, i_half_t)

    new_du = du + GAIN_POS * (min(max(du, floor), ceiling) - du)
    new_dt = dt + GAIN_POS * (min(max(dt, -lateral), lateral) - dt)
    if (new_du, new_dt) == (du, dt):
        return state

    up = frame.up
    side = frame.lateral
    delta = (
        up[0] * (new_du - du) + side[0] * (new_dt - dt),
        up[1] * (new_du - du) + side[1] * (new_dt - dt),
    )
    pos = (state.i.pos[0] + delta[0], state.i.pos[1] + delta[1])
    # Position only. Zeroing the velocity component here as well -- which is what
    # this used to do -- makes the map discontinuous: the moment the position
    # crosses a limit by any amount, the whole velocity component is cleared, so
    # a 1e-17 difference flips a switch. Containment was the only law whose
    # *numerical* floor was not at machine precision (2.3e-02 against 1.7e-14 for
    # the other three), and this was why. The ramped velocity response in
    # `apply_containment` already handles velocity; this pass is a safety net for
    # position and should touch nothing else.
    return WorldState(
        i=Body(pos=pos, vel=state.i.vel, half_extent=state.i.half_extent, mass=state.i.mass),
        j=state.j,
    )


def mechanism_is_manifesting(
    mechanism: str,
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    link_offset: Vec2 | None,
) -> bool:
    """The ``mechanism_active`` predicate, kept distinct from ``compatible``.

    A compatible law may be dormant; that is the ordinary case for a dispositional
    law, and conflating the two would make the compatibility table mean something
    it does not.
    """
    if mechanism == FREE:
        return False
    if mechanism == ATTACHMENT:
        return True

    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)
    if mechanism == SUPPORT:
        return contact_is_touching(du, i_half_u, dt, i_half_t, shape, tol=CONTACT_SLOP)

    floor, ceiling, lateral = _cavity_limits(shape, i_half_u, i_half_t)
    on_wall = abs(dt) >= lateral - CONTACT_SLOP
    on_floor = du <= floor + CONTACT_SLOP
    on_ceiling = du >= ceiling - CONTACT_SLOP
    return on_wall or on_floor or on_ceiling
