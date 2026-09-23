"""The integration step and the rollout.

The world is deliberately austere: two axis-aligned boxes, a gravity frame
sampled once per episode, a ground plane and arena walls every body stays
inside, and the pair law chosen for the episode. Rotation is left out so that a
box stays axis-aligned in a 90-degree gravity frame and every overlap
computation stays exact rather than approximate.

Gravity acts on both bodies. The supporter is roughly fifteen times heavier than
the probe object, so an impulse moves it visibly but not wildly -- which matters,
because "nudge the supporter" is one of the excitations that separates an
attached pair from a supported one.

The step order is: gravity and impulses, then every bound and the pair law at
the velocity level, then integration, then the same constraints again at the
position level. Constraining before integrating is what lets a body descend onto
a surface and stay there instead of sinking and being pushed back each step.

Every response is continuous in the state. The constraints are compliant over
:attr:`WorldConstants.contact_softness` rather than switch-like, because a hard
constraint makes the dynamics a step function of position: a ``1e-17``
difference flips a contact branch and the two runs diverge macroscopically. That
is not a hypothetical -- it is what the numerical floor measured before the
ramp was added, and it would have made the identifiability floor meaningless.
"""

from __future__ import annotations

from .actions import ActionSchedule
from .fillers import StructuralAttributes
from .geometry import GravityFrame, Vec2, half_extents_in_frame
from .mechanisms import apply_law, project_positions, ramp
from .state import DEFAULT_CONSTANTS, Body, WorldConstants, WorldState

__all__ = ["Body", "WorldConstants", "WorldState", "DEFAULT_CONSTANTS", "step", "rollout"]


def _walls(
    body: Body, frame: GravityFrame, constants: WorldConstants
) -> tuple[tuple[Vec2, float, float, float], ...]:
    """The four one-sided limits as ``(basis, coordinate, limit, outward_sign)``.

    Each entry reads "this body may not move past ``limit`` along ``basis`` in
    the direction of ``outward_sign``", so the ground, the ceiling and the two
    arena walls are one kind of thing and get the same continuous treatment.
    """
    up = frame.up
    side = frame.lateral
    half_u, half_t = half_extents_in_frame(body.half_extent, frame)
    height = body.pos[0] * up[0] + body.pos[1] * up[1]
    offset = body.pos[0] * side[0] + body.pos[1] * side[1]
    return (
        (up, height, constants.ground_u + half_u, -1.0),
        (up, height, constants.arena_top_u - half_u, 1.0),
        (side, offset, constants.arena_half_t - half_t, 1.0),
        (side, offset, -constants.arena_half_t + half_t, -1.0),
    )


def _resolve_bounds(body: Body, frame: GravityFrame, constants: WorldConstants) -> Body:
    """Project a body back inside the arena, then damp its outward velocity.

    The position correction is *complete*, not partial: the module claims bodies
    stay inside, and a partial relaxation makes that claim false. A full clamp is
    a projection onto a half-space, which is continuous and Lipschitz-1 in the
    state, so it costs nothing in the continuity the ramps exist to provide -- a
    measured 26.9% of rich-regime body-steps sat outside a wall while the
    correction was partial, because sustained impulses outrun a 0.6 gain.

    The velocity response is still ramped, since *that* is the term that would
    otherwise be a switch: zeroing an outward velocity the instant a threshold is
    crossed is a jump, and a jump is what turns arithmetic noise into
    macroscopic divergence.
    """
    softness = constants.contact_softness
    pos_x, pos_y = body.pos
    vel_x, vel_y = body.vel

    for basis, coordinate, limit, outward_sign in _walls(body, frame, constants):
        violation = (coordinate - limit) * outward_sign
        if violation <= -softness:
            # Clear of the wall by more than the contact range: nothing to do.
            continue
        direction = (basis[0] * outward_sign, basis[1] * outward_sign)

        # Position and velocity need *separate* guards, and conflating them was a
        # real bug. The projection must fire only on a genuine violation, or it
        # would pull a body toward a wall it has not reached. The velocity
        # response must fire on contact, and contact includes touching: a body
        # resting exactly on a surface has `violation == 0`, and skipping it
        # there lets it accumulate a full step of gravity, sink, and be pushed
        # back -- settling into a steady sag rather than resting. Worse, the pair
        # laws damp the *relative* approach, so a supporter that sags makes its
        # supported object sag with it.
        if violation > 0.0:
            pos_x -= direction[0] * violation
            pos_y -= direction[1] * violation

        weight = ramp(violation + softness, softness)
        approach = vel_x * direction[0] + vel_y * direction[1]
        if approach > 0.0:
            vel_x -= direction[0] * approach * weight
            vel_y -= direction[1] * approach * weight

    return Body(
        pos=(pos_x, pos_y),
        vel=(vel_x, vel_y),
        half_extent=body.half_extent,
        mass=body.mass,
    )


def step(
    state: WorldState,
    schedule: ActionSchedule,
    structural: StructuralAttributes,
    mechanism: str,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    step_index: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
    numerical_variant: int = 0,
) -> WorldState:
    """Advance one step under gravity, the episode's impulses, and the pair law.

    ``numerical_variant`` re-groups the same arithmetic (distributing the damping
    factor instead of factoring it out). The result is identical in exact
    arithmetic and differs only in floating-point rounding, which is what makes
    it usable as the numerical floor: a distance between variant 0 and variant 1
    is arithmetic noise and nothing else.
    """
    down = frame.down
    impulse_i = schedule.impulse_vector("i", step_index, frame)
    impulse_j = schedule.impulse_vector("j", step_index, frame)

    def pushed(body: Body, impulse: Vec2) -> Body:
        accel = (
            down[0] * constants.gravity_accel + impulse[0] / body.mass,
            down[1] * constants.gravity_accel + impulse[1] / body.mass,
        )
        if numerical_variant == 0:
            vel = (
                (body.vel[0] + accel[0]) * constants.damping,
                (body.vel[1] + accel[1]) * constants.damping,
            )
        else:
            vel = (
                body.vel[0] * constants.damping + accel[0] * constants.damping,
                body.vel[1] * constants.damping + accel[1] * constants.damping,
            )
        return Body(pos=body.pos, vel=vel, half_extent=body.half_extent, mass=body.mass)

    powered = WorldState(i=pushed(state.i, impulse_i), j=pushed(state.j, impulse_j))
    powered = WorldState(
        i=_resolve_bounds(powered.i, frame, constants),
        j=_resolve_bounds(powered.j, frame, constants),
    )
    powered = apply_law(
        mechanism, powered, structural, frame, link_offset, constants.contact_softness
    )

    integrated = WorldState(
        i=Body(
            pos=(
                powered.i.pos[0] + powered.i.vel[0] * constants.dt,
                powered.i.pos[1] + powered.i.vel[1] * constants.dt,
            ),
            vel=powered.i.vel,
            half_extent=powered.i.half_extent,
            mass=powered.i.mass,
        ),
        j=Body(
            pos=(
                powered.j.pos[0] + powered.j.vel[0] * constants.dt,
                powered.j.pos[1] + powered.j.vel[1] * constants.dt,
            ),
            vel=powered.j.vel,
            half_extent=powered.j.half_extent,
            mass=powered.j.mass,
        ),
    )
    bounded = WorldState(
        i=_resolve_bounds(integrated.i, frame, constants),
        j=_resolve_bounds(integrated.j, frame, constants),
    )
    projected = project_positions(mechanism, bounded, structural, frame, link_offset)
    return WorldState(
        i=_resolve_bounds(projected.i, frame, constants),
        j=projected.j,
    )


def rollout(
    initial: WorldState,
    schedule: ActionSchedule,
    structural: StructuralAttributes,
    mechanism: str,
    frame: GravityFrame,
    link_offset: Vec2 | None,
    horizon: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
    numerical_variant: int = 0,
) -> tuple[WorldState, ...]:
    """The trajectory ``X_1 .. X_H``. ``X_0`` is ``initial`` and is not included."""
    states: list[WorldState] = []
    current = initial
    for step_index in range(horizon):
        current = step(
            current,
            schedule,
            structural,
            mechanism,
            frame,
            link_offset,
            step_index,
            constants,
            numerical_variant,
        )
        states.append(current)
    return tuple(states)
