"""Bodies, world state and world constants.

Split out from :mod:`world` so that :mod:`mechanisms` can name the state type
without importing the module that applies the laws.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Vec2


@dataclass(frozen=True)
class WorldConstants:
    gravity_accel: float = 0.5
    damping: float = 0.99
    ground_u: float = 0.0
    dt: float = 1.0
    # Lateral walls, in frame coordinates. Without them the world is unbounded
    # sideways and a single impulse of magnitude v drifts a body roughly
    # v / (1 - damping) -- about 150 units at the default damping. Every
    # distance would then be dominated by which way a body happened to wander
    # rather than by the law under test, and the same-mechanism null floor would
    # come out comparable to the between-mechanism signal.
    arena_half_t: float = 4.0
    # And a ceiling, for the same reason in the other direction. Gravity bounds
    # an upward impulse only at v^2 / 2g, which at the rich regime's magnitudes is
    # far above the geometry the mechanisms act on, so without a ceiling an
    # excited body spends the horizon throwing the measurement off.
    arena_top_u: float = 6.0
    # Contacts are compliant over this distance rather than switch-like. A hard
    # constraint zeroes a whole velocity the instant a threshold is crossed, and
    # a step function of position means a 1e-17 difference in the state flips a
    # branch and diverges macroscopically -- which makes the *numerical* floor
    # larger than the signal it is supposed to bound. Ramping the response in
    # over `contact_softness` keeps the same physics in the limit while making
    # the map continuous, so small causes have small effects and the floors mean
    # what they claim to.
    contact_softness: float = 0.05


DEFAULT_CONSTANTS = WorldConstants()


@dataclass(frozen=True)
class Body:
    pos: Vec2
    vel: Vec2
    half_extent: Vec2
    mass: float


@dataclass(frozen=True)
class WorldState:
    """The physical outcome state. Nothing here is metadata.

    Position, velocity and geometry only. The mechanism, the activation state and
    the filler identity are deliberately absent, so a distance computed over this
    object cannot accidentally measure something other than the physical outcome.
    """

    i: Body
    j: Body

    def outcome_vector(self) -> tuple[float, ...]:
        return (
            self.i.pos[0],
            self.i.pos[1],
            self.i.vel[0],
            self.i.vel[1],
            self.j.pos[0],
            self.j.pos[1],
            self.j.vel[0],
            self.j.vel[1],
        )
