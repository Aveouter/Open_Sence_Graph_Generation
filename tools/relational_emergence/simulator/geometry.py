"""Pure geometric predicates for the Phase I simulator.

Every function here is a function of geometry and the gravity frame alone.
:func:`pose_is_on_top` implements the ``pose`` factor and must never read the
interaction mechanism ``M``, contact state, contact force, or mechanism-active
state (ADR 0005). That restriction is enforced structurally -- no such argument
exists -- and by a signature test in
``tests/analysis/test_relational_emergence_ontology.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec2 = tuple[float, float]

DEFAULT_TOL = 1e-9


@dataclass(frozen=True)
class MaterialShape:
    """The solid shape of the supporter ``j``, in its own gravity-frame coordinates.

    ``j`` is an outer box of half-extents ``(half_u, half_t)`` with an open-topped
    notch removed: the notch spans ``|t| <= cavity_half_t`` from the top down to
    ``u = -floor_offset``. The remaining material is therefore a floor slab plus
    two side walls.

    Only 90-degree gravity frames are used (see :data:`GRAVITY_ANGLES_DEG`), so a
    world-axis-aligned box stays axis-aligned in the frame and every overlap
    computation below is exact rather than approximate.
    """

    half_u: float
    half_t: float
    cavity_half_t: float
    floor_offset: float

    def material_rects(self) -> tuple[tuple[float, float, float, float], ...]:
        """The three solid rectangles as ``(u_lo, u_hi, t_lo, t_hi)``."""
        return (
            (-self.half_u, -self.floor_offset, -self.half_t, self.half_t),
            (-self.floor_offset, self.half_u, -self.half_t, -self.cavity_half_t),
            (-self.floor_offset, self.half_u, self.cavity_half_t, self.half_t),
        )


@dataclass(frozen=True)
class GravityFrame:
    """A 2D gravity frame.

    ``angle_rad`` rotates the frame about the world origin: ``0`` means gravity
    points along ``-y`` (world-down), and ``pi/2`` means it points along ``+x``.

    The frame is sampled once per episode and is part of the world, not of the
    action policy, so it is identical across the intervention regimes and across
    the mechanisms of one counterfactual group (ADR 0005).
    """

    angle_rad: float

    @property
    def down(self) -> Vec2:
        return (math.sin(self.angle_rad), -math.cos(self.angle_rad))

    @property
    def up(self) -> Vec2:
        down = self.down
        return (-down[0], -down[1])

    @property
    def lateral(self) -> Vec2:
        up = self.up
        return (-up[1], up[0])


def dot(point: Vec2, direction: Vec2) -> float:
    return point[0] * direction[0] + point[1] * direction[1]


def support_extent(half_extent: Vec2, direction: Vec2) -> float:
    """Half-width of an axis-aligned box projected onto unit ``direction``.

    Exact for axis-aligned boxes: the projection width is the sum of each axis
    extent scaled by its component along ``direction``.
    """
    return abs(direction[0]) * half_extent[0] + abs(direction[1]) * half_extent[1]


def pose_is_on_top(
    i_pos: Vec2,
    i_half_extent: Vec2,
    j_pos: Vec2,
    j_half_extent: Vec2,
    frame: GravityFrame,
    tol: float = DEFAULT_TOL,
) -> bool:
    """``pose == "on_top"``: ``i`` is on the gravity side of ``j``'s up-facing
    face, and their footprints overlap laterally.

    This is deliberately **not** a contact test: a hovering ``i`` is still
    ``on_top``. That is exactly why ``contact`` is a separate factor in ``S_0``.
    """
    up = frame.up
    lateral = frame.lateral
    j_face = dot(j_pos, up) + support_extent(j_half_extent, up)
    i_underside = dot(i_pos, up) - support_extent(i_half_extent, up)
    if i_underside < j_face - tol:
        return False
    reach = support_extent(j_half_extent, lateral) + support_extent(i_half_extent, lateral)
    return abs(dot(i_pos, lateral) - dot(j_pos, lateral)) <= reach + tol


def frame_coords(point: Vec2, origin: Vec2, frame: GravityFrame) -> tuple[float, float]:
    """``point`` relative to ``origin``, expressed as ``(along up, along lateral)``."""
    delta = (point[0] - origin[0], point[1] - origin[1])
    return dot(delta, frame.up), dot(delta, frame.lateral)


def half_extents_in_frame(half_extent: Vec2, frame: GravityFrame) -> tuple[float, float]:
    """A world-axis-aligned box's half-extents along ``up`` and ``lateral``."""
    return support_extent(half_extent, frame.up), support_extent(half_extent, frame.lateral)


def world_half_extents(half_u: float, half_t: float, frame: GravityFrame) -> Vec2:
    """The world ``(x, y)`` half-extents of a box whose *frame* extents are given.

    The inverse of :func:`half_extents_in_frame` for the 90-degree frames this
    simulator uses, where the frame's basis vectors are axis-aligned so the map
    between world and frame extents is an axis permutation. Structural attributes
    are stated in frame terms -- ``half_u`` is "how tall" and ``half_t`` is "how
    wide" regardless of which way gravity points -- so this conversion has to
    happen exactly once, where a body is built. Building the body from the raw
    frame numbers instead leaves every box rotated relative to its own material
    shape, which shows up as a small offset rather than as an obvious error.
    """
    up = frame.up
    lateral = frame.lateral
    return (
        abs(up[0]) * half_u + abs(lateral[0]) * half_t,
        abs(up[1]) * half_u + abs(lateral[1]) * half_t,
    )


def rect_gap(
    du: float, i_half_u: float, dt: float, i_half_t: float, rect: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Per-axis separation between a box centred at ``(du, dt)`` and ``rect``.

    Each component is zero when the box and the rectangle overlap on that axis,
    so ``hypot(*rect_gap(...))`` is the true separation distance.
    """
    u_lo, u_hi, t_lo, t_hi = rect
    gap_u = max(0.0, u_lo - (du + i_half_u), (du - i_half_u) - u_hi)
    gap_t = max(0.0, t_lo - (dt + i_half_t), (dt - i_half_t) - t_hi)
    return gap_u, gap_t


def rect_overlap(
    du: float, i_half_u: float, dt: float, i_half_t: float, rect: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Per-axis overlap depth with ``rect``; a non-positive component means no overlap."""
    u_lo, u_hi, t_lo, t_hi = rect
    over_u = min(du + i_half_u, u_hi) - max(du - i_half_u, u_lo)
    over_t = min(dt + i_half_t, t_hi) - max(dt - i_half_t, t_lo)
    return over_u, over_t


def minimum_translation(
    du: float, i_half_u: float, dt: float, i_half_t: float, rect: tuple[float, float, float, float]
) -> tuple[str, float] | None:
    """The cheapest axis and depth that separates the box from ``rect``.

    Returns ``None`` when they do not overlap. The axis is ``"u"`` or ``"t"``,
    and the depth is signed so that adding it to that coordinate separates the
    two. This is the contact normal a surface law pushes along; a purely vertical
    law would instead eject objects sideways through walls.
    """
    over_u, over_t = rect_overlap(du, i_half_u, dt, i_half_t, rect)
    if over_u <= 0.0 or over_t <= 0.0:
        return None
    u_lo, u_hi, _, _ = rect
    if over_u <= over_t:
        return "u", (-over_u if du < (u_lo + u_hi) / 2.0 else over_u)
    t_lo, t_hi, _, _ = rect
    return "t", (-over_t if dt < (t_lo + t_hi) / 2.0 else over_t)


def contact_is_touching(
    du: float,
    i_half_u: float,
    dt: float,
    i_half_t: float,
    shape: MaterialShape,
    tol: float = DEFAULT_TOL,
) -> bool:
    """``contact == "touching"``: the box is within ``tol`` of ``j``'s material.

    Pure geometry: the gap is the minimum distance to any of the material
    rectangles, so an object floating inside the cavity is not touching even
    though it sits inside ``j``'s outer box.
    """
    gap = math.inf
    for rect in shape.material_rects():
        gap_u, gap_t = rect_gap(du, i_half_u, dt, i_half_t, rect)
        gap = min(gap, math.hypot(gap_u, gap_t))
    return gap <= tol


def containment_is_inside(
    du: float,
    i_half_u: float,
    dt: float,
    i_half_t: float,
    shape: MaterialShape,
    tol: float = DEFAULT_TOL,
) -> bool:
    """``containment == "inside"``: the box lies within the notch.

    Inside means laterally within the cavity span, above the notch floor, and
    below the rim. The rim condition is what keeps an object resting on top from
    also counting as inside, which is exactly the joint exclusion the factor
    definitions encode.
    """
    laterally_in = abs(dt) + i_half_t <= shape.cavity_half_t + tol
    above_floor = (du - i_half_u) >= -shape.floor_offset - tol
    below_rim = (du + i_half_u) <= shape.half_u + tol
    return laterally_in and above_floor and below_rim
