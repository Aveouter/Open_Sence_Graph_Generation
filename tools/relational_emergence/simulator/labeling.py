"""Derive ``S_0`` from a configuration using the pure geometric predicates.

``S_0`` is never *attached* to a sample; it is computed from the configuration by
three predicates that know nothing about the mechanism. That is what makes the
``X0-only -> M`` leakage probe meaningful rather than circular: if the label were
carried alongside the state, the probe would be measuring the label.
"""

from __future__ import annotations

from .fillers import StructuralAttributes
from .factors import S0
from .geometry import (
    GravityFrame,
    DEFAULT_TOL,
    containment_is_inside,
    contact_is_touching,
    frame_coords,
    half_extents_in_frame,
    pose_is_on_top,
)
from .state import WorldState


def classify_s0(
    state: WorldState,
    structural: StructuralAttributes,
    frame: GravityFrame,
    tol: float = DEFAULT_TOL,
) -> S0:
    shape = structural.shape()
    du, dt = frame_coords(state.i.pos, state.j.pos, frame)
    i_half_u, i_half_t = half_extents_in_frame(state.i.half_extent, frame)
    return S0(
        containment=(
            "inside"
            if containment_is_inside(du, i_half_u, dt, i_half_t, shape, tol=tol)
            else "outside"
        ),
        contact=(
            "touching"
            if contact_is_touching(du, i_half_u, dt, i_half_t, shape, tol=tol)
            else "non_touching"
        ),
        pose=(
            "on_top"
            if pose_is_on_top(
                state.i.pos,
                state.i.half_extent,
                state.j.pos,
                state.j.half_extent,
                frame,
                tol=tol,
            )
            else "other"
        ),
    )
