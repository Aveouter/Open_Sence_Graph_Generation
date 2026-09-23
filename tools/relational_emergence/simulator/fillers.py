"""Filler properties, split into the structural and nuisance halves of ADR 0005.

Structural attributes are allowed to bear on relation compatibility: an object's
size decides whether it fits in a cavity, and the supporter's cavity geometry
decides what "inside" means. Nuisance attributes must not bear on anything --
they are sampled from a family-specific distribution that never consults the
mechanism, and the M0 leakage audit is what checks that claim rather than
assuming it.

The appearance family is the axis the headline generalization split will hold
out. It lives strictly on the nuisance side: holding out a structural range
would ask the model to extrapolate physics, which is a different question.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .geometry import GravityFrame, MaterialShape, world_half_extents

DEVELOPMENT_FAMILIES: tuple[str, ...] = (
    "dev_sandstone",
    "dev_ceramic",
    "dev_brushed_metal",
    "dev_matte_polymer",
)

AUDIT_FAMILIES: tuple[str, ...] = (
    "audit_slate",
    "audit_porcelain",
    "audit_polished_steel",
    "audit_rubber",
)

ALL_FAMILIES: tuple[str, ...] = DEVELOPMENT_FAMILIES + AUDIT_FAMILIES


@dataclass(frozen=True)
class StructuralAttributes:
    """Attributes that may legitimately affect relation compatibility."""

    i_half_u: float
    i_half_t: float
    j_half_u: float
    j_half_t: float
    cavity_half_t: float
    floor_offset: float
    mass_i: float
    mass_j: float

    def shape(self) -> MaterialShape:
        return MaterialShape(
            half_u=self.j_half_u,
            half_t=self.j_half_t,
            cavity_half_t=self.cavity_half_t,
            floor_offset=self.floor_offset,
        )

    # `i_half_u` / `i_half_t` and the `j_*` pair are stated along the gravity
    # frame's axes -- "tall" and "wide" -- so they mean the same thing whichever
    # way gravity points. World-space box extents are derived per episode.

    def i_world_half_extent(self, frame: GravityFrame) -> tuple[float, float]:
        return world_half_extents(self.i_half_u, self.i_half_t, frame)

    def j_world_half_extent(self, frame: GravityFrame) -> tuple[float, float]:
        return world_half_extents(self.j_half_u, self.j_half_t, frame)


@dataclass(frozen=True)
class NuisanceAttributes:
    """Attributes that must be independent of the interaction mechanism."""

    appearance_family: str
    color: tuple[float, float, float]
    texture_id: int
    identity_id: int
    background_id: int
    camera_id: int
    render_seed: int

    def feature_vector(self) -> tuple[float, ...]:
        """The nuisance feature vector a leakage probe is allowed to see."""
        return (
            *self.color,
            float(self.texture_id),
            float(self.identity_id),
            float(self.background_id),
            float(self.camera_id),
            float(self.render_seed),
        )


@dataclass(frozen=True)
class Filler:
    index: int
    structural: StructuralAttributes
    nuisance: NuisanceAttributes


def _structural(rng: random.Random) -> StructuralAttributes:
    i_half_u = rng.uniform(0.25, 0.35)
    i_half_t = rng.uniform(0.25, 0.35)
    return StructuralAttributes(
        i_half_u=i_half_u,
        i_half_t=i_half_t,
        j_half_u=rng.uniform(0.90, 1.10),
        j_half_t=rng.uniform(1.10, 1.30),
        cavity_half_t=rng.uniform(0.60, 0.80),
        floor_offset=rng.uniform(0.50, 0.70),
        mass_i=rng.uniform(0.8, 1.2),
        mass_j=rng.uniform(14.0, 18.0),
    )


def nuisance_for(family: str, rng: random.Random) -> NuisanceAttributes:
    """Draw nuisance attributes from one appearance family's distribution.

    The only thing the family controls is *which* distribution the nuisance
    attributes come from. It never sees a mechanism, an ``S_0`` tuple, or a
    structural attribute, which is what makes an unseen-family split a question
    about appearance rather than about physics.
    """
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown appearance family {family!r}")
    anchor = ALL_FAMILIES.index(family)
    base = anchor / max(1, len(ALL_FAMILIES))
    return NuisanceAttributes(
        appearance_family=family,
        color=tuple(  # type: ignore[arg-type]
            min(1.0, max(0.0, base + rng.uniform(-0.05, 0.05))) for _ in range(3)
        ),
        texture_id=anchor,
        identity_id=anchor * 100 + rng.randrange(25),
        background_id=anchor * 10 + rng.randrange(4),
        camera_id=rng.randrange(3),
        render_seed=rng.randrange(1_000_000),
    )


def build_filler_pool(
    count: int,
    rng: random.Random,
    families: tuple[str, ...] = DEVELOPMENT_FAMILIES,
) -> list[Filler]:
    """A pool of distinct fillers, round-robining the appearance families."""
    if count < 1:
        raise ValueError("filler pool must be non-empty")
    if not families:
        raise ValueError("at least one appearance family is required")
    pool: list[Filler] = []
    for index in range(count):
        family = families[index % len(families)]
        pool.append(
            Filler(
                index=index,
                structural=_structural(rng),
                nuisance=nuisance_for(family, rng),
            )
        )
    return pool
