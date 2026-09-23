"""Frozen Phase IA configuration.

Kept in the tool tree rather than under ``configs/``. ``tools/ci_validate.py``
requires every newly added ``configs/**.py`` file to declare a top-level
``method`` assignment naming a registered SGG method, and a simulator config is
not one. ``tools/ontology_probe`` keeps its frozen settings the same way.

Thresholds belong here precisely because they are thresholds: a value written
into the frozen config before a run cannot be moved after seeing the result. The
one that matters most is :attr:`Phase1AConfig.leakage_mde_target` — the effect
size the leakage audit must be able to detect. Without it, "no leak found" and
"no leak findable at this sample size" look identical in a report, and the second
is a statement about the experiment rather than about the data.

:func:`config_digest` deliberately covers the world constants too. Gravity,
damping, the arena extents and the contact softness decide the dynamics as much
as the tuple and mechanism sets do, and while they live outside the dataclass a
change to any of them would leave every artifact stamp unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "relational_emergence.schema.v1"

WORLD_DEVELOPMENT = "development_world"
WORLD_AUDIT = "audit_world"
WORLDS: tuple[str, ...] = (WORLD_DEVELOPMENT, WORLD_AUDIT)

DEVELOPMENT_SEEDS: tuple[int, ...] = (0, 1, 2, 3)
AUDIT_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003)

GRAVITY_ANGLES_DEG: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

GEOMETRY_TOL = 1e-9


@dataclass(frozen=True)
class Phase1AConfig:
    """The frozen settings a Phase IA run is reproducible from."""

    schema_version: str = SCHEMA_VERSION
    world: str = WORLD_DEVELOPMENT
    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    audit_seeds: tuple[int, ...] = AUDIT_SEEDS
    gravity_angles_deg: tuple[float, ...] = GRAVITY_ANGLES_DEG
    geometry_tol: float = GEOMETRY_TOL
    # Base contexts per ``S_0`` tuple. The leakage probe's power and the
    # breadth of the ``J_ab`` average both scale with this, so it is a
    # pre-registered quantity rather than a convenience default.
    # 48, not 40: the leakage probe's minimum detectable effect scales as
    # 1/sqrt(n), and at 40 groups per tuple it lands at 0.0528 -- just above the
    # 0.05 the pre-registered target asks for, which would make the null result
    # underpowered by a hair and therefore not reportable as a null result.
    groups_per_tuple: int = 48
    horizon: int = 60
    filler_pool_size: int = 40
    leakage_permutations: int = 40
    oracle_group_limit: int = 60
    # The effect size the leakage audit is required to be able to detect, in
    # accuracy points above the compatibility-aware baseline. Declared before the
    # run rather than computed after it, because "no leak found" and "no leak
    # findable at this sample size" would otherwise be indistinguishable in the
    # report -- and the second is a statement about the experiment, not the data.
    leakage_mde_target: float = 0.05

    def __post_init__(self) -> None:
        if self.world not in WORLDS:
            raise ValueError(f"unknown world {self.world!r}, not in {WORLDS!r}")
        overlap = set(self.development_seeds) & set(self.audit_seeds)
        if overlap:
            raise ValueError(
                f"development and audit seed pools overlap: {sorted(overlap)}; "
                "the audit world must be disjoint by construction, not by convention"
            )
        if not self.development_seeds or not self.audit_seeds:
            raise ValueError("both seed pools must be non-empty")

    def seed_pool(self) -> tuple[int, ...]:
        return self.audit_seeds if self.world == WORLD_AUDIT else self.development_seeds

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = Phase1AConfig()

# Frozen generator seeds, one per world. The development value is the one Phase
# IA already ran under and must not move; the audit value is drawn from the audit
# seed pool rather than being a second arbitrary constant, so the two worlds are
# disjoint by the same rule the pools are.
WORLD_GENERATOR_SEEDS: dict[str, int] = {WORLD_DEVELOPMENT: 20260923, WORLD_AUDIT: 20261107}


def world_seed(world: str) -> int:
    """The filler-pool seed a world generates under."""
    if world not in WORLD_GENERATOR_SEEDS:
        raise ValueError(f"unknown world {world!r}, not in {sorted(WORLD_GENERATOR_SEEDS)!r}")
    return WORLD_GENERATOR_SEEDS[world]


def world_families(world: str) -> tuple[str, ...]:
    """The appearance families a world draws its fillers from.

    The audit world uses families the development world never generates, which is
    what makes the sealed run a test of *generalization to unseen appearance*
    rather than a second sample from the same distribution. Imported lazily so
    this module keeps its single dependency on ``common``.
    """
    from .simulator.fillers import AUDIT_FAMILIES, DEVELOPMENT_FAMILIES

    if world == WORLD_AUDIT:
        return AUDIT_FAMILIES
    if world == WORLD_DEVELOPMENT:
        return DEVELOPMENT_FAMILIES
    raise ValueError(f"unknown world {world!r}")


def config_digest(config: Phase1AConfig | None = None) -> str:
    """The provenance digest every artifact is stamped with.

    Covers :class:`~.simulator.state.WorldConstants` as well as the config
    dataclass. That is not belt-and-braces: gravity, damping, the arena extents
    and the contact softness decide the dynamics as much as the tuple and
    mechanism sets do, and while they live outside the dataclass a change to them
    would leave every stamp identical -- so an artifact could be re-generated
    under different physics and still look like the same experiment.

    Imported lazily to keep the config module free of a simulator import cycle.
    """
    import json
    from dataclasses import asdict

    from .common import sha256_text
    from .simulator.state import DEFAULT_CONSTANTS

    payload = {
        "config": (config or DEFAULT_CONFIG).as_dict(),
        "world_constants": asdict(DEFAULT_CONSTANTS),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))
