"""Counterfactual groups and the twin construction.

A group is one *base context*: fillers, ``S_0`` tuple, ``X_0``, gravity frame and
one action schedule per regime. Everything a twin pair shares is fixed here, and
the only thing that varies between twins is the mechanism -- which is the
condition every identifiability measurement in Phase I rests on.

Two independent samplers feed a group, and keeping them separate is what makes
the leakage probes meaningful:

``context_rng``
    draws the filler, the frame, ``X_0`` and the action schedules. It is seeded
    from the group index and never sees a mechanism.
``noise_rng``
    draws the extra perturbation used for the same-mechanism null rollout. It
    also never sees a mechanism.

Because the action schedules are drawn once per group and reused across the
regimes' *twin sets* and across every mechanism, ``P(A | M) = P(A)`` holds by
construction rather than by argument.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .actions import REGIMES, ActionSchedule, sample_schedule
from .compatibility import compatible_mechanisms
from .factors import S0
from .fillers import Filler
from .geometry import GravityFrame, Vec2
from .sampling import sample_for_tuple
from .state import DEFAULT_CONSTANTS, WorldConstants, WorldState

FRAME_ANGLES_DEG: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)


@dataclass(frozen=True)
class GroupSpec:
    """Everything needed to rebuild a group deterministically."""

    s0: S0
    group_index: int
    filler_index: int
    seed: int

    def key(self) -> str:
        return f"{self.s0.key()}#{self.group_index}"


@dataclass(frozen=True)
class CounterfactualGroup:
    spec: GroupSpec
    filler: Filler
    frame: GravityFrame
    x0: WorldState
    schedules: dict[str, ActionSchedule]
    mechanisms: tuple[str, ...]

    @property
    def s0(self) -> S0:
        return self.spec.s0

    def link_offset(self, mechanism: str) -> Vec2 | None:
        """The frozen offset an attached twin holds, recorded once at ``X_0``."""
        if mechanism != "attachment":
            return None
        return (
            self.x0.i.pos[0] - self.x0.j.pos[0],
            self.x0.i.pos[1] - self.x0.j.pos[1],
        )


def build_group(
    s0: S0,
    group_index: int,
    filler: Filler,
    horizon: int,
    constants: WorldConstants = DEFAULT_CONSTANTS,
    noise_offset: int = 0,
) -> CounterfactualGroup:
    """Build one base context.

    ``noise_offset`` re-draws only the initial velocity perturbation, producing
    the same-mechanism twin used to establish the null floor. It deliberately
    does not touch the filler, the frame, the positions or the actions, so the
    null measures noise and nothing else.
    """
    seed = hash_stable(f"{s0.key()}|{group_index}|{filler.index}")
    context_rng = random.Random(seed)
    angle = FRAME_ANGLES_DEG[context_rng.randrange(len(FRAME_ANGLES_DEG))]
    frame = GravityFrame(angle * 3.141592653589793 / 180.0)
    x0 = sample_for_tuple(s0, filler.structural, frame, random.Random(seed ^ noise_offset), constants)
    schedules = {
        name: sample_schedule(policy, horizon, random.Random(seed + 7717 * (index + 1)))
        for index, (name, policy) in enumerate(sorted(REGIMES.items()))
    }
    return CounterfactualGroup(
        spec=GroupSpec(s0=s0, group_index=group_index, filler_index=filler.index, seed=seed),
        filler=filler,
        frame=frame,
        x0=x0,
        schedules=schedules,
        mechanisms=compatible_mechanisms(s0),
    )


def hash_stable(text: str) -> int:
    """A process-independent seed. ``hash()`` is salted per interpreter."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
