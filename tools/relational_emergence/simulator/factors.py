"""The factorized observable relational state ``S_0``.

``S_0`` is a product of three observable factors, not a single mutually
exclusive label (ADR 0005). ``inside`` / ``on_top`` / ``separated`` survive only
as derived readouts for reporting, and are never used as categories.

The factor domains are frozen here. :func:`enumerate_realizable` derives the
realizable tuples from them rather than listing them, so that changing a domain
cannot silently leave a stale hard-coded set behind.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTAINMENT_VALUES: tuple[str, ...] = ("inside", "outside")
CONTACT_VALUES: tuple[str, ...] = ("touching", "non_touching")
POSE_VALUES: tuple[str, ...] = ("on_top", "other")

FACTOR_DOMAINS: dict[str, tuple[str, ...]] = {
    "containment": CONTAINMENT_VALUES,
    "contact": CONTACT_VALUES,
    "pose": POSE_VALUES,
}


@dataclass(frozen=True, order=True)
class S0:
    """One joint assignment of the observable relational factors."""

    containment: str
    contact: str
    pose: str

    def __post_init__(self) -> None:
        for name, value in zip(FACTOR_DOMAINS, self.as_tuple(), strict=True):
            if value not in FACTOR_DOMAINS[name]:
                raise ValueError(
                    f"factor {name!r} has value {value!r}, "
                    f"not in {FACTOR_DOMAINS[name]!r}"
                )

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.containment, self.contact, self.pose)

    def as_dict(self) -> dict[str, str]:
        return dict(zip(FACTOR_DOMAINS, self.as_tuple(), strict=True))

    def key(self) -> str:
        return "|".join(self.as_tuple())


def jointly_consistent(containment: str, contact: str, pose: str) -> bool:
    """Whether a factor triple can hold simultaneously.

    The frozen factor definitions rule out exactly one combination: ``inside``
    with ``on_top``. ``on_top`` places ``i`` on the gravity side of ``j``'s
    up-facing face, while ``inside`` places it within ``j``'s cavity, on the far
    side of that same face.

    ``contact`` is not jointly constrained by the other two: an object inside a
    cavity may rest on its floor or float clear of it, and an object resting on
    a face may be in contact or hovering just above it.
    """
    for name, value in zip(FACTOR_DOMAINS, (containment, contact, pose), strict=True):
        if value not in FACTOR_DOMAINS[name]:
            raise ValueError(
                f"factor {name!r} has value {value!r}, not in {FACTOR_DOMAINS[name]!r}"
            )
    return not (containment == "inside" and pose == "on_top")


def enumerate_realizable() -> tuple[S0, ...]:
    """All realizable joint ``S_0`` tuples, enumerated from the factor domains.

    Never hard-code this set: the compatibility audit treats its size and shape
    as derived facts, and a stale literal would make the audit agree with itself
    rather than with the definitions.
    """
    return tuple(
        S0(containment, contact, pose)
        for containment in CONTAINMENT_VALUES
        for contact in CONTACT_VALUES
        for pose in POSE_VALUES
        if jointly_consistent(containment, contact, pose)
    )


def derived_readouts(s0: S0) -> dict[str, bool]:
    """Reporting readouts. These are *not* ``S_0`` categories."""
    return {
        "inside": s0.containment == "inside",
        "on_top": s0.pose == "on_top",
        "separated": s0.contact == "non_touching",
    }
