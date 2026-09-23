"""Role assignment for an episode, and its bidirectional ordered-pair view.

Roles come from the *law*, not from the argument order. ``containment`` names its
objects ``(inner, container)`` because that is what the constraint does, not
because one of them happens to be called ``i``; the probe object is the inner one
only because the sampler placed it there. Keeping the two ideas apart is what
lets the ordered-pair view below be derived rather than asserted.

Two properties are load-bearing for Phase IB and are guaranteed here rather than
left to a later evaluator:

``attachment`` is exactly symmetric
    A rigid link has no canonical orientation. The generative process does not
    prefer either endpoint, so a role-equivariance check on this law has a
    ground truth to be measured against. If attachment carried a hidden
    "active end", ``z_ij ~ z_ji`` would be a claim about the encoder rather than
    a property of the world, and there would be nothing to test it with.

The ordered-pair view is derived, never stored twice
    ``ordered_pairs`` produces both ``(i, j)`` and ``(j, i)`` records from one
    episode. Storing both independently would let them drift apart, which is the
    silent-misalignment failure this repository treats as its worst class of bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from .factors import S0

# The role each object plays, per law. `None` means the law assigns no role --
# a non-interacting pair relates nothing.
ROLES_BY_MECHANISM: dict[str, tuple[str, str]] = {
    "containment": ("inner", "container"),
    "support": ("supported", "supporter"),
    "attachment": ("endpoint_1", "endpoint_2"),
    "free": ("none", "none"),
}

SYMMETRIC_MECHANISMS: frozenset[str] = frozenset({"attachment", "free"})


@dataclass(frozen=True)
class Roles:
    """Which role the probe object and the supporter object play."""

    probe: str
    supporter: str
    symmetric: bool

    def reversed(self) -> Roles:
        """The same relation described from the other object first.

        For a symmetric law this is the same relation, which is exactly the
        property a role-equivariance check will later be scored against.
        """
        return Roles(probe=self.supporter, supporter=self.probe, symmetric=self.symmetric)


def roles_for(mechanism: str) -> Roles:
    if mechanism not in ROLES_BY_MECHANISM:
        raise ValueError(
            f"unknown mechanism {mechanism!r}, not in {sorted(ROLES_BY_MECHANISM)!r}"
        )
    probe, supporter = ROLES_BY_MECHANISM[mechanism]
    return Roles(probe=probe, supporter=supporter, symmetric=mechanism in SYMMETRIC_MECHANISMS)


@dataclass(frozen=True)
class OrderedPairRecord:
    """One direction of an unordered episode relation."""

    episode_id: str
    subject: str
    object: str
    relation: str
    subject_role: str
    object_role: str

    def as_dict(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "subject": self.subject,
            "object": self.object,
            "relation": self.relation,
            "subject_role": self.subject_role,
            "object_role": self.object_role,
        }


def ordered_pairs(
    episode_id: str,
    mechanism: str,
    s0: S0 | None = None,
    probe_name: str = "i",
    supporter_name: str = "j",
) -> tuple[OrderedPairRecord, OrderedPairRecord]:
    """Both directions, generated together so they cannot disagree.

    ``s0`` is accepted and unused today, and is kept in the signature because the
    role a law assigns can in principle depend on the configuration it is
    acting at -- for instance a ``support`` law whose supported object sits in
    the cavity rather than on the rim. Threading it now means that refinement
    does not become a schema change.

    The relation name is the *law*, never ``R`` and ``R^-1``: ``containment``
    with roles swapped is still containment, just observed from the container.
    Naming it two ways would double the label space for no information.
    """
    del s0
    roles = roles_for(mechanism)
    forward = OrderedPairRecord(
        episode_id=episode_id,
        subject=probe_name,
        object=supporter_name,
        relation=mechanism,
        subject_role=roles.probe,
        object_role=roles.supporter,
    )
    reverse = OrderedPairRecord(
        episode_id=episode_id,
        subject=supporter_name,
        object=probe_name,
        relation=mechanism,
        subject_role=roles.supporter,
        object_role=roles.probe,
    )
    return forward, reverse


def swap_is_involution(mechanism: str) -> bool:
    """Whether describing the relation from the other side twice is the identity.

    True for every law, and asserted for every law rather than for the symmetric
    ones alone: an evaluation that fits a swap transform expects
    ``T_swap^2 = I``, and a law that broke it would make that expectation wrong
    for a reason that has nothing to do with the encoder.
    """
    roles = roles_for(mechanism)
    return roles.reversed().reversed() == roles
