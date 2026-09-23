"""Compatibility between an observable ``S_0`` tuple and a latent mechanism ``M``.

``compatible`` and ``mechanism_active`` are distinct predicates (ADR 0005).
``C`` constrains sampling eligibility only. Whether a compatible law is
*currently manifesting* is dynamics, not eligibility -- an inactive mechanism is
compatible with the current ``S_0`` in the ordinary case, and that is the whole
point of defining ``M`` as a dispositional law. Activation is therefore not
defined in this module.

Three constraints are audited here, all stated over the enumerated realizable
tuples rather than over a hard-coded list:

1. every realizable tuple is compatible with at least three mechanisms,
   including ``free``;
2. every active mechanism spans at least two distinct tuples;
3. the tuple--mechanism bipartite graph is connected.

If a constraint fails, the escalation is to re-examine the definition of ``M`` or
the ``S_0`` factorization. Lowering the per-tuple mechanism count is not an
available response.
"""

from __future__ import annotations

from collections import deque

from .factors import S0, enumerate_realizable

MECHANISMS: tuple[str, ...] = ("containment", "support", "attachment", "free")
ACTIVE_MECHANISMS: tuple[str, ...] = ("containment", "support", "attachment")
FREE_MECHANISM = "free"

MIN_MECHANISMS_PER_TUPLE = 3
MIN_TUPLES_PER_ACTIVE_MECHANISM = 2


def compatible(s0: S0, mechanism: str) -> bool:
    """``C(S0_tuple, M)``.

    ``free`` is the non-interacting pair and is compatible everywhere by
    definition -- it is what makes every tuple's twin set non-empty.

    ``support`` and ``attachment`` are compatible with every tuple. A surface
    law that resists penetration along its own normal is well defined even when
    the contact is not gravity aligned (it simply does not hold ``i`` up), and a
    rigid link is well defined at any separation, a taut link being still a
    link.

    ``containment`` needs containing geometry: ``j``'s cavity when ``i`` is
    inside it, or ``j``'s rim when ``i`` sits on a tray-like top. With ``i``
    outside and not on top, ``j`` offers nothing to be contained by.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}, not in {MECHANISMS!r}")
    if mechanism == FREE_MECHANISM:
        return True
    if mechanism == "containment":
        return s0.containment == "inside" or s0.pose == "on_top"
    return True


def compatible_mechanisms(s0: S0) -> tuple[str, ...]:
    """The mechanisms eligible for sampling at this tuple, in canonical order."""
    return tuple(m for m in MECHANISMS if compatible(s0, m))


def compatibility_table() -> dict[str, tuple[str, ...]]:
    return {s0.key(): compatible_mechanisms(s0) for s0 in enumerate_realizable()}


def analytic_baseline_by_tuple() -> dict[str, float]:
    """The compatibility-aware M0 baseline, per realizable tuple.

    Defined as the majority-class share of ``M`` conditional on the joint tuple.
    Under the balanced sampling the sampler is required to perform, that share is
    ``1 / |compatible|`` -- which varies by tuple, so no fixed 25% or 33% figure
    is used anywhere. The audit additionally reports the *empirical* majority
    share from realized samples, because a sampler that fails to balance must
    show up as a baseline shift rather than as a silent change of denominator.
    """
    return {key: 1.0 / len(mechanisms) for key, mechanisms in compatibility_table().items()}


def _bipartite_connected(table: dict[str, tuple[str, ...]]) -> bool:
    """Breadth-first reachability over tuples and mechanisms."""
    if not table:
        return False
    adjacency: dict[str, set[str]] = {}
    for tuple_key, mechanisms in table.items():
        adjacency.setdefault(tuple_key, set()).update(mechanisms)
        for mechanism in mechanisms:
            adjacency.setdefault(mechanism, set()).add(tuple_key)
    seen = {next(iter(table))}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == len(adjacency)


def audit_compatibility() -> list[str]:
    """Return constraint violations; an empty list means the table is admissible."""
    table = compatibility_table()
    violations: list[str] = []

    for tuple_key, mechanisms in sorted(table.items()):
        if FREE_MECHANISM not in mechanisms:
            violations.append(f"{tuple_key}: missing the free mechanism")
        if len(mechanisms) < MIN_MECHANISMS_PER_TUPLE:
            violations.append(
                f"{tuple_key}: {len(mechanisms)} compatible mechanisms, "
                f"needs >= {MIN_MECHANISMS_PER_TUPLE}"
            )

    tuple_keys = set(table)
    for mechanism in ACTIVE_MECHANISMS:
        span = [key for key in tuple_keys if mechanism in table[key]]
        if len(span) < MIN_TUPLES_PER_ACTIVE_MECHANISM:
            violations.append(
                f"{mechanism}: spans {len(span)} tuples, "
                f"needs >= {MIN_TUPLES_PER_ACTIVE_MECHANISM}"
            )
        if not span:
            violations.append(f"{mechanism}: compatible with no tuple at all")

    if not _bipartite_connected(table):
        violations.append("tuple--mechanism bipartite graph is disconnected")

    return violations
