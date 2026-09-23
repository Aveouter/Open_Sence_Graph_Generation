"""Materialise counterfactual groups into the frozen episode rows.

Both phases read the same artifact. Phase IA audits it and gates on it; Phase IB
trains the representation arms on it. Writing it once, here, is what stops the
two from describing different experiments -- an earlier draft had the torch side
re-sample its own twins, which would have let the split, the compatibility table
or the noise pairing drift away from the ones that were actually audited.

A row is one ``(group, regime, mechanism)`` triple: one episode, with its own
trajectory and the context needed to identify its twins. The schema is flat
JSONL rather than a tensor dump because it is also the audit's input, and an
audit that reads a pickle cannot be checked by reading it.

Two families of fields, and the distinction is load-bearing:

``structural``
    may bear on compatibility, and is what the pair-recombination split buckets
    into object identities.
``nuisance``
    must not bear on anything, and is the axis the headline generalization split
    holds out. Carried in the row so a leakage audit can be run against the
    *materialised* data rather than the sampler's intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .counterfactuals import CounterfactualGroup
from .roles import ordered_pairs
from .splits import identity_pair
from .world import rollout

DATASET_SCHEMA_VERSION = "relational_emergence.dataset.v1"


@dataclass(frozen=True)
class DatasetRow:
    """One episode. Everything downstream reads these fields and nothing else."""

    group: str
    tuple_key: str
    regime: str
    mechanism: str
    # The laws this episode's ``S_0`` tuple admits, carried per row rather than
    # looked up: the Level-2 oracle's shuffled-mechanism arm needs to relabel a
    # row with a mechanism that is *compatible* with its configuration, and
    # re-deriving that from a table would let the table and the data disagree
    # without anything noticing.
    compatible: tuple[str, ...]
    filler_index: int
    appearance_family: str
    probe_identity: str
    supporter_identity: str
    roles: dict[str, str]
    frame_angle_rad: float
    x0: list[float]
    structural: list[float]
    nuisance: list[float]
    impulses: list[dict]
    trajectory: list[list[float]]

    def as_dict(self) -> dict:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "group": self.group,
            "tuple": self.tuple_key,
            "regime": self.regime,
            "mechanism": self.mechanism,
            "compatible": list(self.compatible),
            "filler_index": self.filler_index,
            "appearance_family": self.appearance_family,
            "probe_identity": self.probe_identity,
            "supporter_identity": self.supporter_identity,
            "roles": self.roles,
            "frame_angle_rad": self.frame_angle_rad,
            "x0": self.x0,
            "filler_structural": self.structural,
            "filler_nuisance": self.nuisance,
            "impulses": self.impulses,
            "trajectory": self.trajectory,
        }


def structural_features(group: CounterfactualGroup) -> list[float]:
    """The eight structural numbers, in a fixed order.

    Fixed rather than derived from the dataclass, because a field added to
    ``StructuralAttributes`` would otherwise silently widen this vector and
    change every downstream input width -- a schema change disguised as a
    refactor.
    """
    structural = group.filler.structural
    return [
        structural.i_half_u,
        structural.i_half_t,
        structural.j_half_u,
        structural.j_half_t,
        structural.cavity_half_t,
        structural.floor_offset,
        structural.mass_i,
        structural.mass_j,
    ]


def build_rows(
    groups: list[CounterfactualGroup],
    horizon: int,
    group_limit: int | None = None,
    extra_links: dict[str, dict[str, object]] | None = None,
) -> list[DatasetRow]:
    """Roll out every ``(group, regime, mechanism)`` triple.

    ``group_limit`` truncates by group index rather than by position, so a run
    that keeps the first N groups keeps the *same* N whatever order the caller
    built them in. Truncating a list would make the dataset depend on iteration
    order, which is the kind of accident that produces two "identical" runs that
    disagree.

    ``extra_links`` lets a caller attach a per-mechanism link offset recorded
    elsewhere -- Phase IB's transplant twins need the same rigid offset Phase IA
    recorded at ``X_0``, and recomputing it here would let the two diverged.
    """
    kept = [group for group in groups if group_limit is None or group.spec.group_index < group_limit]
    rows: list[DatasetRow] = []
    for group in kept:
        probe_identity, supporter_identity = identity_pair(group)
        for regime in sorted(group.schedules):
            schedule = group.schedules[regime]
            for mechanism in group.mechanisms:
                override = None
                if extra_links is not None:
                    override = extra_links.get(f"{group.spec.key()}|{regime}|{mechanism}")
                link_offset = (
                    override["link_offset"] if override is not None else group.link_offset(mechanism)
                )
                trajectory = rollout(
                    group.x0,
                    schedule,
                    group.filler.structural,
                    mechanism,
                    group.frame,
                    link_offset,
                    horizon,
                )
                # Both directions of the relation, generated together so they
                # cannot disagree; the schema carries roles by geometry rather
                # than by argument order (ADR 0005).
                forward, _ = ordered_pairs(group.spec.key(), mechanism, group.s0)
                rows.append(
                    DatasetRow(
                        group=group.spec.key(),
                        tuple_key=group.s0.key(),
                        regime=regime,
                        mechanism=mechanism,
                        compatible=group.mechanisms,
                        filler_index=group.filler.index,
                        appearance_family=group.filler.nuisance.appearance_family,
                        probe_identity=probe_identity,
                        supporter_identity=supporter_identity,
                        roles={"probe": forward.subject_role, "supporter": forward.object_role},
                        frame_angle_rad=group.frame.angle_rad,
                        x0=[round(value, 6) for value in group.x0.outcome_vector()],
                        structural=structural_features(group),
                        nuisance=list(group.filler.nuisance.feature_vector()),
                        impulses=[
                            {
                                "target": impulse.target,
                                "direction": impulse.direction,
                                "magnitude": impulse.magnitude,
                                "start": impulse.start,
                                "steps": impulse.steps,
                            }
                            for impulse in schedule.impulses
                        ],
                        trajectory=[
                            [round(value, 6) for value in state.outcome_vector()]
                            for state in trajectory
                        ],
                    )
                )
    return rows


def write_rows(path: Path, rows: list[DatasetRow]) -> int:
    return write_dicts(path, [row.as_dict() for row in rows])


def write_dicts(path: Path, rows: list[dict]) -> int:
    """Write already-serialised rows, for callers that read the file back first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def condition_of(row: dict) -> str:
    """The CCGP condition: appearance family crossed with intervention regime.

    Both halves are held out independently -- the family by the split, the regime
    by the intervention-richness experiment -- so a readout fit on some
    conditions and tested on others is being asked a question the training
    conditions do not answer.
    """
    return f"{row['appearance_family']}|{row['regime']}"
