"""Four-body shared-supporter world and a label-free cutoff contract.

The v1 contact laws treat the supporter as an externally driven body (no
backreaction). Multiple probes can therefore share its exact trajectory. Probes
do not collide with each other. This is a star interaction graph, not general
many-body physics. Object slots are permuted independently of the query law.
"""

from __future__ import annotations

import random
from dataclasses import replace

from ..simulator.actions import ActionSchedule
from ..simulator.counterfactuals import build_group
from ..simulator.factors import enumerate_realizable
from ..simulator.fillers import build_filler_pool
from ..simulator.mechanisms import mechanism_is_manifesting
from ..simulator.state import WorldState
from ..simulator.world import rollout
from .protocol import DEV_FAMILIES, DEV_SEED, FROM_LEGACY, Protocol


def _coordinates(vector, frame):
    return [
        sum(a * b for a, b in zip(vector, axis, strict=True))
        for axis in (frame.up, frame.lateral)
    ]


def build_scenes(
    protocol: Protocol,
    *,
    world: str = "development",
    numerical_variant: int = 0,
    noise_offset: int = 0,
) -> list[dict]:
    if world != "development":
        raise ValueError(
            "fresh v2 audit allocation is sealed; development validation is required"
        )
    rng = random.Random(DEV_SEED)
    pool = build_filler_pool(40, rng)
    rows = []
    for tuple_index, s0 in enumerate(enumerate_realizable()):
        for index in range(protocol.groups_per_tuple):
            # Two independent base contexts per filler support the within-filler
            # transplant without reusing a target episode as its own source.
            filler = pool[(tuple_index * 5 + (index // 2) * 3) % len(pool)]
            # Fresh context and noise streams as well as fresh filler allocation.
            group_index = DEV_SEED + tuple_index * protocol.groups_per_tuple + index
            primary = build_group(
                s0, group_index, filler, protocol.horizon, noise_offset=noise_offset
            )
            probes = [primary]
            for slot in range(1, protocol.objects - 1):
                other = build_group(
                    s0,
                    group_index + 100003 * slot,
                    filler,
                    protocol.horizon,
                    noise_offset=noise_offset,
                )
                # One gravity frame and one supporter for the entire scene.
                old_frame = other.frame
                pos = _coordinates(other.x0.i.pos, old_frame)
                vel = _coordinates(other.x0.i.vel, old_frame)

                def rotate(values, frame=primary.frame):
                    return tuple(
                        values[0] * frame.up[d] + values[1] * frame.lateral[d]
                        for d in range(2)
                    )

                body = replace(
                    other.x0.i,
                    pos=rotate(pos),
                    vel=rotate(vel),
                    half_extent=filler.structural.i_world_half_extent(primary.frame),
                )
                probes.append(
                    replace(
                        other, frame=primary.frame, x0=WorldState(body, primary.x0.j)
                    )
                )
            order = list(range(protocol.objects))
            rng.shuffle(order)
            query = (order.index(0), order.index(protocol.objects - 1))
            background = [rng.choice(primary.mechanisms) for _ in probes[1:]]
            for regime in sorted(primary.schedules):
                schedules = [primary.schedules[regime]]
                for probe in probes[1:]:
                    schedules.append(
                        ActionSchedule(
                            tuple(
                                [
                                    a
                                    for a in probe.schedules[regime].impulses
                                    if a.target == "i"
                                ]
                                + [
                                    a
                                    for a in primary.schedules[regime].impulses
                                    if a.target == "j"
                                ]
                            )
                        )
                    )
                for mechanism in primary.mechanisms:
                    laws = [mechanism, *background]
                    series, active = [], []
                    for probe, schedule, law in zip(
                        probes, schedules, laws, strict=True
                    ):
                        states = [
                            probe.x0,
                            *rollout(
                                probe.x0,
                                schedule,
                                filler.structural,
                                law,
                                primary.frame,
                                probe.link_offset(law),
                                protocol.horizon,
                                numerical_variant=numerical_variant,
                            ),
                        ]
                        series.append(states)
                        active.append(
                            [
                                mechanism_is_manifesting(
                                    law,
                                    state,
                                    filler.structural,
                                    primary.frame,
                                    probe.link_offset(law),
                                )
                                for state in states
                            ]
                        )
                    scene = []
                    actions = []
                    for t in range(protocol.horizon + 1):
                        if any(states[t].j != series[0][t].j for states in series):
                            raise AssertionError(
                                "shared supporter trajectories disagree"
                            )
                        bodies = [states[t].i for states in series] + [series[0][t].j]
                        scene.append(
                            [
                                _coordinates(bodies[k].pos, primary.frame)
                                + _coordinates(bodies[k].vel, primary.frame)
                                for k in order
                            ]
                        )
                        if t < protocol.horizon:
                            impulses = [
                                schedule.impulse_vector("i", t, primary.frame)
                                for schedule in schedules
                            ]
                            impulses.append(
                                schedules[0].impulse_vector("j", t, primary.frame)
                            )
                            actions.append(
                                [
                                    _coordinates(impulses[k], primary.frame)
                                    for k in order
                                ]
                            )
                    s = filler.structural
                    structural = [
                        [s.i_half_u, s.i_half_t, s.mass_i, 0.0, 0.0] for _ in probes
                    ] + [
                        [
                            s.j_half_u,
                            s.j_half_t,
                            s.mass_j,
                            s.cavity_half_t,
                            s.floor_offset,
                        ]
                    ]
                    rows.append(
                        {
                            "group": f"v2:{primary.spec.key()}",
                            "tuple": s0.key(),
                            "regime": regime,
                            "mechanism": FROM_LEGACY[mechanism],
                            "compatible": [FROM_LEGACY[m] for m in primary.mechanisms],
                            "filler": filler.index,
                            "family": DEV_FAMILIES[filler.index % 4],
                            "nuisance": list(filler.nuisance.feature_vector()),
                            "structural": [structural[k] for k in order],
                            "query": list(query),
                            "states": scene,
                            "actions": actions,
                            "mechanism_active": active[0],
                            "mechanism_exposed": any(active[0][: protocol.cutoff + 1]),
                        }
                    )
    # Exposure is an analysis-only opportunity for the compatible mechanisms
    # to be distinguished. It is shared by twins, including free: an active
    # response is not required for evidence that an interaction is absent.
    contexts = {}
    for row in rows:
        contexts.setdefault((row["group"], row["regime"]), []).append(row)
    for twins in contexts.values():
        reference = twins[0]["states"][: protocol.cutoff + 1]
        exposed = any(
            abs(a - b) > protocol.numerical_tolerance
            for twin in twins[1:]
            for frame_a, frame_b in zip(reference, twin["states"], strict=False)
            for object_a, object_b in zip(frame_a, frame_b, strict=True)
            for a, b in zip(object_a, object_b, strict=True)
        )
        for twin in twins:
            twin["mechanism_exposed"] = exposed
    return rows


def build_batch(rows: list[dict], protocol: Protocol) -> tuple[dict, object]:
    """One row per episode at one cutoff; targets and metadata stay outside inputs.

    q_M sees positions and past actions, no derivatives or frame timestamps.
    The predictor separately sees the same current state and future actions in
    all arms. Static codes see only their explicitly named position snapshot.
    """
    import torch

    if not rows:
        raise ValueError("empty scene batch")
    cut, steps = protocol.cutoff, protocol.prediction_steps
    for row in rows:
        if len(row["states"]) <= cut + steps or len(row["actions"]) < cut + steps:
            raise ValueError(
                "insufficient true history/future; no repeated-state padding"
            )
        if any(len(frame) != protocol.objects for frame in row["states"]):
            raise ValueError("scene object count disagrees with protocol")
    tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    # Incoming actions are attached to the resulting frame; t=0 has no incoming action.
    frames = []
    for row in rows:
        incoming = [
            [[0.0, 0.0] for _ in range(protocol.objects)],
            *row["actions"][:cut],
        ]
        frames.append(
            [
                [
                    state[:2] + action
                    for state, action in zip(states, actions, strict=True)
                ]
                for states, actions in zip(
                    row["states"][: cut + 1], incoming, strict=True
                )
            ]
        )
    inputs = {
        "frames": tensor(frames),
        "initial_positions": tensor(
            [[s[:2] for s in row["states"][0]] for row in rows]
        ),
        "post_positions": tensor([[s[:2] for s in row["states"][cut]] for row in rows]),
        "state": tensor([row["states"][cut] for row in rows]),
        "structural": tensor([row["structural"] for row in rows]),
        "actions": tensor([row["actions"][cut : cut + steps] for row in rows]),
    }
    targets = tensor([row["states"][cut + 1 : cut + steps + 1] for row in rows])
    return inputs, targets


def split_groups(rows: list[dict], seed: int = 0) -> dict[str, list[int]]:
    """Hold out an entire filler family, then validation base groups within fit families."""
    family = sorted({r["family"] for r in rows})[-1]
    test = [i for i, r in enumerate(rows) if r["family"] == family]
    groups = sorted({r["group"] for r in rows if r["family"] != family})
    random.Random(seed).shuffle(groups)
    validation = set(groups[: max(1, len(groups) // 4)])
    split = {"test": test, "val": [], "train": []}
    for i, row in enumerate(rows):
        if row["family"] != family:
            split["val" if row["group"] in validation else "train"].append(i)
    if any(not indices for indices in split.values()):
        raise ValueError("empty group/family partition")
    return split
