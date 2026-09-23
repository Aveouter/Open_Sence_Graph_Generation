"""Turn the frozen episodes into the tensors the representation models consume.

Kept separate from the models so the input contract is in one place. Every field
is a function of the episode alone, so a context's twins produce identical inputs
and differ only in what the *world* does next -- which is what makes the
transplant endpoint meaningful later.

An "object" here is four numbers: position and velocity in the frame's own
coordinates. Frame coordinates rather than world axes because gravity rotates
between episodes and a model should not have to relearn the same physics four
times; the rotation is part of the world, not of the relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..simulator.actions import direction_vector
from ..simulator.geometry import GravityFrame

OBJECT_DIM = 4
GEOMETRY_DIM = 4


@dataclass(frozen=True)
class StreamSpec:
    history: int = 4
    horizon: int = 10
    """Steps of the trajectory used as prediction targets."""


def frame_series(frame: GravityFrame, states: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    """Every trajectory state for both bodies, in frame coordinates.

    Public because the transplant endpoint rolls a decoder from the same frame
    representation the arms were trained in; rebuilding it there would let the two
    drift by a rotation.
    """
    up, lateral = frame.up, frame.lateral
    series: list[tuple[float, ...]] = []
    for state in states:
        row: list[float] = []
        for object_index in (0, 1):
            offset = object_index * OBJECT_DIM
            position = (state[offset], state[offset + 1])
            velocity = (state[offset + 2], state[offset + 3])
            row.extend(
                [
                    position[0] * up[0] + position[1] * up[1],
                    position[0] * lateral[0] + position[1] * lateral[1],
                    velocity[0] * up[0] + velocity[1] * up[1],
                    velocity[0] * lateral[0] + velocity[1] * lateral[1],
                ]
            )
        series.append(tuple(row))
    return series


def action_vector(row: dict, step: int) -> tuple[float, ...]:
    """The world-space impulse at one step, as ``(x, y)`` for each body."""
    frame = GravityFrame(row["frame_angle_rad"])
    total = [0.0, 0.0, 0.0, 0.0]
    for impulse in row["impulses"]:
        start = impulse["start"]
        if not (start <= step < start + impulse["steps"]):
            continue
        unit = direction_vector(impulse["direction"], frame)
        base = 0 if impulse["target"] == "i" else 2
        total[base] += unit[0] * impulse["magnitude"]
        total[base + 1] += unit[1] * impulse["magnitude"]
    return tuple(total)


def relative_geometry(series: Sequence[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Relative configuration of the supporter from the probe, per step."""
    return [
        (row[4] - row[0], row[5] - row[1], row[6] - row[2], row[7] - row[3])
        for row in series
    ]


def mechanism_classes() -> tuple[str, ...]:
    """The label space, sorted, so a class index means the same in every run."""
    from ..simulator.compatibility import MECHANISMS

    return tuple(sorted(MECHANISMS))


def build_part(
    rows: list[dict],
    spec: StreamSpec,
    with_relation: bool = True,
    future_steps: int = 0,
) -> dict[str, Any]:
    """One split's tensors. Rows are episodes; each contributes ``horizon`` steps.

    ``with_relation`` decides whether the mechanism label is put in the batch at
    all. It defaults to true so a direct call is self-describing, but the
    predictive arms are built with it false: the phase's rule is that no relation
    label reaches a predictive model, and a tensor that is present but unread is
    a rule a reviewer has to take on trust, whereas an absent one is not. The
    supervised arms pass true because the auxiliary head *is* their definition.

    ``future_steps`` adds the next ``k`` states and the actions that produce them,
    for a decoder trained to roll rather than to predict one step. It is off by
    default because it costs ``k`` times the memory and only the transplant
    endpoint's decoder needs it. Steps past the end of an episode repeat the last
    observed state, so nothing is extrapolated and a short episode cannot silently
    contribute a shorter sequence than a long one.
    """
    import torch

    classes = mechanism_classes()
    if with_relation:
        for row in rows:
            if row["mechanism"] not in classes:
                raise ValueError(f"unknown mechanism {row['mechanism']!r} in the batch stream")

    state_i: list[tuple[float, ...]] = []
    state_j: list[tuple[float, ...]] = []
    structural: list[tuple[float, ...]] = []
    geometry: list[tuple[float, ...]] = []
    history: list[tuple[float, ...]] = []
    action: list[tuple[float, ...]] = []
    target: list[tuple[float, ...]] = []
    episode: list[int] = []
    futures: list[list[tuple[float, ...]]] = []
    action_sequences: list[list[tuple[float, ...]]] = []

    for episode_index, row in enumerate(rows):
        frame = GravityFrame(row["frame_angle_rad"])
        states = [row["x0"], *row["trajectory"]]
        series = frame_series(frame, states)
        relative = relative_geometry(series)
        for step in range(spec.horizon):
            # `series[step]` is X_step, `series[step + 1]` is X_{step+1}.
            padded = [series[0]] * spec.history + series[1 : step + 1]
            window = padded[-spec.history :]
            state_i.append(series[step][:OBJECT_DIM])
            state_j.append(series[step][OBJECT_DIM:])
            structural.append(tuple(row["filler_structural"]))
            geometry.append(relative[step])
            history.append(tuple(value for entry in window for value in entry[:OBJECT_DIM]))
            action.append(action_vector(row, step))
            target.append(series[step + 1])
            episode.append(episode_index)
            if future_steps:
                horizon = [
                    series[min(step + offset, len(series) - 1)] for offset in range(1, future_steps + 1)
                ]
                futures.append(horizon)
                action_sequences.append(
                    [
                        action_vector(row, min(step + offset, spec.horizon - 1))
                        for offset in range(future_steps)
                    ]
                )

    part = {
        "state_i": torch.tensor(state_i, dtype=torch.float32),
        "state_j": torch.tensor(state_j, dtype=torch.float32),
        "structural": torch.tensor(structural, dtype=torch.float32),
        "geometry_i_to_j": torch.tensor(geometry, dtype=torch.float32),
        "history": torch.tensor(history, dtype=torch.float32).view(len(state_i), spec.history, OBJECT_DIM),
        "action": torch.tensor(action, dtype=torch.float32),
        "target": torch.tensor(target, dtype=torch.float32),
        "episode": torch.tensor(episode, dtype=torch.long),
    }
    if with_relation:
        index_of = {name: position for position, name in enumerate(classes)}
        part["relation"] = torch.tensor(
            [index_of[row["mechanism"]] for row in rows for _ in range(spec.horizon)],
            dtype=torch.long,
        )
    if future_steps:
        part["future"] = torch.tensor(futures, dtype=torch.float32)
        part["action_sequence"] = torch.tensor(action_sequences, dtype=torch.float32)
    return part


ACTUATED_BODIES = 2
ACTION_PER_BODY = 2
"""A world-space impulse is a 2-vector, and the probe can push either body."""


def action_dim() -> int:
    """The action block: a world-space impulse for each of the two bodies."""
    return ACTUATED_BODIES * ACTION_PER_BODY


def struct_dim() -> int:
    return 8


STATE_DIM = 2 * OBJECT_DIM
