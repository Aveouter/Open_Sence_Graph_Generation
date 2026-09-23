"""Phase IB: the factorial arms and the three primary endpoints.

Phase IA asked whether the mechanism is identifiable from the observable state
given the actions. This phase asks the different question the plan is actually
about: *what* experience and *what* inductive bias make a predictive model form a
relational code that survives a change of filler.

The run is a factorial, and every non-compared thing is held fixed -- same frozen
dataset, same contexts, same optimiser, same epoch budget, same parameter count
to within 1%. What varies between arms is the bottleneck and the temporal order,
which is what makes a difference in the endpoints attributable to them.

Endpoints, all measured on the test side of a held-out split, all against a
control that must fail:

``probe``
    a linear readout of ``M`` from ``z``. Diagnostic only: it is the weakest of
    the three and the plan forbids it as primary evidence.
``ccgp``
    the same readout fit on one set of appearance families and scored on another.
    This is the headline.
``role_equivariance``
    one orthogonal swap transform fit on development families, scored on unseen
    ones, against a deranged-pairing control.
``transplant``
    a decoder fed a *counterfactual* code from a different filler, against the
    same source under a different mechanism. This is the causal endpoint.

Writes, under ``outputs/analysis/relational_emergence/phase1b/``:

``config.json``      the frozen settings, the world, and the arguments
``dataset.jsonl``    the frozen episodes every arm read
``splits.json``      the context, condition and validation partitions
``arms.json``        per-arm training records, endpoint values, and parameter counts
``endpoints.json``   the same, flattened, plus the transplant summaries
``transplant.json``  every transplant trial and its contrast
``manifest.json``    artifact hashes
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

from ..common import DEFAULT_OUTPUT_ROOT, git_sha, sha256_file, status_block, write_json
from ..config import (
    DEFAULT_CONFIG,
    WORLDS,
    Phase1AConfig,
    config_digest,
    world_families,
    world_seed,
)
from ..eval import design
from ..eval.endpoints import ccgp, role_equivariance
from ..eval.transplant import (
    cosine,
    flatten,
    paired_transplant,
    sensitivity,
    trajectory_error,
    trajectory_scale,
)
from ..simulator.splits import DEFAULT_SPLIT_SEED, SPLIT_KINDS, build_split
from .run_phase1a import build_groups

TRANSPLANT_HORIZON = 10
"""Steps the decoder is rolled before scoring.

Ten, because that is the horizon the Level-2 oracle was validated at and the
horizon the Phase IA saturation curve selected; a longer rollout would compound
decoder error past the point where the comparison is about the code.
"""

LABEL_BUDGETS: tuple[int | None, ...] = (1, 5, 10, 50, None)
"""H3's x-axis: labels per class for the frozen-encoder readout, then the full set."""


def _index(rows: list[dict]) -> dict[str, dict]:
    return {f"{row['group']}|{row['regime']}|{row['mechanism']}": row for row in rows}


def _by_key(rows: list[dict], field: str) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["group"], set()).add(row[field])
    return {key: sorted(value) for key, value in grouped.items()}


def _rows_for(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    keep = set(keys)
    return [row for row in rows if row["group"] in keep]


def _as_tuples(tensor: Any) -> list[tuple[float, ...]]:
    return [tuple(float(value) for value in row) for row in tensor.tolist()]


def _repeat(values: list[str], times: int) -> list[str]:
    return [value for value in values for _ in range(times)]


def _frame_scale(rows: list[dict], horizon: int) -> tuple[float, ...]:
    """A frozen per-dimension scale over *frame-coordinate* states.

    Phase IA's scale is computed over world coordinates. Gravity points along a
    different world axis in each episode, so reusing that vector here would weight
    a frame's up-axis by how often it happened to be the world's x -- a scale that
    depends on the camera rather than on the dynamics. This one is built from the
    same trajectories after the frame rotation, so a unit of "up" means the same
    thing in every episode.
    """
    from ..models.batch_stream import frame_series
    from ..simulator.geometry import GravityFrame

    vectors: list[tuple[float, ...]] = []
    for row in rows:
        frame = GravityFrame(row["frame_angle_rad"])
        states = [row["x0"], *row["trajectory"]][: horizon + 1]
        vectors.extend(frame_series(frame, states))
    return trajectory_scale(vectors)


def _label_curve(
    fit_latents: list[tuple[float, ...]],
    fit_mechanisms: list[str],
    fit_conditions: list[str],
    test_latents: list[tuple[float, ...]],
    test_mechanisms: list[str],
    test_conditions: list[str],
    condition_partition: design.ConditionPartition,
    chance: dict[str, float],
) -> dict[str, Any]:
    """CCGP as a function of how many labels per class the readout was fit on.

    The H3 endpoint, and the reason it is a curve rather than a point: the
    question is not whether the code decodes but how *few* labels reach the
    mechanism in it. A frozen encoder whose code a handful of labels can read is
    carrying the structure in an accessible form; one that needs the whole label
    budget is carrying it in a form a large readout had to find.

    The test side is fixed across budgets on purpose. Shrinking it with the label
    budget would make the curve measure two things at once.
    """
    from ..models.batch_stream import mechanism_classes

    index_of = {name: position for position, name in enumerate(mechanism_classes())}
    fit_rows = [i for i, c in enumerate(fit_conditions) if c in condition_partition.train]
    test_rows = [i for i, c in enumerate(test_conditions) if c in condition_partition.test]
    if not fit_rows or not test_rows:
        return {"unavailable": "the condition partition does not meet the evaluated rows"}
    labels = [index_of[fit_mechanisms[i]] for i in fit_rows]

    curve: dict[str, Any] = {}
    for budget in LABEL_BUDGETS:
        if budget is not None and budget * len(set(labels)) >= len(fit_rows):
            curve[str(budget)] = {"skipped": "the budget exceeds the available labels"}
            continue
        selected = (
            fit_rows
            if budget is None
            else design.subsample_labels(fit_rows, labels, budget, seed=budget)
        )
        keep = sorted(set(selected) | set(test_rows))
        result = ccgp(
            [*[fit_latents[i] for i in selected], *[test_latents[i] for i in test_rows]],
            [*[fit_mechanisms[i] for i in selected], *[test_mechanisms[i] for i in test_rows]],
            [*[fit_conditions[i] for i in selected], *[test_conditions[i] for i in test_rows]],
            train_conditions={fit_conditions[i] for i in selected},
            test_conditions=set(condition_partition.test),
            chance_level=chance,
        )
        curve[str(budget)] = result.as_dict() | {"n_fit_rows": len(keep) - len(test_rows)}
    return curve


def _evaluate_arm(
    model: Any,
    spec: Any,
    fit_rows: list[dict],
    test_rows: list[dict],
    condition_partition: design.ConditionPartition,
    chance: dict[str, float],
) -> dict[str, Any]:
    """The endpoints that need no decoder, plus the diagnostic probe."""
    import torch

    from ..models.batch_stream import build_part

    fit_part = build_part(fit_rows, spec, with_relation=False, future_steps=1)
    test_part = build_part(test_rows, spec, with_relation=False)
    fit_latents = _as_tuples(freeze(model, fit_part))
    test_latents = _as_tuples(freeze(model, test_part))
    fit_mechanisms = _repeat([row["mechanism"] for row in fit_rows], spec.horizon)
    test_mechanisms = _repeat([row["mechanism"] for row in test_rows], spec.horizon)
    fit_conditions = _repeat([design.condition_of(row) for row in fit_rows], spec.horizon)
    test_conditions = _repeat([design.condition_of(row) for row in test_rows], spec.horizon)

    # The ordinary probe: the *split sides* are the conditions, so it measures
    # whether the code is readable on contexts the encoder did not train on.
    # Reported next to the two primary endpoints and explicitly not among them.
    fit_chance = sum(chance[design.condition_of(row)] for row in fit_rows) / len(fit_rows)
    test_chance = sum(chance[design.condition_of(row)] for row in test_rows) / len(test_rows)
    probe = ccgp(
        [*fit_latents, *test_latents],
        [*fit_mechanisms, *test_mechanisms],
        [*["fit"] * len(fit_latents), *["test"] * len(test_latents)],
        train_conditions={"fit"},
        test_conditions={"test"},
        chance_level={"fit": fit_chance, "test": test_chance},
    )

    headline = ccgp(
        [*fit_latents, *test_latents],
        [*fit_mechanisms, *test_mechanisms],
        [*fit_conditions, *test_conditions],
        train_conditions=set(condition_partition.train),
        test_conditions=set(condition_partition.test),
        chance_level=chance,
    )

    # Role equivariance is fitted on one set of families and scored on another, so
    # it needs both sides in one call; scoring it on the test side alone would
    # leave the fit partition empty.
    both_part = build_part([*fit_rows, *test_rows], spec, with_relation=False)
    with torch.no_grad():
        z_ij, z_ji = model.pair_latents(both_part)
    role = role_equivariance(
        _as_tuples(z_ij),
        _as_tuples(z_ji),
        _repeat([row["appearance_family"] for row in [*fit_rows, *test_rows]], spec.horizon),
        fit_families={design.family_of_condition(c) for c in condition_partition.train},
        eval_families={design.family_of_condition(c) for c in condition_partition.test},
    )

    return {
        "probe": probe.as_dict(),
        "ccgp_unseen_family": headline.as_dict(),
        "role_equivariance": role.as_dict(),
        "ccgp_by_regime": _ccgp_by_regime(
            fit_latents,
            fit_mechanisms,
            fit_rows,
            spec,
            test_latents,
            test_mechanisms,
            test_rows,
            condition_partition,
            chance,
        ),
        "code_ablation": _code_ablation(model, fit_part),
        "label_efficiency": _label_curve(
            fit_latents,
            fit_mechanisms,
            fit_conditions,
            test_latents,
            test_mechanisms,
            test_conditions,
            condition_partition,
            chance,
        ),
        "n_fit_rows": len(fit_rows),
        "n_test_rows": len(test_rows),
    }


def freeze(model: Any, part: dict[str, Any]) -> Any:
    from ..models.transplant import freeze_latents

    return freeze_latents(model, part)


def standardise_latents(reference: Any, latents: Any) -> Any:
    """Scale a code by the *fit* split's per-dimension statistics.

    Not cosmetic. A predictive encoder has no term keeping its code informative,
    so the code is free to be dominated by one large episode-independent component
    with the mechanism carried in a few percent of its variance. That is a real
    property of the representation and the endpoints that read it linearly are
    unaffected -- a ridge readout finds the small subspace -- but a small MLP
    decoder fed raw values will condition on the dominant component and barely
    respond to the rest, and the transplant would then report the decoder's
    insensitivity as the code's.

    Standardising with statistics frozen from the fit split is what any downstream
    user would do, is identical for every arm, and leaks nothing about the test
    side into the instrument.
    """
    import torch

    means = reference.mean(dim=0, keepdim=True)
    deviations = reference.std(dim=0, keepdim=True)
    # A latent dimension that never varies carries nothing, and dividing by its
    # dust would manufacture a signal out of floating point.
    deviations = torch.where(deviations > 1e-6, deviations, torch.ones_like(deviations))
    return (latents - means) / deviations


def oracle_codes(rows: list[dict], spec: Any) -> Any:
    """A one-hot mechanism label standing in for the learned code.

    The must-fire control for the whole endpoint. If the decoder cannot convert a
    code that *is* the mechanism into a prediction that moves the way the law
    moves the world, then a null transplant result says nothing about the learned
    representation -- it says the instrument is broken. This is the same role
    ``leaky_sampler_control`` plays in Phase IA: a case whose answer is known in
    advance, run through the identical machinery.
    """
    import torch

    from ..models.batch_stream import mechanism_classes

    index_of = {name: position for position, name in enumerate(mechanism_classes())}
    encoded = [
        [1.0 if index == index_of[row["mechanism"]] else 0.0 for index in range(len(index_of))]
        for row in rows
        for _ in range(spec.horizon)
    ]
    return torch.tensor(encoded, dtype=torch.float32)


def _transplant(
    model: Any,
    decoder_config: Any,
    rows: list[dict],
    partition: design.ContextPartition,
    fit_rows: list[dict],
    test_rows: list[dict],
    spec: Any,
    scale: tuple[float, ...],
    code: str = "learned",
) -> dict[str, Any]:
    """Train the instrument, then read this arm's code through it."""
    from ..models.batch_stream import (
        action_dim,
        action_vector,
        build_part,
        frame_series,
        struct_dim,
    )
    from ..models.transplant import roll_from_state, train_decoder
    from ..simulator.geometry import GravityFrame

    steps = decoder_config.rollout_steps
    fit_part = build_part(fit_rows, spec, with_relation=False, future_steps=steps)
    test_part = build_part(test_rows, spec, with_relation=False)
    if code == "oracle":
        fit_latents = oracle_codes(fit_rows, spec)
        latents = oracle_codes(test_rows, spec)
    elif code == "learned":
        raw_fit = freeze(model, fit_part)
        fit_latents = standardise_latents(raw_fit, raw_fit)
        latents = standardise_latents(raw_fit, freeze(model, test_part))
    else:
        raise ValueError(f"unknown code {code!r}, expected 'learned' or 'oracle'")
    fitted = train_decoder(
        fit_part,
        fit_latents,
        state_dim=8,
        struct_dim=struct_dim(),
        action_dim=action_dim(),
        latent=int(fit_latents.shape[1]),
        config=decoder_config,
    )
    decoder = fitted["model"]
    # The flat index of each episode's *first* step. The latent tensors are flat
    # over (episode, step) -- each episode contributes `spec.horizon` rows -- so
    # indexing them by the episode number would read a different episode's step
    # altogether. The transplant rolls from step zero, so that is the row it needs.
    part_index = {
        f"{row['group']}|{row['regime']}|{row['mechanism']}": position * spec.horizon
        for position, row in enumerate(test_rows)
    }
    row_index = _index(rows)

    # Targets and sources come from the *materialised* rows, not from the split.
    # A context with no episodes cannot supply or receive a latent, and deriving
    # these lists from the split would turn that into a missing key mid-loop.
    test_keys = sorted({row["group"] for row in test_rows})
    trials = design.transplant_plan(
        targets=test_keys,
        sources=test_keys,
        regimes_by_key=_by_key(rows, "regime"),
        mechanisms_by_key=_by_key(rows, "mechanism"),
        tuple_by_key={row["group"]: row["tuple"] for row in rows},
        filler_by_key={row["group"]: row["filler_index"] for row in rows},
    )

    errors: dict[str, list[float]] = {"self": [], "correct": [], "wrong": []}
    clusters: list[str] = []
    targets_seen: list[str] = []
    records: list[dict] = []
    sensitivity_rows: list[dict] = []
    degenerate_pairs = 0
    inactive_law_pairs = 0
    for trial in trials:
        target_row = row_index.get(f"{trial.target_key}|{trial.regime}|{trial.target_mechanism}")
        correct_index = part_index.get(f"{trial.source_key}|{trial.regime}|{trial.target_mechanism}")
        wrong_indices = [
            part_index[f"{trial.source_key}|{trial.regime}|{name}"]
            for name in trial.wrong_mechanisms
            if f"{trial.source_key}|{trial.regime}|{name}" in part_index
        ]
        self_index = part_index.get(f"{trial.target_key}|{trial.regime}|{trial.target_mechanism}")
        if target_row is None or correct_index is None or not wrong_indices or self_index is None:
            continue

        frame = GravityFrame(target_row["frame_angle_rad"])
        series = frame_series(frame, [target_row["x0"], *target_row["trajectory"]])
        actions = [action_vector(target_row, step) for step in range(TRANSPLANT_HORIZON)]
        structural = _structural_tensor(target_row)
        truth = [list(entry) for entry in series[1 : TRANSPLANT_HORIZON + 1]]

        def roll(latent_index: int) -> list[list[float]]:
            return roll_from_state(
                decoder,
                initial_state=list(series[0]),
                structural=structural,
                actions=actions,
                latent_vector=latents[latent_index],
                steps=TRANSPLANT_HORIZON,
            )

        self_error = trajectory_error(roll(self_index), truth, list(scale))
        correct_error = trajectory_error(roll(correct_index), truth, list(scale))
        wrong_errors = [trajectory_error(roll(index), truth, list(scale)) for index in wrong_indices]
        entry = {
            "target": trial.target_key,
            "source": trial.source_key,
            "regime": trial.regime,
            "target_mechanism": trial.target_mechanism,
            "wrong_mechanisms": list(trial.wrong_mechanisms),
            # The control is the mean over the other laws, so its expected value
            # under "the code carries nothing" is the target's own error under a
            # randomly relabelled law rather than under one arbitrary label.
            "wrong": sum(wrong_errors) / len(wrong_errors),
            "self": self_error,
            "correct": correct_error,
        }
        records.append(entry)
        clusters.append(trial.target_key)
        targets_seen.append(trial.target_mechanism)
        for arm in ("self", "correct", "wrong"):
            errors[arm].append(entry[arm])

        # The sensitivity reading, per (factual, counterfactual) law pair. The
        # target's *own* law is the factual one -- that is the world it is in --
        # and the transplanted law is the counterfactual.
        for wrong_mechanism in trial.wrong_mechanisms:
            factual_index = part_index.get(f"{trial.target_key}|{trial.regime}|{wrong_mechanism}")
            source_factual = part_index.get(f"{trial.source_key}|{trial.regime}|{wrong_mechanism}")
            factual_row = row_index.get(f"{trial.target_key}|{trial.regime}|{wrong_mechanism}")
            if factual_index is None or source_factual is None or factual_row is None:
                continue
            factual_series = frame_series(
                GravityFrame(factual_row["frame_angle_rad"]),
                [factual_row["x0"], *factual_row["trajectory"]],
            )
            moved = [
                a - b
                for a, b in zip(
                    flatten(roll(correct_index)),
                    flatten(roll(source_factual)),
                    strict=True,
                )
            ]
            law_moved = [
                a - b
                for a, b in zip(
                    flatten([list(entry) for entry in series[1 : TRANSPLANT_HORIZON + 1]]),
                    flatten(
                        [
                            list(entry)
                            for entry in factual_series[1 : TRANSPLANT_HORIZON + 1]
                        ]
                    ),
                    strict=True,
                )
            ]
            # Both exclusions are counted rather than scored as zero, because a
            # zero would be a claim -- "the code moved the prediction
            # perpendicular to the law" -- where the truth is that the pair says
            # nothing. An inactive law (ADR 0005) leaves no counterfactual to
            # move toward; a code the decoder does not read leaves no move at
            # all. The two are different facts, one about the world and one about
            # the instrument, so they are counted apart.
            if _norm(law_moved) < 1e-9:
                inactive_law_pairs += 1
                continue
            if _norm(moved) < 1e-9:
                degenerate_pairs += 1
                continue
            sensitivity_rows.append(
                {
                    "target": trial.target_key,
                    "regime": trial.regime,
                    "factual": wrong_mechanism,
                    "counterfactual": trial.target_mechanism,
                    "alignment": cosine(moved, law_moved),
                    "cluster": trial.target_key,
                }
            )

    if not records:
        return {"summary": None, "reason": "no eligible transplant trials", "records": []}
    result = paired_transplant(errors, clusters, targets_seen, seed=decoder_config.seed)
    return {
        "summary": result.as_dict(),
        "sensitivity": _sensitivity_report(
            sensitivity_rows, decoder_config.seed, degenerate_pairs, inactive_law_pairs
        ),
        "records": records,
        "decoder_final_loss": fitted["final_loss"],
        "n_trials": len(records),
    }


def _fmt(record: dict | None, key: str) -> str:
    if not record or key not in record:
        return "n/a"
    return f"{record[key]:+.4f}"


def _ccgp_by_regime(
    fit_latents: list[tuple[float, ...]],
    fit_mechanisms: list[str],
    fit_rows: list[dict],
    spec: Any,
    test_latents: list[tuple[float, ...]],
    test_mechanisms: list[str],
    test_rows: list[dict],
    condition_partition: design.ConditionPartition,
    chance: dict[str, float],
) -> dict[str, Any]:
    """CCGP restricted to one intervention regime at a time.

    The plan's M5 and the H1/H2 comparisons read this: if relation identifiability
    is what makes abstraction possible, the endpoint should improve with
    intervention richness, and a headline that only holds when the three regimes
    are pooled has hidden that.

    Restricted to one regime, the condition is the appearance family alone, which
    is still the held-out axis -- so the readout is still asked to transfer across
    an unseen family, just within a fixed intervention strength.
    """
    # Each row contributes `spec.horizon` steps, so row `i` occupies the flat
    # range `[i * horizon, (i + 1) * horizon)` in the latent and mechanism lists.
    def select(rows: list[dict], latents: list[tuple[float, ...]], mechanisms: list[str], regime: str):
        chosen = [i for i, row in enumerate(rows) if row["regime"] == regime]
        index = [i * spec.horizon + offset for i in chosen for offset in range(spec.horizon)]
        return (
            [latents[i] for i in index],
            [mechanisms[i] for i in index],
            [f"{design.family_of_condition(design.condition_of(rows[i // spec.horizon]))}|{regime}" for i in index],
        )

    regimes = sorted({row["regime"] for row in [*fit_rows, *test_rows]})
    by_regime: dict[str, Any] = {}
    for regime in regimes:
        fit_selection = select(fit_rows, fit_latents, fit_mechanisms, regime)
        test_selection = select(test_rows, test_latents, test_mechanisms, regime)
        if not fit_selection[0] or not test_selection[0]:
            continue
        # The family holdout still applies within the regime; anything the two
        # sides share is trained on and not scored, so the disjointness the
        # evaluator asserts is satisfied rather than assumed.
        train_conditions = set(fit_selection[2]) - set(test_selection[2])
        if not train_conditions:
            by_regime[regime] = {"unavailable": "no training condition outside the test families"}
            continue
        result = ccgp(
            [*fit_selection[0], *test_selection[0]],
            [*fit_selection[1], *test_selection[1]],
            [*fit_selection[2], *test_selection[2]],
            train_conditions=train_conditions,
            test_conditions=set(test_selection[2]),
            chance_level=chance,
        )
        by_regime[regime] = result.as_dict()
    return by_regime


def _code_ablation(model: Any, part: dict[str, Any]) -> dict[str, Any]:
    """How much the arm's own prediction depends on the code.

    The control that makes the transplant null interpretable. ``pooled`` is zeroed
    at prediction time and the loss compared against the unablated one; a small
    change means the predictor routes around the bottleneck, a large one means it
    reads the code and the transplant's null is about *portability* rather than
    about use.
    """
    import torch

    with torch.no_grad():
        normal = model(part, steps=1, ablate_pooled=False)["prediction"]
        ablated = model(part, steps=1, ablate_pooled=True)["prediction"]
        normal_loss = float(torch.nn.functional.mse_loss(normal, part["future"][:, 0]))
        ablated_loss = float(torch.nn.functional.mse_loss(ablated, part["future"][:, 0]))
        # The move the code causes, in the same units the transplant measures.
        move = float((normal - ablated).abs().mean())
    return {
        "one_step_loss_with_code": normal_loss,
        "one_step_loss_without_code": ablated_loss,
        "loss_increase": ablated_loss - normal_loss,
        "relative_increase": (
            (ablated_loss - normal_loss) / normal_loss if normal_loss > 0.0 else None
        ),
        "mean_prediction_move": move,
    }


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _sensitivity_report(
    rows: list[dict], seed: int, degenerate: int = 0, inactive: int = 0
) -> dict[str, Any]:
    """Alignment against a derangement, overall and per regime.

    The derangement is drawn *within* a (regime, law pair) cell, so the control
    compares alignments of the same mechanism contrast at the same intervention
    strength and only breaks the pairing between a target's own scene and the code
    that was transplanted into it. A derangement across cells would compare
    contrasts of different sizes and turn the control into a statement about which
    laws differ most.
    """
    if not rows:
        return {
            "unavailable": "no scorable pairs",
            "pairs_with_an_inactive_law": inactive,
            "pairs_the_code_did_not_move": degenerate,
            "note": (
                "every pair was excluded before scoring; a pair is only scorable "
                "when the law changes the world's trajectory and the code changes "
                "the decoded one"
            ),
        }
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["regime"], row["factual"], row["counterfactual"]), []).append(row)

    # The control is drawn from a *different* law pair in the same regime, never
    # from a permutation inside one cell. Permuting within a cell leaves the mean
    # exactly where it was -- every element there is the same kind of pair -- so
    # the contrast would be identically zero and the control would say nothing.
    # Matching the regime keeps the scale of the two trajectories comparable.
    by_regime: dict[str, list[tuple[str, dict]]] = {}
    for key, cell in sorted(cells.items()):
        by_regime.setdefault(key[0], []).extend((key, row) for row in cell)

    paired: list[dict] = []
    rng = random.Random(seed)
    for key, cell in sorted(cells.items()):
        others = [row for other_key, row in by_regime[key[0]] if other_key != key]
        if not others:
            continue
        for row in cell:
            paired.append(
                {
                    "regime": row["regime"],
                    "alignment": row["alignment"],
                    "control": others[rng.randrange(len(others))]["alignment"],
                    "cluster": row["cluster"],
                }
            )
    if len({entry["cluster"] for entry in paired}) < 2:
        return {"unavailable": "too few clusters for a bootstrap"}

    def run(subset: list[dict]) -> dict[str, Any]:
        return sensitivity(
            [entry["alignment"] for entry in subset],
            [entry["control"] for entry in subset],
            [entry["cluster"] for entry in subset],
            seed=seed,
        ).as_dict()

    per_regime: dict[str, Any] = {}
    for regime in sorted({entry["regime"] for entry in paired}):
        subset = [entry for entry in paired if entry["regime"] == regime]
        if len({entry["cluster"] for entry in subset}) < 2:
            continue
        per_regime[regime] = run(subset)
    return {
        "overall": run(paired),
        "per_regime": per_regime,
        "n_pairs": len(paired),
        "excluded_inactive_law": inactive,
        "excluded_code_had_no_effect": degenerate,
        "n_derangements": {"|".join(key): len(value) for key, value in sorted(cells.items())},
    }


def _structural_tensor(row: dict) -> Any:
    import torch

    return torch.tensor([row["filler_structural"]], dtype=torch.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default=DEFAULT_CONFIG.world, choices=WORLDS)
    parser.add_argument("--groups-per-tuple", type=int, default=DEFAULT_CONFIG.groups_per_tuple)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--split", default="unseen_family", choices=SPLIT_KINDS)
    parser.add_argument("--arms", default="", help="comma-separated subset; default all")
    parser.add_argument("--group-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--code",
        default="learned",
        choices=("learned", "oracle"),
        help="'oracle' feeds the decoder a one-hot mechanism label: the must-fire control",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase1b")
    args = parser.parse_args(argv)

    from ..models.batch_stream import StreamSpec
    from ..models.representation import ARMS, ModelConfig, train_arm
    from ..models.transplant import DecoderConfig
    from ..simulator.dataset import build_rows, read_rows, write_rows

    config = Phase1AConfig(world=args.world, groups_per_tuple=args.groups_per_tuple)
    stamp = status_block(
        world=config.world,
        schema_version=config.schema_version,
        config_sha256=config_digest(config),
        generator_seed=world_seed(config.world),
        families=list(world_families(config.world)),
        git_sha=git_sha(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "config.json",
        {**stamp, "config": config.as_dict(), "args": {**vars(args), "out": str(args.out)}},
    )

    started = time.time()
    groups = build_groups(config)
    # Rebuilt every run rather than reused from a previous one. It is deterministic,
    # and a stale dataset silently paired with a new split is exactly the class of
    # accident this repository treats as its worst.
    write_rows(
        args.out / "dataset.jsonl", build_rows(groups, config.horizon, group_limit=args.group_limit)
    )
    rows = read_rows(args.out / "dataset.jsonl")
    print(f"[phase1b] world={config.world} contexts={len(groups)} rows={len(rows)}")

    split = build_split(groups, args.split, seed=DEFAULT_SPLIT_SEED)
    partition = design.partition_contexts(split.train, split.test, seed=DEFAULT_SPLIT_SEED + 1)
    conditions = design.family_condition_partition(rows, seed=DEFAULT_SPLIT_SEED)
    chance = design.chance_level_by_condition(rows)
    write_json(
        args.out / "splits.json",
        {
            **stamp,
            "kind": split.kind,
            "held_out": list(split.held_out),
            "contexts": {
                "fit": list(partition.fit),
                "validation": list(partition.validation),
                "test": list(partition.test),
            },
            "conditions": {"train": sorted(conditions.train), "test": sorted(conditions.test)},
        },
    )

    fit_rows = _rows_for(rows, partition.fit)
    val_rows = _rows_for(rows, partition.validation)
    test_rows = _rows_for(rows, partition.test)
    spec = StreamSpec(history=4, horizon=TRANSPLANT_HORIZON)
    scale = _frame_scale(fit_rows, TRANSPLANT_HORIZON)
    print(
        f"[phase1b] split={split.kind} held_out={list(split.held_out)} "
        f"fit={len(partition.fit)} val={len(partition.validation)} test={len(partition.test)} | "
        f"fit_rows={len(fit_rows)} test_rows={len(test_rows)}"
    )

    arms = list(ARMS) if not args.arms else [name.strip() for name in args.arms.split(",")]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {sorted(ARMS)}")

    from ..models.batch_stream import action_dim, build_part, struct_dim

    results: dict[str, Any] = {}
    transplants: dict[str, Any] = {}
    for arm in arms:
        arm_started = time.time()
        arm_spec = ARMS[arm]
        with_relation = arm_spec.uses_relation_label
        model_config = ModelConfig(latent=args.latent, epochs=args.epochs, seed=args.seed)
        record = train_arm(
            arm,
            {
                "train": build_part(
                    fit_rows,
                    spec,
                    with_relation=with_relation,
                    future_steps=model_config.rollout_steps,
                ),
                "val": build_part(
                    val_rows,
                    spec,
                    with_relation=with_relation,
                    future_steps=model_config.rollout_steps,
                ),
            },
            state_dim=8,
            struct_dim=struct_dim(),
            action_dim=action_dim(),
            config=model_config,
        )
        model = record["model"]
        endpoints = _evaluate_arm(
            model, spec, fit_rows, test_rows, conditions, chance
        )
        transplant = _transplant(
            model,
            DecoderConfig(seed=args.seed),
            rows,
            partition,
            fit_rows,
            test_rows,
            spec,
            scale,
            code=args.code,
        )
        results[arm] = {
            "n_parameters": record["n_parameters"],
            "val_loss": record["val_loss"],
            "best_epoch": record["best_epoch"],
            "wall_seconds": time.time() - arm_started,
            "endpoints": endpoints,
        }
        transplants[arm] = transplant
        summary = transplant["summary"]
        sens = (transplant.get("sensitivity") or {}).get("overall", {})
        print(
            f"[phase1b]   {arm}: params={record['n_parameters']} "
            f"val={record['val_loss']:.6f} best_epoch={record['best_epoch']} "
            f"ccgp={endpoints['ccgp_unseen_family']['above_baseline']:+.4f} "
            f"role_gain={endpoints['role_equivariance']['role_gain']:+.4f} "
            f"cferror={_fmt(summary, 'gain_wrong_minus_correct')} "
            f"align={_fmt(sens, 'mean_alignment')}"
            f"{' CONFOUNDED' if sens.get('confounded') else ''}"
        )

    write_json(args.out / "arms.json", {**stamp, "code": args.code, "arms": results})
    write_json(args.out / "transplant.json", {**stamp, "horizon": TRANSPLANT_HORIZON, "arms": transplants})
    write_json(
        args.out / "endpoints.json",
        {
            **stamp,
            "transplant_horizon": TRANSPLANT_HORIZON,
            "frozen_scale": list(scale),
            "arms": {
                arm: {
                    **{key: value for key, value in record.items() if key != "endpoints"},
                    **record["endpoints"],
                    "transplant": transplants[arm]["summary"],
                }
                for arm, record in results.items()
            },
        },
    )

    artifacts = {
        name: {"sha256": sha256_file(args.out / name), "bytes": (args.out / name).stat().st_size}
        for name in (
            "config.json",
            "dataset.jsonl",
            "splits.json",
            "arms.json",
            "endpoints.json",
            "transplant.json",
        )
    }
    write_json(
        args.out / "manifest.json",
        {**stamp, "artifacts": artifacts, "wall_seconds": time.time() - started},
    )
    print(f"[phase1b] done in {time.time() - started:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
