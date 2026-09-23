"""Development checks that cannot be replaced by a v1 verdict."""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn

from ..eval.endpoints import ccgp, role_equivariance
from .data import build_scenes, split_groups
from .model import shuffle_frames
from .protocol import MECHANISMS
from .transplant import score_predictions


def world_audit(rows, protocol):
    variants = build_scenes(protocol, numerical_variant=1)
    noise = build_scenes(protocol, noise_offset=7919)
    states = torch.tensor([r["states"] for r in rows], dtype=torch.float64)
    numerical = torch.tensor([r["states"] for r in variants], dtype=torch.float64)
    null = torch.tensor([r["states"] for r in noise], dtype=torch.float64)
    flat = states.flatten(2)
    scale = flat.flatten(0, 1).std(0).clamp_min(0.01)
    numeric_distance = ((states - numerical).flatten(2) / scale).square().mean(2).sqrt()
    null_distance = ((states - null).flatten(2) / scale).square().mean(2).sqrt()
    floor = float(null_distance[:, 1:].quantile(0.95))
    cells = {}
    for i, row in enumerate(rows):
        cells.setdefault((row["group"], row["regime"]), []).append(i)
    curves = {}
    compatible = True
    for (_, regime), indices in cells.items():
        first = rows[indices[0]]
        compatible &= (
            len(indices) >= 3
            and "free" in first["compatible"]
            and {rows[i]["mechanism"] for i in indices} == set(first["compatible"])
        )
        for i in indices:
            compatible &= (
                rows[i]["states"][0] == first["states"][0]
                and rows[i]["actions"] == first["actions"]
                and rows[i]["structural"] == first["structural"]
            )
        for a, b in itertools.combinations(indices, 2):
            key = f"{first['tuple']}|{regime}|{rows[a]['mechanism']}:{rows[b]['mechanism']}"
            distance = ((flat[a, 1:] - flat[b, 1:]) / scale).square().mean(1).sqrt()
            curves.setdefault(key, []).append(
                (distance, rows[a]["mechanism_exposed"] or rows[b]["mechanism_exposed"])
            )
    reports = {}
    for key, values in curves.items():
        mean = torch.stack([v[0] for v in values]).mean(0)
        exposed = [v[0] for v in values if v[1]]
        crossing = (mean > floor).nonzero().flatten()
        saturated = (mean >= 0.9 * mean.max()).nonzero().flatten()
        reports[key] = {
            "curve": mean.tolist(),
            "auc": float(mean.mean()),
            "time_to_divergence": int(crossing[0]) + 1 if len(crossing) else None,
            "saturation_horizon": int(saturated[0]) + 1 if len(saturated) else None,
            "exposed_contexts": len(exposed),
            "exposure_conditioned_auc": float(torch.stack(exposed).mean())
            if exposed
            else None,
        }
    rich = [value for key, value in reports.items() if "|rich|" in key]
    selected = max(v["saturation_horizon"] or protocol.horizon for v in rich)
    # Every compatible tuple/pair must clear its measured self-divergence floor.
    separable = all(v["auc"] > floor for v in rich)
    numeric_ok = float(numeric_distance.max()) <= protocol.numerical_tolerance
    horizon_ok = selected <= min(protocol.cutoff, protocol.prediction_steps)
    return {
        "compatibility": bool(compatible),
        "numerical_max": float(numeric_distance.max()),
        "numerical_mean": float(numeric_distance.mean()),
        "null_floor_p95": floor,
        "selected_horizon": selected,
        "horizon_covered": horizon_ok,
        "rich_cells_separable": separable,
        "curves": reports,
        "status": "PASS"
        if compatible and numeric_ok and separable and horizon_ok
        else "STOP_DATA",
        "scale": scale.tolist(),
        "objects": protocol.objects,
        "topology": "shared exogenous supporter; probes do not interact with each other",
    }


def chronology_audit(rows, batch, *, epochs=80, seed=47):
    """Can a trained classifier recover true frame age on held-out base groups?

    This checks content as well as the slot permutation. Above-chance chronology
    is a failed control, even if the random permutation itself is uniform.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    shuffled, truth = shuffle_frames(batch["frames"], generator)
    split = split_groups(rows, seed)
    ntime = shuffled.shape[1]
    x = shuffled.flatten(2)
    mean, std = (
        x[split["train"]].flatten(0, 1).mean(0),
        x[split["train"]].flatten(0, 1).std(0).clamp_min(1e-6),
    )
    x = (x - mean) / std
    model = nn.Sequential(nn.Linear(x.shape[-1], 32), nn.Tanh(), nn.Linear(32, ntime))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[split["train"]]).flatten(0, 1)
        loss = nn.functional.cross_entropy(logits, truth[split["train"]].flatten())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (
            (model(x[split["test"]]).argmax(-1) == truth[split["test"]]).float().mean(1)
        )
    by_group = {}
    for i, value in zip(split["test"], accuracy.tolist(), strict=True):
        by_group.setdefault(rows[i]["group"], []).append(value)
    values = [sum(v) / len(v) for v in by_group.values()]
    mean_accuracy = sum(values) / len(values)
    se = (
        sum((v - mean_accuracy) ** 2 for v in values)
        / max(1, len(values) - 1)
        / len(values)
    ) ** 0.5
    chance = 1 / ntime
    return {
        "accuracy": mean_accuracy,
        "chance": chance,
        "group_se": se,
        "groups": len(values),
        "status": "FAIL_RECOVERABLE_CHRONOLOGY"
        if mean_accuracy - 1.96 * se > chance
        else "INCONCLUSIVE",
        "note": "Failure blocks temporal teaching claims; a nonsignificant test alone is not a power-calibrated PASS.",
    }


def evaluator_controls():
    """Known-answer sensitivity/specificity fixtures, not real-data endpoint scores."""
    labels = list(MECHANISMS) * 32
    conditions = [f"family{(i // 4) % 4}" for i in range(len(labels))]
    train, test = {"family0", "family1"}, {"family2", "family3"}
    baseline = {c: 0.25 for c in conditions}
    carrying = [tuple(float(m == label) for m in MECHANISMS) for label in labels]
    nuisance = [tuple(float(c == f"family{k}") for k in range(4)) for c in conditions]
    positive = ccgp(carrying, labels, conditions, train, test, baseline)
    negative = ccgp(nuisance, labels, conditions, train, test, baseline)
    forward = [
        (math.sin(i), math.cos(i), math.sin(i * 0.7)) for i in range(len(labels))
    ]
    reverse = [(-a, b, -c) for a, b, c in forward]
    role = role_equivariance(forward, reverse, conditions, train, test)
    symmetric = role_equivariance(forward, forward, conditions, train, test)
    clusters = [f"context{i}" for i in range(24)]
    # These trajectories come from a deterministic residual fixture driven by
    # a one-hot law. Aliasing that code makes both counterfactual predictions equal.
    errors = {"self": [], "correct": [], "wrong": []}
    for index in range(len(clusters)):
        mechanism = index % 4
        truth = [(t + 1) * (mechanism + 1) for t in range(5)]
        wrong = [(t + 1) * ((mechanism + 1) % 4 + 1) for t in range(5)]
        errors["self"].append(0.0)
        errors["correct"].append(0.0)
        errors["wrong"].append(
            sum((a - b) ** 2 for a, b in zip(truth, wrong, strict=True)) ** 0.5
        )
    transplant = score_predictions(
        errors, clusters, errors["wrong"], [True] * 24, unpaired_groups=[]
    )
    alias = score_predictions(
        {k: [1.0] * 24 for k in errors},
        clusters,
        [0.0] * 24,
        [True] * 24,
        unpaired_groups=[],
    )
    passed = (
        positive.accuracy == 1
        and negative.accuracy <= 0.25
        and role.e_role < 1e-6
        and role.e_inv < 1e-6
        and role.e_role_shuffled > role.e_role
        and symmetric.e_inv < 1e-6
        and transplant["primary"]["ci_low"] > 0
        and alias["primary"]["gain_wrong_minus_correct"] == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "scope": "synthetic evaluator fixtures only",
        "ccgp_positive": positive.as_dict(),
        "ccgp_nuisance_negative": negative.as_dict(),
        "role": role.as_dict(),
        "symmetric_role": symmetric.as_dict(),
        "transplant_one_hot": transplant,
        "transplant_alias": alias,
        "real_decoder_calibration": "PENDING",
    }
