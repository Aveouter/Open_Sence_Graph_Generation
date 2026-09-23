"""Matched history-code transplantation through the predictor that learned it."""

from __future__ import annotations

import random

import torch

from ..eval.transplant import paired_transplant
from .model import select_batch


def matched_sources(rows, level):
    """Disjoint context dyads; all mechanisms/regimes stay in their base group.

    The source under correct/wrong M is always the same context. Each context
    participates in one dyad, avoiding treating reused sources as independent.
    Within-filler additionally fixes physical filler identity; cross-filler
    requires a different identity. Matching never uses prediction error.
    """
    if level not in ("within_filler", "cross_filler"):
        raise ValueError("unknown transplant level")
    contexts = {}
    for i, row in enumerate(rows):
        contexts.setdefault(row["group"], {})[(row["regime"], row["mechanism"])] = i
    remaining = sorted(contexts)
    pairs, unmatched = [], []
    while remaining:
        target_group = remaining.pop(0)
        target = rows[next(iter(contexts[target_group].values()))]
        candidate = next(
            (
                g
                for g in remaining
                if rows[next(iter(contexts[g].values()))]["tuple"] == target["tuple"]
                and (
                    (
                        rows[next(iter(contexts[g].values()))]["filler"]
                        == target["filler"]
                    )
                    == (level == "within_filler")
                )
            ),
            None,
        )
        if candidate is None:
            unmatched.append(target_group)
            continue
        remaining.remove(candidate)
        for (regime, mechanism), target_index in sorted(contexts[target_group].items()):
            correct = contexts[candidate].get((regime, mechanism))
            if correct is None:
                raise ValueError("matched source is missing the correct mechanism")
            for wrong_m in target["compatible"]:
                if wrong_m != mechanism:
                    wrong = contexts[candidate].get((regime, wrong_m))
                    if wrong is None:
                        raise ValueError(
                            "matched source is missing a counterfactual twin"
                        )
                    pairs.append(
                        (target_index, correct, wrong, f"{target_group}|{candidate}")
                    )
    return pairs, unmatched


def replace_query_code(model, target_code, source_code, target_row, source_row):
    if not model.pairwise:
        return source_code.clone()
    code = target_code.clone()
    target_edge, source_edge = tuple(target_row["query"]), tuple(source_row["query"])
    for target, source in (
        (target_edge, source_edge),
        (target_edge[::-1], source_edge[::-1]),
    ):
        code[:, model.edges.index(target)] = source_code[:, model.edges.index(source)]
    return code


def score_predictions(
    errors, clusters, sensitivities, exposures, *, unpaired_groups, seed=0
):
    """Zero-sensitivity pairs remain in the primary statistic, never filtered."""
    count = len(sensitivities)
    result = {
        "pairs": count,
        "unpaired_groups": unpaired_groups,
        "off_manifold_exclusions": 0,
        "off_manifold_status": "not inferred from prediction sensitivity",
        "unscorable_pair_rate": sum(s <= 1e-8 for s in sensitivities) / count
        if count
        else None,
        "mean_prediction_sensitivity": sum(sensitivities) / count if count else None,
        "exposed_pairs": sum(exposures),
        "unexposed_pairs": count - sum(exposures),
        "cluster_unit": "disjoint source-target base-context dyad",
    }
    if len(set(clusters)) < 2:
        return {**result, "status": "INSUFFICIENT_CONTEXTS"}
    primary = paired_transplant(errors, clusters, seed=seed).as_dict()
    result.update(status="SCORED", primary=primary)
    subset = [i for i, exposed in enumerate(exposures) if exposed]
    if len({clusters[i] for i in subset}) >= 2:
        result["exposure_conditioned"] = paired_transplant(
            {arm: [values[i] for i in subset] for arm, values in errors.items()},
            [clusters[i] for i in subset],
            seed=seed,
        ).as_dict()
    return result


def evaluate_transplant(model, rows, batch, targets, *, level, scale, seed=0):
    if torch.any(scale <= 0) or not torch.isfinite(scale).all():
        raise ValueError(
            "transplant scale must be positive and frozen on training contexts"
        )
    pairs, unmatched = matched_sources(rows, level)
    model.eval()
    errors = {"self": [], "correct": [], "wrong": []}
    shuffled_errors, clusters, sensitivity, exposure = [], [], [], []
    rng = random.Random(seed)
    with torch.no_grad():
        codes = model.infer_mechanism(batch, torch.Generator().manual_seed(seed))
        for target, correct, wrong, cluster in pairs:
            context = select_batch(batch, [target])
            predicted = {}
            # Shuffling is a mixture of the same source context's correct/wrong
            # mechanisms, not a code drawn from an unmatched context.
            shuffled = rng.choice((correct, wrong))
            for name, source in (
                ("self", target),
                ("correct", correct),
                ("wrong", wrong),
                ("shuffled", shuffled),
            ):
                code = replace_query_code(
                    model,
                    codes[target : target + 1],
                    codes[source : source + 1],
                    rows[target],
                    rows[source],
                )
                prediction = model(context, mechanism=code)
                node = rows[target]["query"][0]
                predicted[name] = prediction[:, :, node]
                error = float(
                    (
                        (
                            (predicted[name] - targets[target : target + 1, :, node])
                            / scale
                        )
                        ** 2
                    )
                    .mean()
                    .sqrt()
                )
                if name == "shuffled":
                    shuffled_errors.append(error)
                else:
                    errors[name].append(error)
            sensitivity.append(
                float(
                    (predicted["wrong"] - predicted["correct"]).square().mean().sqrt()
                )
            )
            exposure.append(
                bool(
                    rows[target]["mechanism_exposed"]
                    and rows[correct]["mechanism_exposed"]
                    and rows[wrong]["mechanism_exposed"]
                )
            )
            clusters.append(cluster)
    result = score_predictions(
        errors, clusters, sensitivity, exposure, unpaired_groups=unmatched, seed=seed
    )
    if shuffled_errors:
        result["shuffled_minus_correct"] = sum(
            w - c for w, c in zip(shuffled_errors, errors["correct"], strict=True)
        ) / len(shuffled_errors)
    return result
