"""M0 leakage audit: can ``M`` be read off the context instead of the dynamics?

Four probes, each a different claim about what a shortcut could use:

``s0_only``
    the joint tuple alone, one-hot. This is the floor: if the sampler balances
    mechanisms within a tuple, a tuple-aware constant predictor is right
    ``1 / |compatible|`` of the time and no more.
``x0_only``
    the continuous initial state. Catches a sampler that places objects
    differently depending on which law they will obey -- the failure mode that
    would make every downstream measurement circular.
``filler_only``
    the nuisance attributes. Catches appearance entangled with mechanism.
``x0_plus_filler``
    both, because a leak can be split across two weak channels.

The bar is the **compatibility-aware baseline**, not a fixed chance level: a
predictor that knows the tuple and guesses uniformly among that tuple's
compatible mechanisms is right ``1 / |compatible|`` of the time, and that varies
by tuple (a quarter for four-compatible tuples, a third for three). Comparing
against a flat 25% or 33% would be wrong for some rows by construction.

Two estimators are run on every feature set, and the verdict uses the stronger:
a ridge readout on the raw features, and the same readout on a frozen random
``tanh`` hidden layer. The second is what ADR 0007 asks for -- a detector of the
same *family* as the oracle rather than a linear one -- because the two failure
modes are not symmetric. A probe that is too weak reports "no leak" for a sampler
that leaks nonlinearly, and that false negative is indistinguishable in a report
from a genuinely clean result. Both are closed-form solves, so the whole audit
stays inside the dependency-free CI job.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..simulator.compatibility import compatible_mechanisms
from ..simulator.counterfactuals import CounterfactualGroup
from ..simulator.factors import enumerate_realizable

FEATURE_SETS: tuple[str, ...] = ("s0_only", "x0_only", "filler_only", "x0_plus_filler")

RIDGE_LAMBDA = 1e-2


@dataclass(frozen=True)
class LeakageRecord:
    tuple_key: str
    mechanism: str
    x0_vector: tuple[float, ...]
    filler_vector: tuple[float, ...]


def build_records(groups: list[CounterfactualGroup]) -> list[LeakageRecord]:
    """One row per (group, mechanism). The mechanism is the label to be predicted."""
    records: list[LeakageRecord] = []
    for group in groups:
        for mechanism in group.mechanisms:
            records.append(
                LeakageRecord(
                    tuple_key=group.s0.key(),
                    mechanism=mechanism,
                    x0_vector=group.x0.outcome_vector(),
                    filler_vector=group.filler.nuisance.feature_vector(),
                )
            )
    return records


def feature_vector(record: LeakageRecord, feature_set: str) -> tuple[float, ...]:
    tuple_one_hot = tuple(
        1.0 if record.tuple_key == s0.key() else 0.0 for s0 in enumerate_realizable()
    )
    if feature_set == "s0_only":
        return tuple_one_hot
    if feature_set == "x0_only":
        return record.x0_vector
    if feature_set == "filler_only":
        return record.filler_vector
    if feature_set == "x0_plus_filler":
        return record.x0_vector + record.filler_vector
    raise ValueError(f"unknown feature set {feature_set!r}, not in {FEATURE_SETS!r}")


def analytic_baseline(records: list[LeakageRecord]) -> float:
    """Expected accuracy of a tuple-aware predictor that ignores everything else."""
    if not records:
        raise ValueError("baseline needs at least one record")
    lookup = {s0.key(): len(compatible_mechanisms(s0)) for s0 in enumerate_realizable()}
    return sum(1.0 / lookup[record.tuple_key] for record in records) / len(records)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting, for a small dense system."""
    size = len(matrix)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * other
                for value, other in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][size] for index in range(size)]


def fit_linear_probe(
    features: list[tuple[float, ...]], labels: list[int], n_classes: int, ridge: float = RIDGE_LAMBDA
) -> list[list[float]]:
    """Ridge regression onto one-hot targets, in closed form.

    Equivalent to a linear classifier trained with a squared loss. Chosen over
    gradient descent so the permutation test below stays cheap enough to run on
    every audit rather than only when someone remembers to.
    """
    if not features:
        raise ValueError("cannot fit a probe on an empty design matrix")
    width = len(features[0]) + 1
    gram = [[0.0] * width for _ in range(width)]
    cross = [[0.0] * n_classes for _ in range(width)]
    for feature, label in zip(features, labels, strict=True):
        row = (*feature, 1.0)
        for a in range(width):
            if row[a] == 0.0:
                continue
            for b in range(width):
                gram[a][b] += row[a] * row[b]
            cross[a][label] += row[a]
    for index in range(width - 1):
        gram[index][index] += ridge
    weights: list[list[float]] = [[0.0] * n_classes for _ in range(width)]
    for klass in range(n_classes):
        column = _solve([row[:] for row in gram], [cross[a][klass] for a in range(width)])
        for a in range(width):
            weights[a][klass] = column[a]
    return weights


def predict_probe(weights: list[list[float]], feature: tuple[float, ...]) -> int:
    row = (*feature, 1.0)
    scores = [
        sum(row[a] * weights[a][klass] for a in range(len(row)))
        for klass in range(len(weights[0]))
    ]
    return max(range(len(scores)), key=lambda klass: scores[klass])


RANDOM_FEATURES = 64
RANDOM_FEATURE_SCALE = 1.0


def random_feature_map(
    features: list[tuple[float, ...]], rng: random.Random, width: int = RANDOM_FEATURES
) -> list[tuple[float, ...]]:
    """A frozen random hidden layer: ``tanh(W x + b)``.

    ADR 0007 asks for a detector of the same family as the oracle rather than a
    linear one, and the reason is asymmetric failure: a probe that is too weak
    reports "no leak" for a sampler that leaks nonlinearly, and that false
    negative is indistinguishable in a report from a clean result. A linear probe
    is exactly that weak probe.

    This is an MLP whose first layer is drawn rather than trained -- random
    features -- so the readout stays a closed-form ridge solve and the whole
    audit remains stdlib and CI-runnable. It is a strictly stronger test than the
    linear probe on the same inputs, and it is still not a *trained* MLP, which
    the torch side would be. Both are reported, and the verdict uses the
    stronger one.
    """
    if not features:
        raise ValueError("cannot expand an empty design matrix")
    dim = len(features[0])
    weights = [
        tuple(rng.gauss(0.0, RANDOM_FEATURE_SCALE) for _ in range(dim)) for _ in range(width)
    ]
    biases = [rng.uniform(-1.0, 1.0) for _ in range(width)]
    expanded: list[tuple[float, ...]] = []
    for row in features:
        expanded.append(
            tuple(
                math.tanh(sum(w * x for w, x in zip(weight, row, strict=True)) + b)
                for weight, b in zip(weights, biases, strict=True)
            )
        )
    return expanded


def standardize(
    features: list[tuple[float, ...]],
) -> tuple[list[tuple[float, ...]], tuple[float, ...], tuple[float, ...]]:
    """Centre and scale each dimension by its standard deviation, floored."""
    width = len(features[0])
    count = len(features)
    means = tuple(sum(row[d] for row in features) / count for d in range(width))
    scales = []
    for d in range(width):
        variance = sum((row[d] - means[d]) ** 2 for row in features) / count
        scales.append(max(variance**0.5, 1e-9))
    standardized = [
        tuple((row[d] - means[d]) / scales[d] for d in range(width)) for row in features
    ]
    return standardized, means, tuple(scales)


def apply_standard(
    features: list[tuple[float, ...]], means: tuple[float, ...], scales: tuple[float, ...]
) -> list[tuple[float, ...]]:
    """Apply the *training* set's centring and scaling to another set.

    Used instead of a second :func:`standardize` call wherever the second set is
    a held-out one. Standardising a test set by its own statistics is transductive
    adaptation: the readout then sees test-time means and variances it could not
    have known at fit time, and on a condition-generalization endpoint -- where
    the whole question is whether the code transfers to conditions the readout
    never saw -- that inflates the number along exactly the axis being measured.
    """
    return [
        tuple((row[d] - means[d]) / scales[d] for d in range(len(means))) for row in features
    ]


def evaluate_feature_set(
    records: list[LeakageRecord],
    feature_set: str,
    n_permutations: int = 40,
    seed: int = 0,
    test_fraction: float = 0.5,
    probe: str = "linear",
) -> dict:
    """Group-blocked split-half accuracy against the analytic baseline.

    The split is blocked on the tuple so that every fold sees every tuple, and
    the permutation test re-draws mechanism labels *within each tuple* -- the
    same structure a shortcut would have to exploit, and therefore the only null
    that isolates "reads the context" from "knows the tuple frequencies".
    """
    mechanisms = sorted({record.mechanism for record in records})
    index_of = {mechanism: position for position, mechanism in enumerate(mechanisms)}
    raw_features = [feature_vector(record, feature_set) for record in records]
    features, _, _ = standardize(raw_features)
    if probe == "random_feature":
        # One map, drawn once, shared by the fit and every permutation: redrawing
        # per permutation would make the null include the variation of the
        # feature map rather than only the label shuffle.
        features = random_feature_map(features, random.Random(seed))
    elif probe != "linear":
        raise ValueError(f"unknown probe {probe!r}; expected 'linear' or 'random_feature'")
    labels = [index_of[record.mechanism] for record in records]

    rng = random.Random(seed)
    by_tuple: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        by_tuple.setdefault(record.tuple_key, []).append(position)
    train_rows: list[int] = []
    test_rows: list[int] = []
    for positions in by_tuple.values():
        shuffled = positions[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * test_fraction))
        test_rows.extend(shuffled[:cut])
        train_rows.extend(shuffled[cut:])

    def accuracy(train: list[int], test: list[int], target: list[int]) -> float:
        weights = fit_linear_probe([features[i] for i in train], [target[i] for i in train], len(mechanisms))
        hits = sum(1 for i in test if predict_probe(weights, features[i]) == target[i])
        return hits / len(test)

    observed = accuracy(train_rows, test_rows, labels)

    null_scores: list[float] = []
    for _ in range(n_permutations):
        permuted = labels[:]
        for positions in by_tuple.values():
            values = [labels[i] for i in positions]
            rng.shuffle(values)
            for i, value in zip(positions, values, strict=True):
                permuted[i] = value
        null_scores.append(accuracy(train_rows, test_rows, permuted))
    null_scores.sort()
    exceed = sum(1 for score in null_scores if score >= observed)
    p_value = (exceed + 1) / (len(null_scores) + 1)

    baseline = analytic_baseline(records)
    return {
        "probe": probe,
        "feature_set": feature_set,
        "n_records": len(records),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "accuracy": observed,
        "compatibility_aware_baseline": baseline,
        "minimum_detectable_effect": minimum_detectable_effect(len(test_rows), baseline),
        "permutation_p_value": p_value,
        "null_mean": sum(null_scores) / len(null_scores),
        "null_max": null_scores[-1],
        "null_quantile_95": null_scores[min(len(null_scores) - 1, int(0.95 * len(null_scores)))],
    }


def minimum_detectable_effect(
    n_test: int, baseline: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest accuracy gap this probe could have caught, in accuracy points.

    "Not significantly above baseline" and "underpowered" look identical in a
    results table, and the difference matters: the first is a finding about the
    sampler, the second is a finding about the experiment. Reporting the minimum
    detectable effect alongside the verdict is what keeps them apart, and ADR 0007
    requires effects below it to be described as undetectable rather than absent.

    Normal approximation, one-sided, on a binomial proportion.
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must be a proportion strictly between 0 and 1")
    z_alpha = 1.6448536269514722 if alpha == 0.05 else _norm_ppf(1.0 - alpha)
    z_power = 0.8416212335729143 if power == 0.8 else _norm_ppf(power)
    return (z_alpha + z_power) * (baseline * (1.0 - baseline) / max(1, n_test)) ** 0.5


def normal_quantile(probability: float) -> float:
    """The standard normal quantile, public so other pure-stdlib probes can use it."""
    return _norm_ppf(probability)


def _norm_ppf(probability: float) -> float:
    """Acklam's rational approximation, so this module stays stdlib-only."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1.0 - 0.02425
    if probability < low:
        q = (-2.0 * math.log(probability)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability > high:
        q = (-2.0 * math.log(1.0 - probability)) ** 0.5
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def leaky_sampler_control(groups: list[CounterfactualGroup]) -> list[LeakageRecord]:
    """Must-fire control: a context deliberately made to reveal its mechanism.

    Shifts the recorded ``X_0`` by an amount that depends on the mechanism, which
    is exactly the leak ``x0_only`` exists to catch. If the probe cannot detect
    *this*, it is too weak to certify anything, and the audit's clean result on
    the real sampler would mean nothing.
    """
    records = build_records(groups)
    offset = {mechanism: float(index) for index, mechanism in enumerate(sorted({r.mechanism for r in records}))}
    return [
        LeakageRecord(
            tuple_key=record.tuple_key,
            mechanism=record.mechanism,
            x0_vector=tuple(
                value + (offset[record.mechanism] if index == 0 else 0.0)
                for index, value in enumerate(record.x0_vector)
            ),
            filler_vector=record.filler_vector,
        )
        for record in records
    ]
