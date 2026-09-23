"""The third primary endpoint: matched latent transplantation.

The claim this endpoint exists to test is causal rather than decodable. A probe
shows that ``M`` is *readable* from ``z``; CCGP shows the reading transfers to
unseen conditions; neither shows that the predictor *uses* ``z`` the way the
world uses ``M``. Feeding the wrong latent and watching the prediction move to
the counterfactual future is what closes that gap.

Three arms per evaluated context, and the contrast between the last two is the
measurement:

``self``
    the latent from this context's own episode under the target mechanism. Not a
    causal claim -- this is the ordinary prediction the decoder was trained for,
    and it is here as a ceiling so a gain can be read against a scale.
``correct``
    the latent from a **different** context under the *target* mechanism.
``wrong``
    the latent from that same different context under a *different* mechanism --
    the source context is held fixed and only the mechanism changes.

The gain is ``mean(wrong) - mean(correct)``. Because the two arms differ in
nothing but which mechanism the transplanted code came from, a positive gain is
evidence that the code carries mechanism, and that this survives the change of
filler. A zero gain means the decoder ignores ``z``, or reads from it something
that does not transfer, and either way the causal claim is not supported.

The arithmetic is pure stdlib and lives here; the decoder that produces the
predictions is torch-side, because a decoder is a learned object. Keeping the
scoring separate is what lets it be unit-tested against trajectories with a
known answer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..audits.leakage import normal_quantile

ARMS: tuple[str, ...] = ("self", "correct", "wrong")


MAD_TO_SIGMA = 1.4826
RELATIVE_FLOOR = 1e-2
ABSOLUTE_FLOOR = 1e-9


def _median(sorted_values: list[float]) -> float:
    count = len(sorted_values)
    if count % 2 == 1:
        return sorted_values[count // 2]
    return 0.5 * (sorted_values[count // 2 - 1] + sorted_values[count // 2])


def trajectory_scale(
    vectors: list[tuple[float, ...]],
    relative_floor: float = RELATIVE_FLOOR,
    absolute_floor: float = ABSOLUTE_FLOOR,
) -> tuple[float, ...]:
    """Per-dimension ``1.4826 * MAD``, floored against the *widest* dimension.

    The same estimator the identifiability audit uses, with one change that
    matters once the trajectories being compared are decoded rather than
    simulated. That audit floors each dimension at a fraction of its *own* spread,
    which is right when every dimension moves. Here one of them does not: the
    supporter's velocity along the gravity axis is held near zero by a mass an
    order of magnitude larger than the probe's, so its MAD and its range are both
    dust, and a self-relative floor is dust too. Dividing by it weights that one
    coordinate a hundred thousand times more heavily than the rest, every decoded
    rollout scores as maximally wrong on it, and the arm contrast is decided by
    floating point rather than by the mechanism.

    Flooring against the widest dimension's spread instead keeps the weighting a
    property of the scene. A coordinate that genuinely does not move is then
    treated as uninformative rather than as infinitely informative, which is what
    it is.
    """
    if not vectors:
        raise ValueError("a trajectory scale needs at least one vector")
    width = len(vectors[0])
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("state vectors must share a width")

    spread: list[float] = []
    robust: list[float] = []
    for index in range(width):
        column = sorted(vector[index] for vector in vectors)
        median = _median(column)
        robust.append(MAD_TO_SIGMA * _median(sorted(abs(value - median) for value in column)))
        spread.append(column[-1] - column[0])
    floor = max(absolute_floor, relative_floor * max(spread))
    return tuple(max(value, floor) for value in robust)


DEVIATION_CAP = 5.0
"""Per-step, per-dimension deviations are clipped here, in scale units.

An autoregressive rollout of a contact law diverges: a decoder with no constraint
in it is off by a hair at step one and by hundreds of scale units at step ten, and
the raw root-mean-square then reports the *size of the divergence* rather than
whether the prediction followed the right law. Two arms that both diverged would
be ordered by floating-point luck. Clipping at a few scale units makes every
diverged rollout score as "maximally wrong", which is what it is, and leaves the
contrast to be decided by the steps before divergence -- the ones that carry the
mechanism's signature.

Five units is not a free parameter: it is far outside the spread of any real
trajectory's own deviation from its twins (Phase IA's separability curve saturates
well before it) and far inside the range a diverged rollout reaches, so the
clipping changes nothing except for rollouts that have already stopped being
about the mechanism.
"""


def trajectory_error(
    predicted: list[list[float]],
    truth: list[list[float]],
    scale: list[float] | tuple[float, ...],
    cap: float | None = DEVIATION_CAP,
) -> float:
    """Root-mean-square deviation, per dimension-scaled and per-step averaged.

    Divided by the frozen per-dimension scale rather than by the trajectory's own
    spread. Normalising by the sample would make every error look like O(1) --
    including the error of a decoder that output the mean of everything -- and the
    comparisons here are between arms of the *same* trajectory, so a common scale
    is what makes them comparable at all.

    ``cap`` is applied to each deviation before squaring; see
    :data:`DEVIATION_CAP`. Pass ``None`` for the unclipped error, which is the
    right thing when the trajectories are known not to have diverged -- a
    well-converged one-step comparison, say.
    """
    if len(predicted) != len(truth):
        raise ValueError(
            f"predicted and truth differ in length: {len(predicted)} vs {len(truth)}"
        )
    if not predicted:
        raise ValueError("cannot score an empty trajectory")
    if len(scale) != len(predicted[0]):
        raise ValueError("the scale does not match the state width")
    if any(entry <= 0.0 for entry in scale):
        raise ValueError("every scale entry must be positive")
    if cap is not None and cap <= 0.0:
        raise ValueError("the cap must be positive, or None")
    total = 0.0
    for predicted_state, true_state in zip(predicted, truth, strict=True):
        if len(predicted_state) != len(true_state):
            raise ValueError("a step disagrees on the state width")
        for a, b, s in zip(predicted_state, true_state, scale, strict=True):
            deviation = abs(a - b) / s
            if cap is not None:
                deviation = min(deviation, cap)
            total += deviation * deviation
    return math.sqrt(total / (len(predicted) * len(scale)))


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine of the angle between two flattened trajectories.

    Used for the *sensitivity* reading rather than the error one. A decoder is an
    imperfect instrument: its absolute error is large for reasons that have
    nothing to do with the code, and both arms of a transplant share that error,
    so a difference of errors can be swamped by it. The difference of two
    predictions under two codes has the shared error cancel out of it, and the
    question "did swapping the code move the prediction the way the law moves the
    world" survives a decoder that is merely directionally right.
    """
    if len(left) != len(right):
        raise ValueError("the vectors differ in length")
    if not left:
        raise ValueError("cannot take a cosine of nothing")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("a zero vector has no direction")
    return dot / (left_norm * right_norm)


def flatten(series: list[list[float]]) -> list[float]:
    return [value for state in series for value in state]


@dataclass(frozen=True)
class SensitivityResult:
    """Whether the code moves the prediction in the direction the law moves the world."""

    mean_alignment: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_clusters: int
    mde: float
    mean_control_alignment: float

    @property
    def positive(self) -> bool:
        """Whether the alignment is distinguishable from zero."""
        return self.ci_low > 0.0

    @property
    def confounded(self) -> bool:
        """Whether the control is as aligned as the real pairing.

        True means the statistic cannot separate a matched code from an
        arbitrary one, and nothing may be read from ``mean_alignment``.
        """
        return self.ci_low <= self.mean_control_alignment <= self.ci_high

    def as_dict(self) -> dict:
        return {
            "mean_alignment": self.mean_alignment,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pairs": self.n_pairs,
            "n_clusters": self.n_clusters,
            "mde": self.mde,
            "mean_control_alignment": self.mean_control_alignment,
            "positive": self.positive,
            "confounded": self.confounded,
            "note": (
                "the control is as aligned as the pairing, so this statistic does "
                "not separate a matched code from an arbitrary one; read CFError"
                if self.confounded
                else "the control is below the pairing's interval"
            ),
        }


def sensitivity(
    alignment: list[float],
    control: list[float],
    clusters: list[str],
    seed: int = 0,
    resamples: int = 2000,
    alpha: float = 0.05,
    power: float = 0.8,
) -> SensitivityResult:
    """Aggregate the paired alignment against a control of the same shape.

    **The control must be drawn from a different law pair, not a permutation
    within one.** A derangement inside a single ``(regime, factual,
    counterfactual)`` cell permutes the alignment values among themselves, so its
    mean is *identical* to the real pairing's by construction and the contrast is
    identically zero -- an earlier version of this docstring claimed the
    derangement answered "would any pair of codes produce this alignment", and it
    could not, because every element of the cell is the same kind of pair. The
    caller supplies a control drawn across cells, which is a null the statistic
    can actually reject.

    **The alignment is tested against zero, not against the control.** Measured
    on the oracle code, the control's mean came out at 0.5901 against the real
    pairing's 0.5918 -- because both differences are dominated by the same large
    direction (gravity, and the action schedule), so *any* two of them are
    positively aligned. A statistic whose control cannot differ from its subject
    cannot reject anything, and reporting `alignment − control` would have hidden
    that behind a number that is always about zero.

    What the control is good for is the opposite: it is kept as a **confound
    check**, and `confounded` is true when its mean falls inside the real
    pairing's interval. When it does, this statistic carries no information about
    the code and the endpoint's evidence must rest on `CFError` instead. That is
    the case here, and the report says so rather than quoting the alignment as
    though it were a result.

    The interval is a bootstrap over contexts. The unit of independent evidence is
    the context: the rows inside one share their ``X_0``, their action schedule
    and their filler, so resampling them independently would report an interval
    narrower than the evidence supports.
    """
    if not (len(alignment) == len(control) == len(clusters)):
        raise ValueError("alignment, control and clusters must be aligned")
    if not alignment:
        raise ValueError("cannot aggregate an empty sensitivity set")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(alignment, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    keys = sorted(grouped)
    if len(keys) < 2:
        raise ValueError("a cluster bootstrap needs at least two clusters")
    per_cluster = [sum(grouped[key]) / len(grouped[key]) for key in keys]
    mean = sum(per_cluster) / len(per_cluster)
    variance = sum((value - mean) ** 2 for value in per_cluster) / (len(per_cluster) - 1)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(len(per_cluster)):
            total += per_cluster[rng.randrange(len(per_cluster))]
        draws.append(total / len(per_cluster))
    draws.sort()
    return SensitivityResult(
        mean_alignment=sum(alignment) / len(alignment),
        ci_low=draws[max(0, int((alpha / 2.0) * resamples) - 1)],
        ci_high=draws[min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))],
        n_pairs=len(alignment),
        n_clusters=len(keys),
        mde=(normal_quantile(1.0 - alpha / 2.0) + normal_quantile(power))
        * math.sqrt(variance)
        / math.sqrt(len(per_cluster)),
        mean_control_alignment=sum(control) / len(control),
    )


@dataclass(frozen=True)
class TransplantResult:
    """One split's transplant measurement."""

    arm_mean: dict[str, float]
    gain: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_clusters: int
    cluster_sd: float
    mde: float
    per_target_mechanism: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def positive(self) -> bool:
        """Whether the gain is distinguishable from zero at the pre-registered level."""
        return self.ci_low > 0.0

    @property
    def adequately_powered(self) -> bool:
        return self.mde <= abs(self.gain) or self.positive

    def as_dict(self) -> dict:
        return {
            "arm_mean": dict(self.arm_mean),
            "gain_wrong_minus_correct": self.gain,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pairs": self.n_pairs,
            "n_clusters": self.n_clusters,
            "cluster_sd": self.cluster_sd,
            "mde": self.mde,
            "positive": self.positive,
            "per_target_mechanism": {
                name: dict(values) for name, values in sorted(self.per_target_mechanism.items())
            },
        }


def _cluster_means(gains: list[float], clusters: list[str]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for gain, cluster in zip(gains, clusters, strict=True):
        grouped.setdefault(cluster, []).append(gain)
    return grouped


def paired_transplant(
    arm_errors: dict[str, list[float]],
    clusters: list[str],
    targets: list[str] | None = None,
    seed: int = 0,
    resamples: int = 2000,
    alpha: float = 0.05,
    power: float = 0.8,
) -> TransplantResult:
    """Aggregate the three arms, with a cluster bootstrap over contexts.

    Resampling *contexts* rather than rows is not a refinement: the rows inside a
    context share ``X_0``, the action schedule and the filler, so they are one
    draw from the generator and resampling them independently would report a
    confidence interval narrower than the evidence supports. The unit of
    independent evidence here is the context, and that is what is resampled.

    A two-sided interval is reported against a one-sided minimum detectable
    effect, which is the pairing ADR 0007 asks for: the decision is "is the gain
    above zero", and the power statement is about the smallest gain this design
    could have resolved.
    """
    for arm in ARMS:
        if arm not in arm_errors:
            raise ValueError(f"missing arm {arm!r}")
    lengths = {arm: len(arm_errors[arm]) for arm in ARMS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"the arms are not aligned: {lengths}")
    if not arm_errors["self"]:
        raise ValueError("cannot aggregate an empty transplant set")
    if len(clusters) != lengths["self"]:
        raise ValueError("clusters and errors must be aligned")

    gains = [
        wrong - correct
        for wrong, correct in zip(arm_errors["wrong"], arm_errors["correct"], strict=True)
    ]
    grouped = _cluster_means(gains, clusters)
    cluster_keys = sorted(grouped)
    if len(cluster_keys) < 2:
        raise ValueError("a cluster bootstrap needs at least two clusters")

    per_cluster = [sum(grouped[key]) / len(grouped[key]) for key in cluster_keys]
    mean_gain = sum(per_cluster) / len(per_cluster)
    variance = sum((value - mean_gain) ** 2 for value in per_cluster) / (len(per_cluster) - 1)
    cluster_sd = math.sqrt(variance)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(len(per_cluster)):
            total += per_cluster[rng.randrange(len(per_cluster))]
        draws.append(total / len(per_cluster))
    draws.sort()
    low = draws[max(0, int((alpha / 2.0) * resamples) - 1)]
    high = draws[min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))]

    # Paired, so the effect size is the gain itself and the spread is the
    # between-context spread of that gain -- not the pooled spread of the two
    # arms, which would be dominated by how different one context is from another.
    mde = (normal_quantile(1.0 - alpha / 2.0) + normal_quantile(power)) * cluster_sd / math.sqrt(
        len(per_cluster)
    )

    per_target: dict[str, dict[str, float]] = {}
    if targets is not None:
        if len(targets) != lengths["self"]:
            raise ValueError("targets and errors must be aligned")
        by_target: dict[str, dict[str, list[float]]] = {}
        for index, target in enumerate(targets):
            bucket = by_target.setdefault(target, {arm: [] for arm in ARMS})
            for arm in ARMS:
                bucket[arm].append(arm_errors[arm][index])
        for target, bucket in sorted(by_target.items()):
            per_target[target] = {
                arm: sum(bucket[arm]) / len(bucket[arm]) for arm in ARMS
            } | {
                "gain": sum(
                    w - c for w, c in zip(bucket["wrong"], bucket["correct"], strict=True)
                )
                / len(bucket["wrong"])
            }

    return TransplantResult(
        arm_mean={arm: sum(arm_errors[arm]) / len(arm_errors[arm]) for arm in ARMS},
        gain=mean_gain,
        ci_low=low,
        ci_high=high,
        n_pairs=lengths["self"],
        n_clusters=len(cluster_keys),
        cluster_sd=cluster_sd,
        mde=mde,
        per_target_mechanism=per_target,
    )
