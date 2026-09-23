"""The two endpoints that need no decoder: CCGP and role equivariance.

Both are pure stdlib, so they run in the dependency-free CI job alongside the
simulator. That matters for the same reason it does there: these are the numbers
the phase reports, and a measurement that only runs on a machine with torch is
rarely re-checked.

Every evaluator here is paired with a **control** that it must fail on. A
readout that scores well on latents carrying no mechanism information is a broken
readout, not a discovery, and the only way to tell those apart is to point it at
something whose answer is known in advance.

``CCGP``
    Train a linear readout ``z -> M`` on latents from one set of conditions, test
    it on a disjoint set. The reported number is test accuracy against the
    compatibility-aware baseline, so "it generalizes" means "better than guessing
    a compatible mechanism", not "better than chance".

``RoleEquivariance``
    Fit one orthogonal ``T_swap`` on development filler families, then measure on
    unseen ones ``E_role = ||z_ji - T_swap z_ij||`` and ``E_inv = ||T_swap^2 - I||``.
    Orthogonal rather than general linear because a general map can absorb an
    arbitrary re-encoding, and then neither number says anything about roles.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..audits.leakage import apply_standard, fit_linear_probe, predict_probe, standardize
from .linalg import matmul, orthogonal_procrustes


@dataclass(frozen=True)
class CCGPResult:
    accuracy: float
    baseline: float
    n_train: int
    n_test: int
    n_conditions_train: int
    n_conditions_test: int

    @property
    def above_baseline(self) -> float:
        return self.accuracy - self.baseline

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "compatibility_aware_baseline": self.baseline,
            "above_baseline": self.above_baseline,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_conditions_train": self.n_conditions_train,
            "n_conditions_test": self.n_conditions_test,
        }


def ccgp(
    latents: list[tuple[float, ...]],
    mechanisms: list[str],
    conditions: list[str],
    train_conditions: set[str],
    test_conditions: set[str],
    chance_level: dict[str, float],
) -> CCGPResult:
    """Cross-condition generalization performance.

    ``conditions`` is the axis held out -- a filler family, an identity, an
    intervention regime. The readout is fit once, on the training conditions, and
    never sees a test condition; that is the whole point, so the two sets are
    checked disjoint rather than assumed to be.
    """
    if set(train_conditions) & set(test_conditions):
        raise ValueError("the condition partitions overlap; this is not held out")
    if len(latents) != len(mechanisms) or len(latents) != len(conditions):
        raise ValueError("latents, mechanisms and conditions must be aligned")

    mechanisms_sorted = sorted(set(mechanisms))
    index_of = {name: index for index, name in enumerate(mechanisms_sorted)}

    train_rows = [i for i, c in enumerate(conditions) if c in train_conditions]
    test_rows = [i for i, c in enumerate(conditions) if c in test_conditions]
    if not train_rows or not test_rows:
        raise ValueError("a condition partition is empty")

    # The test rows are scaled with the *training* set's statistics. Scaling them
    # by their own would let the readout use test-time means and variances it
    # could not have had at fit time, which is transductive adaptation and biases
    # this endpoint upward on precisely the axis it measures.
    train_features, means, scales = standardize([latents[i] for i in train_rows])
    test_features = apply_standard([latents[i] for i in test_rows], means, scales)
    weights = fit_linear_probe(
        train_features, [index_of[mechanisms[i]] for i in train_rows], len(mechanisms_sorted)
    )
    hits = sum(
        1
        for position, row in enumerate(test_rows)
        if predict_probe(weights, test_features[position]) == index_of[mechanisms[row]]
    )

    # Compatibility-aware baseline: guessing uniformly among the mechanisms the
    # row's condition admits. Supplied per condition rather than as one number
    # because the admissible count varies by tuple, and a flat 1/4 would be too
    # easy a baseline wherever it does not.
    baseline = sum(chance_level[conditions[row]] for row in test_rows) / len(test_rows)

    return CCGPResult(
        accuracy=hits / len(test_rows),
        baseline=baseline,
        n_train=len(train_rows),
        n_test=len(test_rows),
        n_conditions_train=len(train_conditions),
        n_conditions_test=len(test_conditions),
    )


@dataclass(frozen=True)
class RoleEquivarianceResult:
    e_role: float
    e_inv: float
    e_role_shuffled: float
    n_fit: int
    n_eval: int
    n_families_fit: int
    n_families_eval: int

    def as_dict(self) -> dict:
        return {
            "e_role": self.e_role,
            "e_inv": self.e_inv,
            "e_role_shuffled": self.e_role_shuffled,
            "role_gain": self.e_role_shuffled - self.e_role,
            "n_fit": self.n_fit,
            "n_eval": self.n_eval,
            "n_families_fit": self.n_families_fit,
            "n_families_eval": self.n_families_eval,
        }


def _rms(rows: list[tuple[float, ...]]) -> float:
    if not rows:
        raise ValueError("cannot take an rms of nothing")
    return math.sqrt(sum(value * value for row in rows for value in row) / (len(rows) * len(rows[0])))


def role_equivariance(
    forward: list[tuple[float, ...]],
    reverse: list[tuple[float, ...]],
    families: list[str],
    fit_families: set[str],
    eval_families: set[str],
    mechanism: str | None = None,
) -> RoleEquivarianceResult:
    """Fit one orthogonal ``T_swap`` on development families; score it on unseen ones.

    ``e_role_shuffled`` is the control and the number that makes ``e_role``
    readable: the same fit, scored against a *mismatched* pairing of forwards and
    reverses. If a transform that swaps nothing achieves a comparable residual,
    then ``e_role`` was measuring how alike the latents are in general rather
    than whether a shared swap explains them.
    """
    if set(fit_families) & set(eval_families):
        raise ValueError("the family partitions overlap; this is not held out")
    if not (len(forward) == len(reverse) == len(families)):
        raise ValueError("forward, reverse and families must be aligned")

    fit_rows = [i for i, family in enumerate(families) if family in fit_families]
    eval_rows = [i for i, family in enumerate(families) if family in eval_families]
    if not fit_rows or not eval_rows:
        raise ValueError("a family partition is empty")

    width = len(forward[0])
    transform = orthogonal_procrustes(
        [forward[i] for i in fit_rows], [reverse[i] for i in fit_rows]
    )
    if len(transform) != width:
        raise ValueError("the fitted transform does not match the latent width")

    def residual(rows: list[int], permuted: bool) -> float:
        rotated = matmul([forward[i] for i in rows], transform)
        targets = [reverse[i] for i in rows]
        if permuted:
            # A derangement of the targets, so no pair is compared with itself
            # and the control is not accidentally easier.
            order = list(range(len(rows)))
            rng = random.Random(12345)
            for _ in range(50):
                rng.shuffle(order)
                if all(a != b for a, b in zip(range(len(rows)), order, strict=True)):
                    break
            targets = [targets[position] for position in order]
        differences = [
            tuple(a - b for a, b in zip(predicted, target, strict=True))
            for predicted, target in zip(rotated, targets, strict=True)
        ]
        return _rms(differences)

    identity = [[1.0 if i == j else 0.0 for j in range(width)] for i in range(width)]
    squared = matmul(transform, transform)
    e_inv = _rms(
        [
            tuple(squared[i][j] - identity[i][j] for j in range(width))
            for i in range(width)
        ]
    )
    return RoleEquivarianceResult(
        e_role=residual(eval_rows, permuted=False),
        e_inv=e_inv,
        e_role_shuffled=residual(eval_rows, permuted=True),
        n_fit=len(fit_rows),
        n_eval=len(eval_rows),
        n_families_fit=len(fit_families),
        n_families_eval=len(eval_families),
    )


def synthetic_latents(
    mechanisms: list[str],
    conditions: list[str],
    width: int,
    seed: int,
    mechanism_weight: float,
    condition_weight: float,
) -> list[tuple[float, ...]]:
    """Latents with a known answer, for validating the evaluators.

    ``mechanism_weight`` controls how much of ``M`` the latent carries and
    ``condition_weight`` how much of the condition does. A CCGP readout must
    follow the first and ignore the second, so sweeping the two is a direct test
    of whether the evaluator measures what it claims: high mechanism weight with
    zero condition weight must generalize across conditions, and the reverse must
    not.
    """
    rng = random.Random(seed)
    mechanisms_sorted = sorted(set(mechanisms))
    conditions_sorted = sorted(set(conditions))
    mechanism_index = {name: i for i, name in enumerate(mechanisms_sorted)}
    condition_index = {name: i for i, name in enumerate(conditions_sorted)}
    mechanism_vectors = [
        tuple(rng.gauss(0.0, 1.0) for _ in range(width)) for _ in mechanisms_sorted
    ]
    condition_vectors = [
        tuple(rng.gauss(0.0, 1.0) for _ in range(width)) for _ in conditions_sorted
    ]
    return [
        tuple(
            mechanism_weight * mechanism_vectors[mechanism_index[mechanism]][dim]
            + condition_weight * condition_vectors[condition_index[condition]][dim]
            + rng.gauss(0.0, 1.0)
            for dim in range(width)
        )
        for mechanism, condition in zip(mechanisms, conditions, strict=True)
    ]
