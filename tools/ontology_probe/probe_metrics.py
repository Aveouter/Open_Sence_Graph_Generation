"""Metrics and paired significance tests for the probe matrix.

Deliberately **pure stdlib**: these numbers are the experiment's output, so they
must be computable -- and unit-testable -- in the dependency-free CI job rather
than only where torch happens to be installed.

Reporting discipline, enforced here rather than by convention:

* ``on`` alone is 28.5% of VG150 train relations, so a constant-``on`` predictor
  scores 0.285 micro accuracy.  :func:`micro_metric_warning` is attached to every
  report for that reason; micro accuracy alone is never evidence.
* ``mR@1`` *is* macro recall.  Both are reported so nobody double-counts them as
  two independent results.
* Macro figures are only comparable across label spaces over a shared class set,
  so callers pass ``classes`` explicitly (see ``splits.pairwise_comparison_classes``).
"""

from __future__ import annotations

import collections
import math
from fractions import Fraction
from typing import Any, Iterable, Sequence

__all__ = [
    "accuracy",
    "macro_recall",
    "mean_nll",
    "recall_at_k",
    "mean_recall_at_k",
    "per_class_report",
    "hbt_strata",
    "paired_accuracy_diff",
    "paired_mean_diff",
    "paired_bootstrap_accuracy_diff",
    "paired_bootstrap_mean_diff",
    "ranked_predictions",
    "mcnemar_test",
    "mcnemar_exact",
    "micro_metric_warning",
    "summarise",
]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return float("nan")
    return sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)


def macro_recall(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Iterable[int] | None = None,
    min_support: int = 1,
) -> float:
    """Mean per-class recall over ``classes`` (default: every class present).

    Recall is used rather than precision because a probe predicts exactly one
    class per relation, so per-class recall and per-class accuracy coincide and
    the precision denominator would just be the predicted-class count.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    totals: dict[int, int] = collections.Counter(y_true)
    hits: dict[int, int] = collections.Counter(
        t for t, p in zip(y_true, y_pred, strict=True) if t == p
    )
    pool = sorted(totals) if classes is None else sorted(set(classes))
    recalls = [
        hits.get(c, 0) / totals[c] for c in pool if totals.get(c, 0) >= min_support
    ]
    if not recalls:
        return float("nan")
    return sum(recalls) / len(recalls)


def mean_nll(y_true: Sequence[int], probs: Sequence[Sequence[float]]) -> float:
    """Mean negative log-likelihood of the true class."""
    if len(y_true) != len(probs):
        raise ValueError("y_true and probs must have equal length")
    if not y_true:
        return float("nan")
    floor = 1e-12
    total = 0.0
    for truth, row in zip(y_true, probs, strict=True):
        total -= math.log(max(float(row[truth]), floor))
    return total / len(y_true)


def ranked_predictions(probs: Sequence[Sequence[float]]) -> list[list[int]]:
    """Class indices ordered by descending probability, one list per row."""
    return [sorted(range(len(row)), key=lambda c: -row[c]) for row in probs]


def recall_at_k(
    y_true: Sequence[int],
    probs: Sequence[Sequence[float]],
    k: int,
    ranked: Sequence[Sequence[int]] | None = None,
) -> dict[int, float]:
    """Per-class recall@k: fraction of class-c rows with c in the top-k.

    ``ranked`` lets a caller reuse one ordering across several k, which matters
    because ranking every row is the dominant cost of :func:`summarise` and it
    was previously repeated once per k.
    """
    if ranked is None:
        ranked = ranked_predictions(probs)
    totals: dict[int, int] = collections.Counter(y_true)
    hits: dict[int, int] = collections.Counter()
    for truth, order in zip(y_true, ranked, strict=True):
        if truth in order[:k]:
            hits[truth] += 1
    return {c: hits.get(c, 0) / n for c, n in totals.items() if n}


def mean_recall_at_k(
    y_true: Sequence[int],
    probs: Sequence[Sequence[float]],
    k: int,
    classes: Iterable[int] | None = None,
    ranked: Sequence[Sequence[int]] | None = None,
) -> float:
    """``mR@k``: per-class recall@k averaged over classes.

    ``mR@1`` equals :func:`macro_recall` by construction; they are reported
    together only so the relationship is explicit.
    """
    per_class = recall_at_k(y_true, probs, k, ranked=ranked)
    pool = sorted(per_class) if classes is None else sorted(set(classes))
    values = [per_class[c] for c in pool if c in per_class]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def per_class_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Sequence[Sequence[float]] | None = None,
    class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-class support, recall, and mean NLL, name-keyed for joining."""
    totals: dict[int, int] = collections.Counter(y_true)
    hits: dict[int, int] = collections.Counter(
        t for t, p in zip(y_true, y_pred, strict=True) if t == p
    )
    predicted: dict[int, int] = collections.Counter(y_pred)

    nll_by_class: dict[int, float] = collections.defaultdict(float)
    if probs is not None:
        floor = 1e-12
        for truth, row in zip(y_true, probs, strict=True):
            nll_by_class[truth] -= math.log(max(float(row[truth]), floor))

    rows: list[dict[str, Any]] = []
    for class_id in sorted(totals):
        n = totals[class_id]
        name = class_names[class_id] if class_names else str(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "support": n,
                "accuracy": hits.get(class_id, 0) / n,
                "n_predicted": predicted.get(class_id, 0),
                "mean_nll": (nll_by_class[class_id] / n) if probs is not None else None,
            }
        )
    return rows


def hbt_strata(
    per_class_rows: Sequence[dict[str, Any]],
    class_to_group: dict[str, str],
    value_key: str = "accuracy",
) -> dict[str, float]:
    """Average a per-class metric within the VG head/body/tail strata."""
    buckets: dict[str, list[float]] = collections.defaultdict(list)
    for row in per_class_rows:
        group = class_to_group.get(row["class_name"])
        if group and row[value_key] is not None:
            buckets[group].append(float(row[value_key]))
    return {
        group: sum(values) / len(values)
        for group, values in sorted(buckets.items())
        if values
    }


def _contingency(correct_a: Sequence[int], correct_b: Sequence[int]) -> tuple[int, int, int, int]:
    """(both correct, a only, b only, neither)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have equal length")
    both = a_only = b_only = neither = 0
    for a, b in zip(correct_a, correct_b, strict=True):
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    return both, a_only, b_only, neither


def paired_accuracy_diff(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """CI for ``acc(a) - acc(b)`` on paired predictions, from the exact variance.

    The paired accuracy difference is ``(A - B)/n`` where ``A`` and ``B`` are two
    cells of a multinomial over ``n`` paired outcomes, so its variance is
    available in closed form::

        Var(A - B) = n * [p1 + p2 - (p1 - p2)^2] / n^2

    with ``p1 = A/n``, ``p2 = B/n``.  A row-level bootstrap would need ``n``
    draws per resample, and stdlib ``random`` has no multinomial sampler before
    3.12, so it costs minutes per comparison to reproduce an interval the
    closed form gives exactly.  Significance is reported separately by
    :func:`mcnemar_test`, which does use the exact binomial on every row.
    """
    both, a_only, b_only, neither = _contingency(correct_a, correct_b)
    n = both + a_only + b_only + neither
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p1 = a_only / n
    p2 = b_only / n
    variance = (p1 + p2 - (p1 - p2) ** 2) / n
    se = math.sqrt(max(variance, 0.0))
    z = _norm_ppf(1.0 - alpha / 2.0)
    observed = p1 - p2
    return {
        "diff": observed,
        "lo": observed - z * se,
        "hi": observed + z * se,
        "se": se,
        "n_a_only": a_only,
        "n_b_only": b_only,
        "n_both": both,
        "n_neither": neither,
        "n": n,
        "ci_method": "multinomial_normal",
    }


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Avoids a scipy dependency for one constant; accurate to ~1e-9, far tighter
    than the interval needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


#: Previous name; kept so downstream callers keep working.
paired_bootstrap_accuracy_diff = paired_accuracy_diff


def paired_mean_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """CI for a paired mean difference, e.g. NLL, from the standard error.

    The statistic is a mean of per-row differences, so ``SE = sd(d) / sqrt(n)``
    is exact up to the CLT and costs one O(n) pass.  A row-level bootstrap would
    need ``n`` draws per resample -- minutes on a 62k-row set -- to approximate
    the same interval.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have equal length")
    diffs = [a - b for a, b in zip(values_a, values_b, strict=True)]
    n = len(diffs)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = sum(diffs) / n
    if n < 2:
        return {"diff": mean, "lo": float("nan"), "hi": float("nan"), "n": n}
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(variance / n)
    z = _norm_ppf(1.0 - alpha / 2.0)
    return {
        "diff": mean,
        "lo": mean - z * se,
        "hi": mean + z * se,
        "se": se,
        "n": n,
        "ci_method": "normal_se",
    }


#: Previous name; kept so downstream callers keep working.
paired_bootstrap_mean_diff = paired_mean_diff


#: Above this many discordant pairs the exact binomial sum is switched for a
#: normal approximation. The two agree closely at this size, and the exact sum
#: would otherwise add thousands of large-integer terms per comparison.
MCNEMAR_EXACT_LIMIT = 2000


def mcnemar_test(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict[str, Any]:
    """Two-sided McNemar test on paired correctness.

    Exact binomial rather than the chi-square approximation, because discordant
    counts are small for rare predicates where the approximation is
    anti-conservative.  The tail sum is evaluated as a :class:`~fractions.Fraction`
    since ``2.0**n`` overflows a float once there are more than ~1,024
    discordant pairs -- which pair_ood eval sets exceed easily.

    Above :data:`MCNEMAR_EXACT_LIMIT` discordant pairs a continuity-corrected
    normal approximation is used; ``method`` reports which was applied so a
    borderline p-value can be re-checked by hand.
    """
    _both, a_only, b_only, _neither = _contingency(correct_a, correct_b)
    n = a_only + b_only
    if n == 0:
        return {"b": a_only, "c": b_only, "p_value": 1.0, "method": "exact"}

    smaller = min(a_only, b_only)
    if n <= MCNEMAR_EXACT_LIMIT:
        tail = sum(math.comb(n, k) for k in range(0, smaller + 1))
        p = min(1.0, float(Fraction(2 * tail, 2**n)))
        method = "exact"
    else:
        # Continuity-corrected normal approximation, the standard large-sample
        # form: z = (|b - c| - 1) / sqrt(b + c).
        z = (abs(a_only - b_only) - 1) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2.0))
        method = "normal_approximation"
    return {
        "b": a_only,
        "c": b_only,
        "p_value": p,
        "method": method,
        "n_discordant": n,
    }


#: Backwards-compatible alias; the test is no longer exact at every size.
mcnemar_exact = mcnemar_test


def micro_metric_warning(predicate_share: float | None = None) -> str:
    share = 28.5 if predicate_share is None else predicate_share
    return (
        "micro accuracy is dominated by the most frequent predicate "
        f"(`on` is ~{share:.1f}% of VG150 train); report macro recall, "
        "per-predicate rows, and the C_* subsets instead of micro alone"
    )


def summarise(
    y_true: Sequence[int],
    probs: Sequence[Sequence[float]],
    classes: Iterable[int] | None = None,
    class_names: Sequence[str] | None = None,
    class_to_group: dict[str, str] | None = None,
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Full metric block for one cell."""
    # Ranked once and reused: ranking every row dominates the cost of this
    # function, and doing it per k tripled it for no benefit.
    ranked = ranked_predictions(probs)
    y_pred = [order[0] for order in ranked]
    report: dict[str, Any] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_recall": macro_recall(y_true, y_pred, classes),
        "nll": mean_nll(y_true, probs),
        "n": len(y_true),
        "micro_metric_warning": micro_metric_warning(),
    }
    for k in topk:
        key = f"mR@{k}"
        report[key] = mean_recall_at_k(y_true, probs, k, classes, ranked=ranked)
    report["mR@1_is_macro_recall"] = True
    if class_names:
        rows = per_class_report(y_true, y_pred, probs, class_names)
        report["per_class"] = rows
        if class_to_group:
            report["hbt"] = hbt_strata(rows, class_to_group)
    return report
