"""Prior baselines B0 and B1_lookup, plus their calibration.

``B0`` is the constant global predicate distribution. ``B1_lookup`` is the
object-pair prior ``P(r | c_s, c_o)`` with Dirichlet backoff to the global
distribution -- the exact analogue of ``PairFrequencyBias`` in
``src/models/motifs.py``, which is the thing the visual probe has to beat.

Counts are built **only** from the split's own train subset, and the resulting
artifact records ``source_split_id`` so a downstream consumer can refuse a prior
fitted on the wrong data.  Under a strict pair-OOD eval set every eval pair has
zero training counts, so ``B1_lookup`` collapses to ``B0``: that is not a bug,
it is the reason ``C_prior_conflict`` must be measured on the relaxed split and
on IID val instead (see ``ood_split.build_pair_known_split``).

Stdlib only.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tools.ontology_probe.canonical_map import CanonicalMap
from tools.ontology_probe.vg_annotations import RelationTable, undirected_pair_key

__all__ = [
    "PriorTable",
    "build_prior_table",
    "calibrate_prior",
    "B0Predictor",
    "B1LookupPredictor",
]

ALPHA_GRID = (0.5, 1.0, 5.0, 20.0)
TEMPERATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
#: alpha used when no calibration has been run; matches the SGB default order.
DEFAULT_ALPHA = 1.0


@dataclass
class PriorTable:
    """Predicate statistics for one (split subset, label space) pair."""

    n_classes: int
    global_counts: list[float]
    pair_counts: dict[int, list[float]] = field(default_factory=dict)
    pair_totals: dict[int, float] = field(default_factory=dict)
    source_split_id: str = ""
    level: str = "vg50"
    n_relations: int = 0
    #: Undirected pair key -> set of class indices observed on it.  Used for the
    #: ambiguity and conflict diagnostics rather than for prediction.
    pair_classes: dict[int, set[int]] = field(default_factory=dict)

    @property
    def global_probs(self) -> list[float]:
        total = sum(self.global_counts)
        if total <= 0:
            return [1.0 / self.n_classes] * self.n_classes
        return [c / total for c in self.global_counts]

    def n_observed_on_pair(self, pair_key: int) -> float:
        return self.pair_totals.get(pair_key, 0.0)

    def is_pair_seen(self, c_s: int, c_o: int) -> bool:
        """True when the unordered pair has any training support in either order."""
        return self.n_observed_on_pair(undirected_pair_key(c_s, c_o)) > 0


def build_prior_table(
    table: RelationTable,
    rows: Iterable[int],
    cmap: CanonicalMap,
    source_split_id: str = "",
) -> PriorTable:
    """Count predicates for ``rows`` in the label space given by ``cmap``.

    ``cmap`` is required, including for the raw VG50 space (pass
    :func:`identity_map`).  A ``cmap=None`` branch would have to translate
    predicate ids 1..50 into class indices 0..49 by hand, and getting that wrong
    is a silent off-by-one that corrupts every prior downstream -- so the
    translation always goes through one code path.

    Counts are aggregated from raw fine counts rather than by pooling an
    already-smoothed fine distribution.  For *this* backoff family the two are in
    fact identical -- ``sum_{r in C} (n(r,pair) + a*Pg(r)) / (n(pair) + a)``
    equals ``(n(C,pair) + a*Pg(C)) / (n(pair) + a)`` exactly, because both the
    smoothing and the pooling are linear in the counts.  Rebuilding here is still
    the right choice: it keeps one definition of the prior, and it is the version
    that keeps working if the smoothing is ever made non-linear.  The identity is
    asserted in tests/analysis/test_ontology_probe_probes.py, and its practical
    consequence is that Δ_ontology is *identically zero* for prior-only cells --
    the whole ontology signal has to come from the learned probes.
    """
    rows = list(rows)
    n_classes = cmap.n_classes
    global_counts = [0.0] * n_classes
    pair_counts: dict[int, list[float]] = {}
    pair_totals: dict[int, float] = {}
    pair_classes: dict[int, set[int]] = collections.defaultdict(set)

    pairs = table.undirected_pairs()
    for row in rows:
        class_idx = cmap.remap(table.pred_id[row])
        global_counts[class_idx] += 1.0
        key = pairs[row]
        counts = pair_counts.get(key)
        if counts is None:
            counts = [0.0] * n_classes
            pair_counts[key] = counts
        counts[class_idx] += 1.0
        pair_totals[key] = pair_totals.get(key, 0.0) + 1.0
        pair_classes[key].add(class_idx)

    return PriorTable(
        n_classes=n_classes,
        global_counts=global_counts,
        pair_counts=pair_counts,
        pair_totals=pair_totals,
        source_split_id=source_split_id,
        level=cmap.level,
        n_relations=len(rows),
        pair_classes=dict(pair_classes),
    )


def _backoff_probs(
    table: PriorTable,
    c_s: int,
    c_o: int,
    alpha: float,
) -> list[float]:
    """``(n(r,pair) + alpha * P_global(r)) / (n(pair) + alpha)``."""
    global_probs = table.global_probs
    key = undirected_pair_key(c_s, c_o)
    counts = table.pair_counts.get(key)
    total = table.pair_totals.get(key, 0.0)
    if counts is None:
        # Unseen pair: the prior is exactly the global distribution. This is the
        # degenerate case that makes C_prior_conflict unmeasurable under a strict
        # pair-OOD split.
        return list(global_probs)
    denominator = total + alpha
    return [
        (counts[c] + alpha * global_probs[c]) / denominator
        for c in range(table.n_classes)
    ]


def _apply_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    """``p^(1/T)`` renormalised; T=1 leaves the distribution unchanged."""
    if temperature == 1.0:
        return list(probs)
    powered = [max(float(p), 0.0) ** (1.0 / temperature) for p in probs]
    total = sum(powered)
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in powered]


def _predict_rows(
    table: PriorTable,
    sub_table: RelationTable,
    rows: Sequence[int],
    alpha: float,
    temperature: float,
) -> list[list[float]]:
    cache: dict[tuple[int, int], list[float]] = {}
    out: list[list[float]] = []
    for row in rows:
        key = (sub_table.c_s[row], sub_table.c_o[row])
        probs = cache.get(key)
        if probs is None:
            probs = _apply_temperature(
                _backoff_probs(table, key[0], key[1], alpha), temperature
            )
            cache[key] = probs
        out.append(probs)
    return out


class B0Predictor:
    """Constant global predicate distribution; no pair information at all."""

    name = "B0"

    def __init__(self, table: PriorTable, temperature: float = 1.0) -> None:
        self.table = table
        self.temperature = temperature

    def predict_probs(self, sub_table: RelationTable, rows: Sequence[int]) -> list[list[float]]:
        probs = _apply_temperature(self.table.global_probs, self.temperature)
        return [list(probs) for _ in rows]

    def describe(self) -> dict[str, Any]:
        return {
            "probe": self.name,
            "inputs": [],
            "temperature": self.temperature,
            "source_split_id": self.table.source_split_id,
            "level": self.table.level,
        }


class B1LookupPredictor:
    """Object-pair lookup with Dirichlet backoff to the global distribution."""

    name = "B1_lookup"

    def __init__(
        self,
        table: PriorTable,
        alpha: float = DEFAULT_ALPHA,
        temperature: float = 1.0,
    ) -> None:
        self.table = table
        self.alpha = alpha
        self.temperature = temperature

    def predict_probs(self, sub_table: RelationTable, rows: Sequence[int]) -> list[list[float]]:
        return _predict_rows(self.table, sub_table, rows, self.alpha, self.temperature)

    def describe(self) -> dict[str, Any]:
        return {
            "probe": self.name,
            "inputs": ["c_s", "c_o"],
            "alpha": self.alpha,
            "temperature": self.temperature,
            "source_split_id": self.table.source_split_id,
            "level": self.table.level,
            "note": (
                "unseen pairs fall back to the global distribution, so under a "
                "strict pair-OOD eval set this predictor equals B0"
            ),
        }


def _dev_nll(
    table: PriorTable,
    dev_table: RelationTable,
    dev_rows: Sequence[int],
    labels: Sequence[int],
    alpha: float,
    temperature: float,
) -> float:
    probs = _predict_rows(table, dev_table, dev_rows, alpha, temperature)
    floor = 1e-12
    return -sum(
        math.log(max(row[labels[i]], floor)) for i, row in enumerate(probs)
    ) / len(probs)


def calibrate_prior(
    table: PriorTable,
    dev_table: RelationTable,
    dev_rows: Sequence[int],
    labels: Sequence[int],
    alpha_grid: Sequence[float] = ALPHA_GRID,
    temperature_grid: Sequence[float] = TEMPERATURE_GRID,
) -> dict[str, Any]:
    """Grid-search (alpha, temperature) on dev NLL.

    Selected on **dev**, never on the eval set, so the reported eval NLL is not
    an artifact of tuning to the number being reported.
    """
    if len(dev_rows) != len(labels):
        raise ValueError("dev_rows and labels must have equal length")
    results = []
    for alpha in alpha_grid:
        for temperature in temperature_grid:
            nll = _dev_nll(table, dev_table, dev_rows, labels, alpha, temperature)
            results.append({"alpha": alpha, "temperature": temperature, "dev_nll": nll})
    best = min(results, key=lambda r: r["dev_nll"])
    return {
        "alpha": best["alpha"],
        "temperature": best["temperature"],
        "dev_nll": best["dev_nll"],
        "grid": results,
        "source_split_id": table.source_split_id,
    }


def conflict_mask(
    table: PriorTable,
    sub_table: RelationTable,
    rows: Sequence[int],
    labels: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
) -> list[bool]:
    """Rows whose pair *is* seen in training but whose prior argmax is wrong.

    This is the strict ``C_prior_conflict`` set: the prior is confident and
    incorrect, so only visual evidence can recover the relation.  Rows with zero
    pair support are excluded -- there the prior is a constant and "the prior is
    wrong" carries no information.
    """
    mask = []
    for row, label in zip(rows, labels, strict=True):
        c_s, c_o = sub_table.c_s[row], sub_table.c_o[row]
        if not table.is_pair_seen(c_s, c_o):
            mask.append(False)
            continue
        probs = _backoff_probs(table, c_s, c_o, alpha)
        mask.append(max(range(len(probs)), key=lambda c: probs[c]) != label)
    return mask


def pair_ambiguity(table: PriorTable) -> dict[str, Any]:
    """How often a seen pair admits more than one predicate."""
    multi = sum(1 for classes in table.pair_classes.values() if len(classes) > 1)
    return {
        "n_pairs_seen": len(table.pair_classes),
        "n_pairs_multi_predicate": multi,
        "fraction_ambiguous": multi / len(table.pair_classes) if table.pair_classes else 0.0,
    }
