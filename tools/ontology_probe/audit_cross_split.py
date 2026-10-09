"""Cross-split isolation and prior-conflict audit for the ontology probe.

The probe pipeline fits every arm on one fitting set and scores several eval
sets.  Nothing in the pipeline checked that the *actual* fitting set is disjoint
from each eval set, so this audit does exactly that, and then characterises the
prior-conflict cohort under definitions stricter than the one the shipped mask
uses.  It exists because a split can audit itself clean and still be evaluated
on rows the model was fitted on -- the failure mode is invisible to the split's
own check.

Three independent questions, all answered from existing artifacts:

``isolation``
    Row- and image-level intersection between the actual fitting set
    (``pair_ood.train`` + ``pair_ood.dev``) and each eval set.  Rows are indices
    into one shared relation table, so an index present on both sides is the
    same annotated relation, not merely a similar one.

``conflict``
    The shipped ``C_prior_conflict`` mask against a direction-sensitive variant
    and against explicit per-pair-count and confidence thresholds.  Reported
    separately, never pooled: an ordered ``P(r | c_s -> c_o)`` and an unordered
    ``P(r | {c_s, c_o})`` are different objects, and averaging them would hide
    which one a number came from.

``clean_cohort``
    The subset of an eval set whose images are outside the fitting set.  This is
    the part that is held out under the actual fitting protocol, so it is where
    a conflict cohort can be measured at all.

Read-only.  The split JSONs, ``rel.json`` and ``train.json`` are opened for
reading and nothing else; the only file ever written is ``--output``.  Stdlib
only, so it runs in the dependency-free CI job.

Usage::

    python -m tools.ontology_probe.audit_cross_split \
        --data-root data/VisualGenome \
        --splits-dir outputs/analysis/ontology_probe/splits \
        --output reproduction/evidence/ontology_probe/cross_split_leakage.json

Exit status: 0 when the audit ran (regardless of what it found), 2 when an input
is missing or a recorded hash does not match.  A nonzero *intersection* is a
finding, not a tooling error, so it does not change the exit status -- the
result belongs in the JSON, where it can be cited.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    sha256_file,
    status_block,
    write_json,
)
from tools.ontology_probe.vg_annotations import (  # noqa: E402
    RelationTable,
    directed_pair_key,
    undirected_pair_key,
)

__all__ = [
    "CONFIDENCE_THRESHOLDS",
    "MIN_PAIR_COUNTS",
    "CountIndex",
    "PairCountKey",
    "build_count_index",
    "backoff_probs",
    "check_pins",
    "conflict_flags",
    "confidence_bins",
    "overlap_report",
    "run_audit",
    "main",
]

#: Confidence floors for the high-confidence-wrong-prior subset.  The shipped
#: ``conflict_mask`` applies none, so these are the thresholds that turn its
#: "wrong argmax" count into a claim about confidence.
CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9)

#: Minimum per-pair training count.  ``k=1`` is the shipped behaviour: any
#: support at all makes the pair eligible.
MIN_PAIR_COUNTS = (1, 2, 5, 10)

#: ``ordered`` keys on (c_s, c_o); ``unordered`` pools both directions, which is
#: what ``prior_baselines.is_pair_seen`` and therefore ``conflict_mask`` do.
DIRECTIONS = ("unordered", "ordered")

#: Direction-aware keys are the only place the two key definitions differ.
PairCountKey = int


@dataclass
class CountIndex:
    """Pair-conditioned predicate counts for one direction convention."""

    direction: str
    n_classes: int
    global_counts: list[float]
    pair_counts: dict[PairCountKey, list[float]] = field(default_factory=dict)
    pair_totals: dict[PairCountKey, float] = field(default_factory=dict)

    @property
    def global_probs(self) -> list[float]:
        total = sum(self.global_counts)
        if total <= 0:
            return [1.0 / self.n_classes] * self.n_classes
        return [c / total for c in self.global_counts]

    def key_for(self, c_s: int, c_o: int) -> PairCountKey:
        if self.direction == "ordered":
            return directed_pair_key(c_s, c_o)
        return undirected_pair_key(c_s, c_o)


def build_count_index(
    table: RelationTable,
    rows: Iterable[int],
    labels: Sequence[int],
    n_classes: int,
    direction: str = "unordered",
) -> CountIndex:
    """Count predicates per pair over ``rows`` under one direction convention.

    ``labels`` is the label space of each row, positionally aligned with
    ``rows``, and is required rather than defaulted to ``table.pred_id``.
    ``pred_id`` is 1..50 while a class index is 0..49, so the shortcut would be
    a silent off-by-one that shifts every count -- the same reason
    ``prior_baselines.build_prior_table`` demands a ``CanonicalMap`` even for the
    raw space instead of translating ids by hand.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    rows = list(rows)
    if len(labels) != len(rows):
        raise ValueError("labels must be positionally aligned with rows")

    index = CountIndex(direction=direction, n_classes=n_classes, global_counts=[0.0] * n_classes)
    for position, row in enumerate(rows):
        label = labels[position]
        if not 0 <= label < n_classes:
            raise ValueError(
                f"label {label} outside [0, {n_classes}); pred_id is 1-based and "
                "must be remapped through a CanonicalMap before it is used as a "
                "class index"
            )
        index.global_counts[label] += 1.0
        key = index.key_for(table.c_s[row], table.c_o[row])
        counts = index.pair_counts.get(key)
        if counts is None:
            counts = [0.0] * n_classes
            index.pair_counts[key] = counts
        counts[label] += 1.0
        index.pair_totals[key] = index.pair_totals.get(key, 0.0) + 1.0
    return index


def backoff_probs(index: CountIndex, key: PairCountKey, alpha: float) -> list[float] | None:
    """``(n(r,key) + alpha * P_global(r)) / (n(key) + alpha)``, or None if unseen.

    ``None`` rather than the global distribution: an unseen pair is not a
    prediction, it is the absence of one, and callers must decide what that
    means instead of silently scoring a constant.
    """
    counts = index.pair_counts.get(key)
    if counts is None:
        return None
    global_probs = index.global_probs
    denominator = index.pair_totals[key] + alpha
    return [
        (counts[c] + alpha * global_probs[c]) / denominator for c in range(index.n_classes)
    ]


def conflict_flags(
    table: RelationTable,
    rows: Sequence[int],
    labels: Sequence[int],
    index: CountIndex,
    alpha: float,
) -> list[dict[str, Any]]:
    """Per-row prior-conflict record: seen, wrong-argmax, confidence, support.

    ``seen`` is False for a pair with no training support, and such rows are
    excluded from every conflict count -- there the prior is a constant, so
    "the prior is wrong" carries no information.
    """
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have equal length")
    flags: list[dict[str, Any]] = []
    # The prior depends only on the pair key, and VG150 has a few thousand keys
    # against tens of thousands of eval rows, so resolve each key once.
    cache: dict[PairCountKey, tuple[int, float] | None] = {}
    for row, label in zip(rows, labels, strict=True):
        key = index.key_for(table.c_s[row], table.c_o[row])
        if key not in cache:
            probs = backoff_probs(index, key, alpha)
            if probs is None:
                cache[key] = None
            else:
                predicted = max(range(len(probs)), key=lambda c: probs[c])
                cache[key] = (predicted, probs[predicted])
        resolved = cache[key]
        if resolved is None:
            flags.append(
                {"seen": False, "wrong": False, "confidence": 0.0, "support": 0}
            )
            continue
        predicted, confidence = resolved
        flags.append(
            {
                "seen": True,
                "wrong": predicted != label,
                "confidence": confidence,
                "support": int(index.pair_totals[key]),
            }
        )
    return flags


def confidence_bins(flags: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count flags at or above each confidence floor.

    Confidence is read from the same distribution that selected the row.  Mixing
    an ordered selection with unordered confidence (or the reverse) measures two
    different priors, so callers must pass flags from one index.
    """
    return {
        f"tau>={threshold}": sum(
            1 for flag in flags if flag["wrong"] and flag["confidence"] >= threshold
        )
        for threshold in CONFIDENCE_THRESHOLDS
    }


def overlap_report(
    table: RelationTable,
    rows_a: Sequence[int],
    rows_b: Sequence[int],
) -> dict[str, Any]:
    """Row- and image-level intersection of two index sets over one table."""
    set_a, set_b = set(rows_a), set(rows_b)
    shared_rows = set_a & set_b
    images_a = {table.image_id[row] for row in set_a}
    images_b = {table.image_id[row] for row in set_b}
    shared_images = images_a & images_b
    return {
        "n_rows_a": len(set_a),
        "n_rows_b": len(set_b),
        "shared_rows": len(shared_rows),
        "shared_row_sample": sorted(shared_rows)[:20],
        "n_images_a": len(images_a),
        "n_images_b": len(images_b),
        "shared_images": len(shared_images),
        "shared_image_sample": sorted(shared_images)[:20],
        "rows_in_b_on_shared_images": sum(
            1 for row in set_b if table.image_id[row] in images_a
        ),
        "fraction_of_b_rows_shared": len(shared_rows) / len(set_b) if set_b else 0.0,
        "fraction_of_b_images_shared": (
            len(shared_images) / len(images_b) if images_b else 0.0
        ),
    }


def check_pins(provenance: dict[str, Any], expected: dict[str, Any]) -> None:
    """Raise if a recorded input hash does not match what was just measured.

    A number computed from inputs other than the recorded ones looks exactly
    like a number computed from them, so a mismatch is fatal rather than a
    warning -- this is the whole reason the audit records input digests.
    """
    for field_name, pinned in expected.items():
        actual = provenance.get(field_name)
        if actual is None:
            raise ValueError(f"cannot pin {field_name}: not present in provenance")
        if pinned != actual:
            raise ValueError(
                f"{field_name} mismatch: recorded {pinned}, actual {actual}; "
                "refusing to report numbers from inputs that are not the "
                "recorded ones"
            )


def _conflict_block(
    table: RelationTable,
    rows: Sequence[int],
    labels: Sequence[int],
    index: CountIndex,
    alpha: float,
) -> dict[str, Any]:
    """Conflict counts for one cohort under one direction convention.

    The count floor and the confidence floor are applied to the same flags, so
    ``k`` and ``tau`` compose: a ``tau>=0.9`` figure inside ``k>=5`` is the
    high-confidence subset of a cohort that already required five observations.
    """
    flags = conflict_flags(table, rows, labels, index, alpha)
    block: dict[str, Any] = {
        "n_rows": len(rows),
        "n_rows_on_seen_pair": sum(1 for flag in flags if flag["seen"]),
    }
    for k in MIN_PAIR_COUNTS:
        selected = [
            flag for flag in flags if flag["seen"] and flag["support"] >= k and flag["wrong"]
        ]
        entry: dict[str, Any] = {"n": len(selected)}
        if selected:
            entry["confidence_bins"] = confidence_bins(selected)
            entry["max_confidence"] = max(flag["confidence"] for flag in selected)
        block[f"k>={k}"] = entry
    return block


def run_audit(
    data_root: Path,
    splits_dir: Path,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full audit and return the machine-readable record.

    ``expected`` optionally pins input hashes (``rel_json_sha256``,
    ``train_json_sha256``, ``relation_order_sha256``); a mismatch raises rather
    than reporting numbers computed from inputs other than the recorded ones.
    """
    from tools.ontology_probe.canonical_map import identity_map
    from tools.ontology_probe.prior_baselines import build_prior_table, calibrate_prior
    from tools.ontology_probe.splits import load_split
    from tools.ontology_probe.vg_annotations import (
        build_relation_table,
        load_vg_index,
    )

    data_root, splits_dir = Path(data_root), Path(splits_dir)
    expected = expected or {}

    index = load_vg_index(data_root, "train")
    table = build_relation_table(index)
    from tools.ontology_probe.build_splits import relation_order_sha256

    order_sha = relation_order_sha256(table)

    ood = load_split("pair_ood", splits_dir, table=table, verify=True)
    known = load_split("pair_known", splits_dir, table=table, verify=True)
    ood_train, ood_dev, ood_eval = ood.rows("train"), ood.rows("dev"), ood.rows("eval")
    known_eval = known.rows("eval")

    provenance: dict[str, Any] = {
        "rel_json_sha256": sha256_file(data_root / "rel.json"),
        "train_json_sha256": sha256_file(data_root / "train.json"),
        "relation_order_sha256": order_sha,
        "n_relations": len(table),
        "split_file_sha256": {
            name: sha256_file(splits_dir / f"split_{name}.json")
            for name in ("pair_ood", "pair_known")
        },
    }
    check_pins(provenance, expected)

    # The fitting set is what the pipeline actually fit on, not what a split
    # file says its own train part is.  eval_matrix and train_probes both pass
    # pair_ood.train (counts, probe fitting) and pair_ood.dev (calibration) for
    # every arm, so the union is the set an eval image must avoid.
    fitting_rows = sorted(set(ood_train) | set(ood_dev))
    fitting_images = {table.image_id[row] for row in fitting_rows}

    isolation = {
        "fitting_set": "pair_ood.train + pair_ood.dev",
        "comparisons": {
            name: overlap_report(table, rows_a, rows_b)
            for name, rows_a, rows_b in (
                ("fitting_vs_pair_known_eval", fitting_rows, known_eval),
                ("pair_ood_train_vs_pair_known_eval", ood_train, known_eval),
                ("pair_ood_dev_vs_pair_known_eval", ood_dev, known_eval),
                # Control channels.  The first is disjoint by construction and
                # the second is the only channel the split audits itself, so
                # both must read zero for the measurement above to be trusted;
                # a nonzero value here would mean the method is broken, not that
                # a leak was found.
                ("pair_ood_train_vs_pair_ood_eval", ood_train, ood_eval),
                ("pair_known_train_vs_pair_known_eval", known.rows("train"), known_eval),
            )
        },
    }

    # The iid arm is a different table (VG val) scored with the same fitting
    # set, so it cannot appear above.  VG's own split is what keeps it clean, and
    # that is checked here rather than assumed -- the finding above does not
    # generalise to iid unless this reads zero.
    val_index = load_vg_index(data_root, "val")
    val_images = set(val_index.rels_by_image)
    isolation["iid_control"] = {
        "eval_set": "VG val (every relation, as eval_matrix scores it)",
        "n_val_images": len(val_images),
        "n_fitting_images": len(fitting_images),
        "shared_images": len(val_images & fitting_images),
        "shared_image_sample": sorted(val_images & fitting_images)[:20],
    }

    cmap = identity_map(index.predicate_names)
    prior = build_prior_table(table, ood_train, cmap=cmap, source_split_id="pair_ood.train")
    calibration = calibrate_prior(
        prior, table, ood_dev, [cmap.remap(table.pred_id[row]) for row in ood_dev]
    )
    alpha = calibration["alpha"]

    clean_eval = [row for row in known_eval if table.image_id[row] not in fitting_images]
    clean_set = set(clean_eval)
    excluded_eval = [row for row in known_eval if row not in clean_set]

    cohorts = {
        "pair_known.eval": known_eval,
        "pair_known.eval_clean_subset": clean_eval,
        "pair_known.eval_in_fitting_images": excluded_eval,
    }

    # One count index per direction, built once and reused across cohorts: the
    # fitting set is identical for all of them, only the eval rows differ.
    train_labels = [cmap.remap(table.pred_id[row]) for row in ood_train]
    indices = {
        direction: build_count_index(
            table, ood_train, train_labels, cmap.n_classes, direction=direction
        )
        for direction in DIRECTIONS
    }

    conflict: dict[str, Any] = {
        "alpha": alpha,
        "alpha_source": "pair_ood.dev calibration (same grid the shipped pipeline uses)",
        "cohorts": {},
    }
    for cohort_name, rows in cohorts.items():
        labels = [cmap.remap(table.pred_id[row]) for row in rows]
        conflict["cohorts"][cohort_name] = {
            direction: _conflict_block(table, rows, labels, indices[direction], alpha)
            for direction in DIRECTIONS
        }

    return status_block(
        audit="ontology_probe_cross_split",
        read_only=True,
        provenance=provenance,
        isolation=isolation,
        conflict=conflict,
        clean_cohort_definition=(
            "eval rows whose image is absent from the actual fitting set "
            "(pair_ood.train + pair_ood.dev); this is the part held out under "
            "the fitting protocol the pipeline actually uses"
        ),
        interpretation={
            "historical_pair_known_eval": "NON_HELD_OUT",
            "clean_subset_status": "CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED",
            "scope": (
                "measures isolation and cohort size only; it does not recompute "
                "any model metric and does not establish that contamination "
                "explains the conflict-rate difference between cohorts"
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--splits-dir", default=str(DEFAULT_OUTPUT_ROOT / "splits"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expect-rel-json-sha256",
        default=None,
        help="fail unless rel.json matches this digest",
    )
    parser.add_argument("--expect-train-json-sha256", default=None)
    parser.add_argument("--expect-relation-order-sha256", default=None)
    args = parser.parse_args(argv)

    expected = {
        key: value
        for key, value in (
            ("rel_json_sha256", args.expect_rel_json_sha256),
            ("train_json_sha256", args.expect_train_json_sha256),
            ("relation_order_sha256", args.expect_relation_order_sha256),
        )
        if value is not None
    }

    for path in (Path(args.data_root) / "rel.json", Path(args.data_root) / "train.json"):
        if not path.exists():
            print(f"[audit_cross_split] missing input: {path}", file=sys.stderr)
            return 2

    try:
        report = run_audit(Path(args.data_root), Path(args.splits_dir), expected=expected)
    except ValueError as exc:
        print(f"[audit_cross_split] {exc}", file=sys.stderr)
        return 2

    write_json(Path(args.output), report)
    comparisons = report["isolation"]["comparisons"]
    print("[audit_cross_split] isolation:")
    for name, block in comparisons.items():
        print(
            f"  {name}: rows={block['shared_rows']} images={block['shared_images']}"
        )
    clean = report["conflict"]["cohorts"]["pair_known.eval_clean_subset"]
    print(
        "[audit_cross_split] clean-cohort conflict (unordered): "
        + " ".join(f"k>={k}={clean['unordered'][f'k>={k}']['n']}" for k in MIN_PAIR_COUNTS)
    )
    print(f"[audit_cross_split] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
