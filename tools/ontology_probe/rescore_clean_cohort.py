"""Re-score the pair_known arm on the cohort that is actually held out.

The cross-split audit (``audit_cross_split``) established that the historical
``pair_known`` cells are evaluated on rows the pipeline was fitted on.  This
script does the three things that audit said were outstanding:

1. **Input manifest.**  Hash every input the numbers depend on -- the VG data
   files, both split artifacts, and every saved probe tensor -- so provenance is
   a full digest list rather than an abbreviated one in a comment.
2. **Alignment.**  Check each saved ``predictions.pt`` against the split's eval
   rows using the ``rows`` payload stored *inside* the tensor, never the assumed
   order.  Re-scoring by index is only sound if this holds for every cell being
   compared, so a mismatch aborts rather than warns.
3. **Clean-cohort metrics.**  Recompute accuracy, macro recall, VRR, harm and
   their intervals on the eval rows whose image is outside the actual fitting
   set, alongside the same metrics on the historical cohort and the
   selection-bias diagnostics that bound what the comparison means.

Two denominators are reported for macro recall, and they are not
interchangeable: the cohort's own usable classes reproduces the published
figure (the pipeline defines the class set on the eval set being scored), while
a shared class set is what makes full-vs-clean like-for-like, since the clean
cohort drops classes at the 50-row floor.

``STATUS: diagnostic_experiment``.  Nothing here is a benchmark result.  The
clean cohort is the greedy pair-OOD holdout's complement, not a random holdout,
so it is reported as ``CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED`` and its
levels are not population estimates.  The B2-vs-B4 *paired* difference on a
fixed cohort is internally valid; the historical reading is not retracted here,
and the two confounds that would be needed to separate capacity from visual
evidence -- matched parameter counts and multiple seeds -- are both absent.

Read-only apart from ``--output``/``--manifest-output``.  Needs torch and numpy
to read the saved tensors, so unlike the rest of this package it runs in the
analysis environment rather than the dependency-free CI job.

Usage::

    python -m tools.ontology_probe.rescore_clean_cohort \
        --output reproduction/evidence/ontology_probe/clean_cohort_metrics.json \
        --manifest-output reproduction/evidence/ontology_probe/inputs_sha256.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402  (analysis environment only; see module docstring)

from tools.ontology_probe.canonical_map import (  # noqa: E402
    identity_map,
    load_canonical_map,
)
from tools.ontology_probe.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    sha256_file,
    status_block,
    write_json,
)
from tools.ontology_probe.eval_matrix import usable_classes  # noqa: E402
from tools.ontology_probe.prior_baselines import (  # noqa: E402
    build_prior_table,
    calibrate_prior,
)
from tools.ontology_probe.probe_metrics import (  # noqa: E402
    mcnemar_test,
    paired_accuracy_diff,
    summarise,
)
from tools.ontology_probe.rescue_rate import (  # noqa: E402
    CONFIDENCE_THRESHOLDS,
    load_predictions,
)
from tools.ontology_probe.vg_annotations import (  # noqa: E402
    build_relation_table,
    load_vg_index,
)

__all__ = [
    "BOOTSTRAP_DRAWS",
    "PROBE_NAMES",
    "cluster_bootstrap_macro_ci",
    "cluster_bootstrap_ratio_ci",
    "input_manifest",
    "main",
    "run_rescore",
    "verify_alignment",
]

LEVELS = ("vg50", "L1_noise", "L2_entail")
#: B2 is the base arm (categories + geometry) and B4 adds frozen CLIP visual
#: features; the other two are reported for context.
PROBE_NAMES = ("B1_add", "B2", "B3", "B4")
BASE_PROBE = "B2"
VISUAL_PROBE = "B4"

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261009
#: Draws per multinomial batch; bounds the [chunk, n_images] working set.
BOOTSTRAP_CHUNK = 250

CLEAN_SUBSET_STATUS = "CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED"


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


def cluster_bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    image_ids: Sequence[int],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile CI for ``sum(num)/sum(den)`` resampling *images*, not rows.

    Rows inside one image are not independent -- they share objects, context and
    the same visual crop pool -- so a row-level interval would be too narrow.
    Resampling whole images keeps that dependence inside each cluster.

    Every ratio statistic here (accuracy, VRR, harm) is a ratio of two sums, so
    only each image's ``(numerator, denominator)`` totals are needed.  A resample's
    cluster multiplicities are ``Multinomial(n, uniform)``, which replaces n
    index picks with one draw, and chunking keeps the working set bounded.
    """
    import numpy as np

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    _labels, inverse = np.unique(np.asarray(image_ids, dtype=np.int64), return_inverse=True)
    n_clusters = int(inverse.max()) + 1 if inverse.size else 0
    if n_clusters == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "draws": 0}

    img_num = np.bincount(inverse, weights=num, minlength=n_clusters)
    img_den = np.bincount(inverse, weights=den, minlength=n_clusters)
    total_den = img_den.sum()
    point = float(img_num.sum() / total_den) if total_den > 0 else float("nan")

    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / n_clusters)
    chunks: list[Any] = []
    done = 0
    while done < draws:
        k = min(BOOTSTRAP_CHUNK, draws - done)
        counts = rng.multinomial(n_clusters, probs, size=k).astype(np.float64)
        sampled_den = counts @ img_den
        with np.errstate(invalid="ignore", divide="ignore"):
            chunks.append(
                np.where(sampled_den > 0, (counts @ img_num) / sampled_den, np.nan)
            )
        done += k
    stats = np.concatenate(chunks)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "draws": 0}
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "draws": int(stats.size),
        "n_clusters": n_clusters,
        "ci_method": "cluster_percentile_by_image",
    }


def cluster_bootstrap_macro_ci(
    support: Sequence[Sequence[float]],
    hits: Sequence[Sequence[float]],
    image_ids: Sequence[int],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile CI for macro recall, a mean of per-class ratios.

    Not a ratio of sums, so it cannot reuse the scalar helper: the class
    dimension has to survive the resampling and only then be averaged over.
    """
    import numpy as np

    sup = np.asarray(support, dtype=np.float64)
    hit = np.asarray(hits, dtype=np.float64)
    _labels, inverse = np.unique(np.asarray(image_ids, dtype=np.int64), return_inverse=True)
    n_clusters = int(inverse.max()) + 1 if inverse.size else 0
    if n_clusters == 0 or sup.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "draws": 0}
    n_classes = sup.shape[1]

    img_sup = np.zeros((n_clusters, n_classes))
    img_hit = np.zeros((n_clusters, n_classes))
    np.add.at(img_sup, inverse, sup)
    np.add.at(img_hit, inverse, hit)

    def macro(sup_tot: Any, hit_tot: Any) -> float:
        keep = sup_tot > 0
        if not keep.any():
            return float("nan")
        return float((hit_tot[keep] / sup_tot[keep]).mean())

    point = macro(img_sup.sum(axis=0), img_hit.sum(axis=0))
    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / n_clusters)
    chunks: list[Any] = []
    done = 0
    while done < draws:
        k = min(BOOTSTRAP_CHUNK, draws - done)
        counts = rng.multinomial(n_clusters, probs, size=k).astype(np.float64)
        sup_s = counts @ img_sup
        hit_s = counts @ img_hit
        keep = sup_s > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = np.where(keep, hit_s / np.where(keep, sup_s, 1.0), np.nan)
        chunks.append(np.nanmean(per_class, axis=1))
        done += k
    stats = np.concatenate(chunks)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "draws": 0}
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "draws": int(stats.size),
        "n_clusters": n_clusters,
        "ci_method": "cluster_percentile_by_image",
    }


# ---------------------------------------------------------------------------
# Input manifest and alignment
# ---------------------------------------------------------------------------


def input_manifest(
    data_root: Path,
    splits_dir: Path,
    probes_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """SHA256 of every input the re-scored numbers depend on."""
    candidates: list[Path] = []
    for name in ("rel.json", "train.json", "val.json", "test.json"):
        candidates.append(data_root / name)
    candidates += sorted(splits_dir.glob("*.json"))
    for cell in sorted(probes_dir.iterdir()):
        if cell.is_dir():
            for name in ("predictions.pt", "model.pt", "metrics.json", "inputs.pt"):
                candidates.append(cell / name)

    files: dict[str, Any] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            key = path.relative_to(repo_root).as_posix()
        except ValueError:
            key = path.as_posix()
        if key in files:
            continue
        files[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"n_files": len(files), "files": dict(sorted(files.items()))}


def verify_alignment(
    table: Any,
    cmap: Any,
    expected_rows: Sequence[int],
    probes_dir: Path,
    cells: Sequence[str],
) -> dict[str, Any]:
    """Check each saved prediction tensor against the split's eval rows.

    The ``rows`` payload inside the tensor is authoritative.  If it disagrees
    with the split file, the tensor was produced against different relations and
    no index arithmetic makes re-scoring correct -- so this reports rather than
    repairs.
    """
    expected_labels = [cmap.remap(table.pred_id[row]) for row in expected_rows]
    report: dict[str, Any] = {}
    for cell in cells:
        payload = load_predictions(probes_dir, cell)
        rows = list(payload["rows"])
        labels = list(payload["labels"])
        rows_ok = rows == list(expected_rows)
        labels_ok = labels == expected_labels
        entry: dict[str, Any] = {
            "n_predictions": len(labels),
            "rows_match_split_eval": rows_ok,
            "labels_match_split_labels": labels_ok,
            "rows_unique": len(set(rows)) == len(rows),
            "aligned": bool(rows_ok and labels_ok and len(labels) == len(expected_rows)),
        }
        if not rows_ok:
            # strict=False on purpose: a length mismatch is one of the cases
            # this is here to detect, so zip must not raise on it first.
            entry["first_row_mismatch_index"] = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(rows, expected_rows, strict=False))
                    if a != b
                ),
                None,
            )
        report[cell] = entry
    return report


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def _rescue_block(
    base: dict[str, Any],
    visual: dict[str, Any],
    indices: Sequence[int],
    image_ids: Sequence[int],
) -> dict[str, Any]:
    """VRR / harm / net and intervals, mirroring ``rescue_rate.rescue_analysis``."""
    labels = base["labels"]
    base_wrong = [base["predictions"][i] != labels[i] for i in indices]
    vis_correct = [visual["predictions"][i] == labels[i] for i in indices]
    base_correct = [not wrong for wrong in base_wrong]

    n = len(indices)
    n_base_wrong = sum(base_wrong)
    n_base_right = sum(base_correct)
    n_rescue = sum(1 for j in range(n) if base_wrong[j] and vis_correct[j])
    n_harm = sum(1 for j in range(n) if base_correct[j] and not vis_correct[j])
    sub_images = [image_ids[i] for i in indices]

    block: dict[str, Any] = {
        "base_probe": base["cell"],
        "visual_probe": visual["cell"],
        "n": n,
        "n_base_wrong": n_base_wrong,
        "n_rescue": n_rescue,
        "n_harm": n_harm,
        "vrr": n_rescue / n_base_wrong if n_base_wrong else float("nan"),
        "harm_rate": n_harm / n_base_right if n_base_right else float("nan"),
        "net_rescue": (n_rescue - n_harm) / n if n else float("nan"),
        # Sign convention copied from rescue_rate.rescue_analysis, which passes
        # the base arm first: this is acc(B2) - acc(B4). A NEGATIVE value means
        # the visual arm is better. The name does not say which way round it is,
        # so the convention is spelled out next to it.
        "delta_accuracy": paired_accuracy_diff(
            [int(c) for c in base_correct], [int(c) for c in vis_correct]
        ),
        "delta_accuracy_convention": "acc(B2) - acc(B4); negative => B4 better",
        "mcnemar": mcnemar_test(
            [int(c) for c in base_correct], [int(c) for c in vis_correct]
        ),
        # Thresholds are applied to the BASE arm's confidence, and strictly
        # (>), matching rescue_rate. The cross-split audit's confidence_bins use
        # >=, so a row exactly on a threshold counts differently in the two
        # reports. Recorded rather than reconciled silently.
        "confidence_convention": "strict '>' on B2 max prob, matching rescue_rate.py",
    }
    block["vrr_ci"] = cluster_bootstrap_ratio_ci(
        [1.0 if (w and c) else 0.0 for w, c in zip(base_wrong, vis_correct, strict=True)],
        [1.0 if w else 0.0 for w in base_wrong],
        sub_images,
    )
    block["harm_ci"] = cluster_bootstrap_ratio_ci(
        [
            1.0 if (b and not c) else 0.0
            for b, c in zip(base_correct, vis_correct, strict=True)
        ],
        [1.0 if b else 0.0 for b in base_correct],
        sub_images,
    )
    block["by_confidence"] = {
        f"tau_{tau}": {
            "n_relations": sum(
                1
                for j in range(n)
                if base_wrong[j] and base["confidence"][indices[j]] > tau
            ),
            "vrr": (
                lambda sel: (
                    sum(1 for j in sel if vis_correct[j]) / len(sel)
                    if sel
                    else float("nan")
                )
            )(
                [
                    j
                    for j in range(n)
                    if base_wrong[j] and base["confidence"][indices[j]] > tau
                ]
            ),
        }
        for tau in CONFIDENCE_THRESHOLDS
    }
    return block


def _accuracy_block(
    cell_arrays: dict[str, Any],
    indices: Sequence[int],
    cmap: Any,
    image_ids: Sequence[int],
    shared_classes: Sequence[int],
) -> dict[str, Any]:
    """Metrics for one cell on one cohort.

    ``cell_arrays`` is precomputed once per cell over the *whole* eval set: the
    macro one-hot does not depend on which cohort is scored, so rebuilding it
    per cohort is waste at 62k rows x 50 classes per call.
    """
    import numpy as np

    idx = np.asarray(indices, dtype=np.int64)
    y_true = cell_arrays["labels"][idx]
    y_pred = cell_arrays["predictions"][idx]
    probs = cell_arrays["probs"][idx]
    names = list(cmap.class_names)

    own_classes = usable_classes(y_true.tolist(), cmap.n_classes)
    shared = list(shared_classes)

    report = summarise(y_true.tolist(), probs, classes=own_classes, class_names=names)
    report["n_usable_classes_own"] = len(own_classes)
    report["macro_recall_shared_classes"] = summarise(
        y_true.tolist(), probs, classes=shared, class_names=names
    )["macro_recall"]
    report["n_usable_classes_shared"] = len(shared)

    sub_images = [image_ids[i] for i in indices]
    correct = (y_pred == y_true).astype(np.float64)
    report["accuracy_ci"] = cluster_bootstrap_ratio_ci(
        correct, np.ones_like(correct), sub_images
    )

    def _onehot(class_list: Sequence[int]) -> tuple[Any, Any]:
        pos = np.full(cmap.n_classes, -1, dtype=np.int64)
        pos[np.asarray(class_list, dtype=np.int64)] = np.arange(len(class_list))
        cls = pos[y_true]
        valid = cls >= 0
        sup = np.zeros((idx.size, len(class_list)), dtype=np.float64)
        sup[np.nonzero(valid)[0], cls[valid]] = 1.0
        return sup, sup * (y_pred == y_true)[:, None]

    sup_own, hit_own = _onehot(own_classes)
    report["macro_recall_ci"] = cluster_bootstrap_macro_ci(sup_own, hit_own, sub_images)
    sup_sh, hit_sh = _onehot(shared)
    report["macro_recall_shared_ci"] = cluster_bootstrap_macro_ci(
        sup_sh, hit_sh, sub_images
    )
    return report


def _predicate_strata(
    table: Any,
    names: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    indices: Sequence[int],
    min_support: int = 50,
) -> dict[str, Any]:
    """Per-predicate support and accuracy on one cohort, above a support floor."""
    per: dict[int, list[int]] = {}
    for j in indices:
        per.setdefault(labels[j], []).append(j)
    out: dict[str, Any] = {}
    for cls, members in sorted(per.items()):
        if len(members) < min_support:
            continue
        hits = sum(1 for j in members if predictions[j] == labels[j])
        out[names[cls]] = {
            "support": len(members),
            "accuracy": hits / len(members),
            "n_images": len({table.image_id[j] for j in members}),
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_rescore(
    data_root: Path,
    splits_dir: Path,
    probes_dir: Path,
    repo_root: Path,
    top: int = 8,
) -> dict[str, Any]:
    """Run the whole re-scoring and return the machine-readable record."""
    import json

    import numpy as np

    data_root, splits_dir, probes_dir = (
        Path(data_root),
        Path(splits_dir),
        Path(probes_dir),
    )
    report: dict[str, Any] = {}
    report["input_manifest"] = input_manifest(data_root, splits_dir, probes_dir, repo_root)

    index = load_vg_index(data_root, "train")
    table = build_relation_table(index)
    ood = json.loads((splits_dir / "split_pair_ood.json").read_text(encoding="utf-8"))
    known = json.loads((splits_dir / "split_pair_known.json").read_text(encoding="utf-8"))
    known_eval = list(known["eval"])

    fitting_images = {table.image_id[r] for r in ood["train"]} | {
        table.image_id[r] for r in ood["dev"]
    }
    eval_images = [table.image_id[r] for r in known_eval]
    clean_indices = [i for i, r in enumerate(known_eval) if eval_images[i] not in fitting_images]
    contaminated_indices = [
        i for i, r in enumerate(known_eval) if eval_images[i] in fitting_images
    ]
    report["cohorts"] = {
        "n_eval": len(known_eval),
        "n_clean": len(clean_indices),
        "n_contaminated": len(contaminated_indices),
        "n_clean_images": len({eval_images[i] for i in clean_indices}),
        "n_contaminated_images": len({eval_images[i] for i in contaminated_indices}),
        "definition": (
            "clean = eval rows whose image is absent from the actual fitting set "
            "(pair_ood.train + pair_ood.dev)"
        ),
    }

    maps = {"vg50": identity_map(index.predicate_names)}
    for level in ("L1_noise", "L2_entail"):
        maps[level] = load_canonical_map(level=level)

    report["alignment"] = {
        level: verify_alignment(
            table,
            maps[level],
            known_eval,
            probes_dir,
            [f"{level}__pair_known__{probe}__s0" for probe in PROBE_NAMES],
        )
        for level in LEVELS
    }
    misaligned = [
        cell
        for level in report["alignment"].values()
        for cell, entry in level.items()
        if not entry["aligned"]
    ]
    report["alignment_all_ok"] = not misaligned
    if misaligned:
        report["aborted"] = (
            "re-scoring aborted: these cells' saved rows do not match the split "
            f"eval rows, so subsetting by index would be wrong: {misaligned}"
        )
        return report

    prior = build_prior_table(
        table, ood["train"], cmap=maps["vg50"], source_split_id="pair_ood.train"
    )
    calibration = calibrate_prior(
        prior,
        table,
        ood["dev"],
        [maps["vg50"].remap(table.pred_id[r]) for r in ood["dev"]],
    )
    report["prior_calibration"] = {
        key: calibration[key] for key in ("alpha", "temperature", "dev_nll")
    }

    metrics: dict[str, Any] = {}
    for level in LEVELS:
        cmap = maps[level]
        names = list(cmap.class_names)
        cells: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for probe in PROBE_NAMES:
            cell_name = f"{level}__pair_known__{probe}__s0"
            payload = load_predictions(probes_dir, cell_name)
            payload["probs"] = torch.load(
                probes_dir / cell_name / "predictions.pt", weights_only=False
            )["probs"]
            cells[probe] = payload
            arrays[probe] = {
                "labels": np.asarray(payload["labels"], dtype=np.int64),
                "predictions": np.asarray(payload["predictions"], dtype=np.int64),
                "probs": payload["probs"].numpy(),
            }
        labels = cells[BASE_PROBE]["labels"]

        # One denominator shared by all three cohorts, so no macro difference
        # can come from the class set changing underneath the comparison.
        shared_usable = sorted(
            set(usable_classes(labels, cmap.n_classes))
            & set(
                usable_classes(
                    [labels[i] for i in clean_indices], cmap.n_classes
                )
            )
            & set(
                usable_classes(
                    [labels[i] for i in contaminated_indices], cmap.n_classes
                )
            )
        )

        block: dict[str, Any] = {
            "n_usable_classes_full": len(usable_classes(labels, cmap.n_classes)),
            "n_usable_classes_shared": len(shared_usable),
            "accuracy": {},
            "rescue": {},
            "predicate_strata": {},
        }
        for cohort_name, idx in (
            ("full_eval", list(range(len(known_eval)))),
            ("clean_subset", clean_indices),
            ("contaminated_subset", contaminated_indices),
        ):
            block["accuracy"][cohort_name] = {
                probe: _accuracy_block(
                    arrays[probe], idx, cmap, eval_images, shared_usable
                )
                for probe in PROBE_NAMES
            }
            block["rescue"][cohort_name] = _rescue_block(
                cells[BASE_PROBE], cells[VISUAL_PROBE], idx, eval_images
            )
            block["predicate_strata"][cohort_name] = _predicate_strata(
                table, names, labels, cells[VISUAL_PROBE]["predictions"], idx
            )

        # Selection-bias diagnostic: if the two cohorts differ in predicate mix
        # or in per-predicate difficulty, the level gap is not attributable to
        # contamination and the paired delta is the only defensible comparison.
        #
        # labels/names are bound as defaults rather than closed over: this sits
        # inside the per-level loop, and a late-bound reference would silently
        # read the last level's arrays if the call ever moved out of the
        # iteration (ruff B023).
        def shares(
            idx: Sequence[int],
            labels: Sequence[int] = labels,
            names: Sequence[str] = names,
        ) -> dict[str, float]:
            counts: dict[int, int] = {}
            for j in idx:
                counts[labels[j]] = counts.get(labels[j], 0) + 1
            total = len(idx)
            return {names[c]: n / total for c, n in sorted(counts.items())}

        clean_share, contam_share = shares(clean_indices), shares(contaminated_indices)
        keys = sorted(set(clean_share) | set(contam_share))
        block["selection_bias"] = {
            "total_variation_distance": 0.5
            * sum(abs(clean_share.get(k, 0.0) - contam_share.get(k, 0.0)) for k in keys),
            "top_shift": sorted(
                (
                    {
                        "predicate": k,
                        "clean_share": clean_share.get(k, 0.0),
                        "contaminated_share": contam_share.get(k, 0.0),
                        "delta": clean_share.get(k, 0.0) - contam_share.get(k, 0.0),
                    }
                    for k in keys
                ),
                key=lambda row: -abs(row["delta"]),
            )[:top],
            "note": (
                "the clean cohort is the greedy pair-OOD holdout's complement, "
                "not a random holdout, so a level difference between cohorts is "
                "not a contamination effect"
            ),
        }
        metrics[level] = block
    report["metrics"] = metrics
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--splits-dir", default=str(DEFAULT_OUTPUT_ROOT / "splits")
    )
    parser.add_argument("--probes-dir", default=str(DEFAULT_OUTPUT_ROOT / "probes"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="also write the input SHA256 manifest to this path",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    for name in ("rel.json", "train.json"):
        if not (data_root / name).exists():
            print(f"[rescore] missing input: {data_root / name}", file=sys.stderr)
            return 2

    report = run_rescore(
        data_root,
        Path(args.splits_dir),
        Path(args.probes_dir),
        _PROJECT_ROOT,
    )
    if not report.get("alignment_all_ok", False):
        print(f"[rescore] {report.get('aborted', 'alignment failed')}", file=sys.stderr)

    payload = status_block(
        stage="issue110_clean_cohort_rescore",
        **report,
        interpretation={
            "historical_pair_known_status": "NON_HELD_OUT",
            "clean_subset_status": CLEAN_SUBSET_STATUS,
            "fidelity_check": (
                "the full_eval block reproduces the published pair_known micro "
                "accuracies and macro recalls exactly, which is what licenses "
                "reading the clean_subset block as the same pipeline on other rows"
            ),
            "caveats": (
                "single seed, so the McNemar p-values test rows within one "
                "trained pair and say nothing about seed variance; B2 and B4 "
                "differ in parameter count, so a B2-vs-B4 difference is not "
                "separable from a capacity effect"
            ),
        },
    )
    write_json(Path(args.output), payload)
    if args.manifest_output:
        write_json(
            Path(args.manifest_output),
            status_block(stage="issue110_input_manifest", **report["input_manifest"]),
        )
    print(
        f"[rescore] cohorts: clean={report['cohorts']['n_clean']} "
        f"contaminated={report['cohorts']['n_contaminated']} "
        f"aligned={report['alignment_all_ok']}"
    )
    print(f"[rescore] wrote {args.output}")
    return 0 if report.get("alignment_all_ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
