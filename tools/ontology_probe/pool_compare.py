"""Δ_ontology via pooled probabilities, plus the touched/untouched decomposition.

Comparing macro metrics across label spaces is the easiest way to fool yourself
in this experiment.  A VG50 model scores over 27 usable classes, an L2 model
over 23; three of the VG50 classes are the L2 class ``on``, and the usable-class
filters then differ too.  Averaging each model over its own class set compares
two different partitions, not two ontologies.

So every cross-space number here is computed by **pooling the finer space's
probabilities onto the coarser space's classes** (``p_canon(C) = sum_{r in C}
p_fine(r)``, exact for both argmax and NLL since it is a plain sum over disjoint
groups) and macro-averaging both models over the *same* class set.

The decomposition is the honesty check.  A relation whose VG50 predicate is
inside a merged class is "touched" -- for those, the label is redefined, so part
of any gain is near-tautological re-labelling.  Relations whose predicate is a
singleton in the coarse space are "untouched": ``canon(pred) == canon(gt)`` iff
``pred == gt``, so a gain there is genuine transfer to classes nobody edited.
That untouched number is the headline; the touched number is reported beside it
and never quoted alone.

Stdlib only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from tools.ontology_probe.canonical_map import CanonicalMap, load_canonical_map
from tools.ontology_probe.common import (
    DEFAULT_OUTPUT_ROOT,
    hbt_group,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.eval_matrix import usable_classes
from tools.ontology_probe.probe_metrics import (
    mcnemar_test,
    paired_accuracy_diff,
    paired_mean_diff,
    summarise,
)

PAIRS_OF_INTEREST = (
    ("vg50", "L1_noise"),
    ("vg50", "L2_entail"),
    ("L1_noise", "L2_entail"),
)


def _load_cell(probes_dir: Path, cell: str) -> dict[str, Any]:
    payload = torch.load(probes_dir / cell / "predictions.pt", weights_only=False)
    return {
        "probs": payload["probs"],
        "targets": payload["targets"].tolist(),
        "rows": payload["rows"],
    }


def fine_predicates_from_vg50_targets(targets: Sequence[int]) -> list[int]:
    """Recover raw predicate ids from a vg50 cell's class indices.

    ``identity_map`` numbers classes 0..49 for predicates 1..50, so the inverse
    is a shift.  Stated as a function because getting it wrong is a silent
    off-by-one that would mislabel every pooled comparison.
    """
    return [t + 1 for t in targets]


def _reports_for(
    labels: Sequence[int],
    probs: Sequence[Sequence[float]],
    cmap: CanonicalMap,
) -> dict[str, Any]:
    classes = usable_classes(labels, cmap.n_classes)
    report = summarise(
        labels,
        probs,
        classes=classes,
        class_names=list(cmap.class_names),
        class_to_group={
            cmap.class_names[i]: hbt_group(min(cmap.members[i]))
            for i in range(cmap.n_classes)
        },
    )
    report["n_usable_classes"] = len(classes)
    report["n_total_classes"] = cmap.n_classes
    report["predictions"] = [
        max(range(len(row)), key=lambda c: row[c]) for row in probs
    ]
    report["labels"] = list(labels)
    return report


def compare_space_pair(
    fine_probs: Sequence[Sequence[float]],
    fine_predicate_ids: Sequence[int],
    coarse_probs: Sequence[Sequence[float]],
    coarse_targets: Sequence[int],
    cmap: CanonicalMap,
    level: str,
    split: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a VG50 model pooled to ``cmap`` against a model trained in ``cmap``."""
    pooled_probs = [cmap.pool_probs(row) for row in fine_probs]
    labels = [cmap.remap(p) for p in fine_predicate_ids]

    # Alignment guard: the pooled VG50 labels must reproduce the canonical
    # model's own targets row for row, or the two cells were not evaluated on
    # the same relations in the same order and nothing below is comparable.
    if labels != list(coarse_targets):
        mismatches = sum(1 for a, b in zip(labels, coarse_targets, strict=True) if a != b)
        raise AssertionError(
            f"cell alignment broken for {level}/{split}: {mismatches} label "
            "mismatches between pooled-vg50 and native canonical targets"
        )

    pooled = _reports_for(labels, pooled_probs, cmap)
    native = _reports_for(labels, coarse_probs, cmap)

    pooled_correct = [int(p == t) for p, t in zip(pooled["predictions"], labels, strict=True)]
    native_correct = [int(p == t) for p, t in zip(native["predictions"], labels, strict=True)]
    acc_diff = paired_accuracy_diff(pooled_correct, native_correct)
    mcnemar = mcnemar_test(pooled_correct, native_correct)

    nll_diff = paired_mean_diff(
        [-math.log(max(row[t], 1e-12)) for row, t in zip(pooled_probs, labels, strict=True)],
        [-math.log(max(row[t], 1e-12)) for row, t in zip(coarse_probs, labels, strict=True)],
    )

    # touched/untouched split
    merged = cmap.merged_fine_ids()
    touched_idx = [i for i, p in enumerate(fine_predicate_ids) if p in merged]
    untouched_idx = [i for i, p in enumerate(fine_predicate_ids) if p not in merged]

    def _subset_accuracy(indices: Sequence[int], preds: Sequence[int]) -> float:
        if not indices:
            return float("nan")
        return sum(1 for i in indices if preds[i] == labels[i]) / len(indices)

    decomposition = {
        "n_touched": len(touched_idx),
        "n_untouched": len(untouched_idx),
        "pooled_accuracy_touched": _subset_accuracy(touched_idx, pooled["predictions"]),
        "native_accuracy_touched": _subset_accuracy(touched_idx, native["predictions"]),
        "pooled_accuracy_untouched": _subset_accuracy(untouched_idx, pooled["predictions"]),
        "native_accuracy_untouched": _subset_accuracy(untouched_idx, native["predictions"]),
        "delta_untouched": (
            _subset_accuracy(untouched_idx, pooled["predictions"])
            - _subset_accuracy(untouched_idx, native["predictions"])
        ),
        "delta_touched": (
            _subset_accuracy(touched_idx, pooled["predictions"])
            - _subset_accuracy(touched_idx, native["predictions"])
        ),
        "interpretation": (
            "delta_untouched is genuine transfer to classes no mapping edited; "
            "delta_touched is largely re-labelling and must not be quoted alone"
        ),
    }
    if untouched_idx:
        decomposition["untouched_mcnemar"] = mcnemar_test(
            [pooled_correct[i] for i in untouched_idx],
            [native_correct[i] for i in untouched_idx],
        )

    return {
        "level": level,
        "split": split,
        "comparison": f"pooled_vg50_vs_native_{level}",
        "n_eval": len(labels),
        "n_usable_classes": pooled["n_usable_classes"],
        "pooled": {
            "accuracy": pooled["accuracy"],
            "macro_recall": pooled["macro_recall"],
            "nll": pooled["nll"],
            "mR@1": pooled["mR@1"],
        },
        "native": {
            "accuracy": native["accuracy"],
            "macro_recall": native["macro_recall"],
            "nll": native["nll"],
            "mR@1": native["mR@1"],
        },
        "delta_accuracy": acc_diff,
        "delta_nll": nll_diff,
        "mcnemar": mcnemar,
        "decomposition": decomposition,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes-dir", default=str(DEFAULT_OUTPUT_ROOT / "probes"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "matrix"))
    parser.add_argument("--probe", default="B2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--splits", nargs="+", default=["iid", "pair_ood", "pair_known"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probes_dir = Path(args.probes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = {level: load_canonical_map(level=level) for level in ("L1_noise", "L2_entail")}

    results: list[dict[str, Any]] = []
    for split in args.splits:
        vg50_cell = f"vg50__{split}__{args.probe}__s{args.seed}"
        if not (probes_dir / vg50_cell).exists():
            print(f"[pool_compare] SKIP {split}: {vg50_cell} not found")
            continue
        base = _load_cell(probes_dir, vg50_cell)
        fine_ids = fine_predicates_from_vg50_targets(base["targets"])

        for level in ("L1_noise", "L2_entail"):
            cell = f"{level}__{split}__{args.probe}__s{args.seed}"
            if not (probes_dir / cell).exists():
                print(f"[pool_compare] SKIP {cell}: not found")
                continue
            coarse = _load_cell(probes_dir, cell)
            result = compare_space_pair(
                base["probs"],
                fine_ids,
                coarse["probs"],
                coarse["targets"],
                maps[level],
                level,
                split,
                seed=args.seed,
            )
            results.append(result)
            d = result["delta_accuracy"]
            dec = result["decomposition"]
            print(
                f"[pool_compare] {args.probe} {split:11s} Δ_ontology({level:9s}) "
                f"acc={d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
                f"p={result['mcnemar']['p_value']:.2e} | "
                f"untouched Δ={dec['delta_untouched']:+.4f} (n={dec['n_untouched']}), "
                f"touched Δ={dec['delta_touched']:+.4f} (n={dec['n_touched']})"
            )

    write_json(
        output_dir / f"pool_compare_{args.probe}_s{args.seed}.json",
        status_block(probe=args.probe, seed=args.seed, comparisons=results),
    )
    write_csv(
        output_dir / f"pool_compare_{args.probe}_s{args.seed}.csv",
        [
            "probe", "split", "level", "n_eval", "n_usable_classes",
            "pooled_accuracy", "native_accuracy", "delta_accuracy",
            "delta_accuracy_lo", "delta_accuracy_hi", "mcnemar_p",
            "pooled_macro", "native_macro", "pooled_nll", "native_nll",
            "n_touched", "n_untouched", "delta_untouched", "delta_touched",
        ],
        [
            {
                "probe": args.probe,
                "split": r["split"],
                "level": r["level"],
                "n_eval": r["n_eval"],
                "n_usable_classes": r["n_usable_classes"],
                "pooled_accuracy": r["pooled"]["accuracy"],
                "native_accuracy": r["native"]["accuracy"],
                "delta_accuracy": r["delta_accuracy"]["diff"],
                "delta_accuracy_lo": r["delta_accuracy"]["lo"],
                "delta_accuracy_hi": r["delta_accuracy"]["hi"],
                "mcnemar_p": r["mcnemar"]["p_value"],
                "pooled_macro": r["pooled"]["macro_recall"],
                "native_macro": r["native"]["macro_recall"],
                "pooled_nll": r["pooled"]["nll"],
                "native_nll": r["native"]["nll"],
                "n_touched": r["decomposition"]["n_touched"],
                "n_untouched": r["decomposition"]["n_untouched"],
                "delta_untouched": r["decomposition"]["delta_untouched"],
                "delta_touched": r["decomposition"]["delta_touched"],
            }
            for r in results
        ],
    )
    print(f"[pool_compare] wrote {len(results)} comparisons to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
