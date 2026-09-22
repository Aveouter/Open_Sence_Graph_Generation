"""M8: when the semantic prior is wrong, can vision fix it?

The average ``Δ_V`` answers "how many points does vision add". This answers the
question that actually matters for a relation model:

    given a relation where prior+geometry is wrong, how often does vision
    recover the correct predicate?

with a harder variant restricted to cases where B2 was **confidently** wrong
(``p_B2(ŷ) > τ``).  That is the direct test of "can visual evidence overturn a
high-confidence but incorrect semantic prior", and unlike the strict pair-OOD
prior-conflict set it has real support: measured on IID val and the relaxed
split, where the pair is seen, 38.8% and 43.8% of relations have a prior that
is confidently wrong.  On strict pair-OOD the corresponding set is empty.

The counterweight matters as much as the headline: ``harm_rate`` is how often
B4 *breaks* a case B2 already got right.  A model that rescues 5% while harming
5% has not demonstrated usable visual evidence, and reporting rescue alone
would hide that.

Stdlib only apart from loading the saved tensors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.canonical_map import level_map
from tools.ontology_probe.common import (
    DEFAULT_OUTPUT_ROOT,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.eval_matrix import usable_classes
from tools.ontology_probe.probe_metrics import (
    mcnemar_test,
    paired_accuracy_diff,
)

CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9)


def load_predictions(probes_dir: Path, cell: str) -> dict[str, Any]:
    payload = torch.load(probes_dir / cell / "predictions.pt", weights_only=False)
    probs = payload["probs"]
    labels = payload["targets"].tolist()
    predictions = probs.argmax(dim=-1).tolist()
    confidence = probs.max(dim=-1).values.tolist()
    return {
        "cell": cell,
        "rows": list(payload["rows"]),
        "labels": labels,
        "predictions": predictions,
        "confidence": confidence,
    }


def align(*cells: dict[str, Any]) -> None:
    """Refuse to compare cells that were not scored on the same relations."""
    reference = cells[0]
    for other in cells[1:]:
        if other["rows"] != reference["rows"]:
            raise AssertionError(
                f"{other['cell']} was evaluated on different relations than "
                f"{reference['cell']}; rescue rates would be meaningless"
            )
        if other["labels"] != reference["labels"]:
            raise AssertionError(
                f"{other['cell']} has different labels than {reference['cell']}"
            )


def _rate(mask: Sequence[bool], correct: Sequence[bool]) -> float:
    selected = [i for i, flag in enumerate(mask) if flag]
    if not selected:
        return float("nan")
    return sum(1 for i in selected if correct[i]) / len(selected)


def rescue_analysis(
    base: dict[str, Any],
    visual: dict[str, Any],
    cmap,
    thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS,
) -> dict[str, Any]:
    labels = base["labels"]
    base_correct = [p == t for p, t in zip(base["predictions"], labels, strict=True)]
    visual_correct = [p == t for p, t in zip(visual["predictions"], labels, strict=True)]

    base_error = [not c for c in base_correct]
    base_right = list(base_correct)

    classes = usable_classes(labels, cmap.n_classes)
    chance = 1.0 / len(classes) if classes else float("nan")

    result: dict[str, Any] = {
        "base_cell": base["cell"],
        "visual_cell": visual["cell"],
        "n_eval": len(labels),
        "n_usable_classes": len(classes),
        "chance_accuracy": chance,
        "base_accuracy": sum(base_correct) / len(labels),
        "visual_accuracy": sum(visual_correct) / len(labels),
        "n_base_wrong": sum(base_error),
        # VRR over every relation the prior got wrong.
        "vrr": _rate(base_error, visual_correct),
        # The counterweight: cases the prior had right that vision breaks.
        "harm_rate": _rate(base_right, [not c for c in visual_correct]),
        "net_rescue": (
            sum(1 for i in range(len(labels)) if base_error[i] and visual_correct[i])
            - sum(1 for i in range(len(labels)) if base_right[i] and not visual_correct[i])
        )
        / len(labels),
    }
    result["delta_accuracy"] = paired_accuracy_diff(
        [int(c) for c in base_correct], [int(c) for c in visual_correct]
    )
    result["mcnemar"] = mcnemar_test(
        [int(c) for c in base_correct], [int(c) for c in visual_correct]
    )

    by_threshold = {}
    for tau in thresholds:
        # "confidently wrong": the prior was both incorrect and sure.
        mask = [
            base_error[i] and base["confidence"][i] > tau for i in range(len(labels))
        ]
        selected = sum(1 for flag in mask if flag)
        by_threshold[f"tau_{tau}"] = {
            "n_relations": selected,
            "vrr": _rate(mask, visual_correct),
            "interpretation": (
                "fraction of high-confidence prior errors that visual evidence "
                "recovers; n_relations near zero means the subset is too small "
                "to support a claim"
            ),
        }
    result["by_confidence"] = by_threshold
    result["mcnemar_base_wrong_subset"] = mcnemar_test(
        [int(base_correct[i]) for i in range(len(labels)) if base_error[i]],
        [int(visual_correct[i]) for i in range(len(labels)) if base_error[i]],
    )
    return result


def per_predicate_rescue(
    base: dict[str, Any],
    visual: dict[str, Any],
    cmap,
    min_support: int = 30,
) -> list[dict[str, Any]]:
    labels = base["labels"]
    base_correct = [p == t for p, t in zip(base["predictions"], labels, strict=True)]
    visual_correct = [p == t for p, t in zip(visual["predictions"], labels, strict=True)]

    rows: list[dict[str, Any]] = []
    for class_id in sorted(set(labels)):
        members = [i for i in range(len(labels)) if labels[i] == class_id]
        errors = [i for i in members if not base_correct[i]]
        rows.append(
            {
                "predicate": cmap.class_names[class_id],
                "support": len(members),
                "base_recall": sum(base_correct[i] for i in members) / len(members),
                "visual_recall": sum(visual_correct[i] for i in members) / len(members),
                "n_base_wrong": len(errors),
                "vrr": (
                    sum(1 for i in errors if visual_correct[i]) / len(errors)
                    if errors
                    else float("nan")
                ),
                "reliable": len(errors) >= min_support,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes-dir", default=str(DEFAULT_OUTPUT_ROOT / "probes"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "rescue"))
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="cells as BASE:VISUAL, e.g. vg50__iid__B2__s0:vg50__iid__B4__s0",
    )
    parser.add_argument("--level", default=None, help="defaults to the base cell's level")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probes_dir = Path(args.probes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    predicate_rows = []
    for spec in args.pairs:
        base_cell, _, visual_cell = spec.partition(":")
        if not visual_cell:
            raise ValueError(f"--pairs entries must be BASE:VISUAL, got {spec!r}")
        base = load_predictions(probes_dir, base_cell)
        visual = load_predictions(probes_dir, visual_cell)
        align(base, visual)
        level = args.level or base_cell.split("__")[0]
        cmap = level_map(level)
        result = rescue_analysis(base, visual, cmap)
        results.append(result)
        predicate_rows.extend(
            {"pair": spec, **row} for row in per_predicate_rescue(base, visual, cmap)
        )
        print(
            f"[rescue] {base_cell} -> {visual_cell}: "
            f"VRR={result['vrr']:.4f} harm={result['harm_rate']:.4f} "
            f"net={result['net_rescue']:+.4f} "
            f"(B2 wrong on {result['n_base_wrong']}/{result['n_eval']})"
        )
        for key, value in result["by_confidence"].items():
            print(
                f"[rescue]   {key}: n={value['n_relations']:5d} "
                f"VRR={value['vrr']:.4f}"
            )

    write_json(output_dir / "rescue_results.json", status_block(results=results))
    write_csv(
        output_dir / "rescue_summary.csv",
        [
            "base_cell", "visual_cell", "n_eval", "base_accuracy", "visual_accuracy",
            "n_base_wrong", "vrr", "harm_rate", "net_rescue", "chance_accuracy",
        ],
        [
            {
                "base_cell": r["base_cell"],
                "visual_cell": r["visual_cell"],
                "n_eval": r["n_eval"],
                "base_accuracy": r["base_accuracy"],
                "visual_accuracy": r["visual_accuracy"],
                "n_base_wrong": r["n_base_wrong"],
                "vrr": r["vrr"],
                "harm_rate": r["harm_rate"],
                "net_rescue": r["net_rescue"],
                "chance_accuracy": r["chance_accuracy"],
            }
            for r in results
        ],
    )
    if predicate_rows:
        write_csv(
            output_dir / "rescue_by_predicate.csv",
            list(predicate_rows[0].keys()),
            predicate_rows,
        )
    print(f"[rescue] wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
