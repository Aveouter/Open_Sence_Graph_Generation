"""M7: visual intervention. Does B4 actually use the visual correspondence?

``B4 > B2`` alone does not establish that vision helps: B4 has more parameters,
so it could win by capacity. The intervention removes that ambiguity, because
shuffling visual features keeps the architecture and parameter count **exactly**
the same and only destroys which feature belongs to which relation.

So the capacity-controlled comparison is ``B4`` vs ``B4_shuffled``; ``B4`` vs
``B2`` answers the different question "what does vision add on top of semantics".

Permutations are applied only to visual tensors. ``(c_s, c_o)``, geometry, the
label and the target are asserted unchanged, so any difference is attributable
to the correspondence and nothing else.

Two groupings bracket the effect:

* ``global`` -- shuffle across the whole eval set. Visual evidence becomes not
  just uninformative but actively misleading, since a feature now carries the
  appearance of an unrelated relation.
* ``within_predicate`` -- shuffle only among rows sharing the ground-truth
  predicate. This preserves ``P(V | r)`` and destroys only the instance link,
  which isolates "did the model use *this* image" from "did it learn the
  predicate's typical appearance".

Usage::

    python tools/ontology_probe/shuffle_test.py --probes-dir <probes> \\
        --cells L2_entail__pair_ood__B4__s0 --variants union sub_obj all
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
    hbt_group,
    read_json,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.eval_matrix import usable_classes
from tools.ontology_probe.probe_metrics import (
    mcnemar_test,
    paired_accuracy_diff,
    summarise,
)
from tools.ontology_probe.probe_models import build_probe

VISUAL_KEYS = ("visual_s", "visual_o", "visual_u")
#: variant name -> which visual blocks it permutes
VARIANTS: dict[str, tuple[str, ...]] = {
    "union": ("visual_u",),
    "sub_obj": ("visual_s", "visual_o"),
    "all": VISUAL_KEYS,
}


def permutation(n: int, seed: int, groups: Sequence[int] | None = None) -> torch.Tensor:
    """A permutation of ``range(n)``; with ``groups``, permutes within groups.

    Groups are keyed by label, so ``within_predicate`` keeps every row's
    predicate-conditioned feature distribution intact and destroys only the
    pairing between a row and its own image.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.arange(n)
    if groups is None:
        return order[torch.randperm(n, generator=generator)]
    result = order.clone()
    buckets: dict[int, list[int]] = {}
    for i, group in enumerate(groups):
        buckets.setdefault(int(group), []).append(i)
    for members in buckets.values():
        if len(members) < 2:
            continue
        permuted = torch.tensor(members)[
            torch.randperm(len(members), generator=generator)
        ]
        for target, source in zip(members, permuted.tolist(), strict=True):
            result[target] = source
    return result


def load_cell(probes_dir: Path, cell: str) -> dict[str, Any]:
    model_payload = torch.load(probes_dir / cell / "model.pt", weights_only=False)
    predictions = torch.load(
        probes_dir / cell / "predictions.pt", weights_only=False
    )
    inputs = torch.load(probes_dir / cell / "inputs.pt", weights_only=False)
    metrics = read_json(probes_dir / cell / "metrics.json")
    return {
        "model": model_payload,
        "predictions": predictions,
        "inputs": inputs,
        "metrics": metrics,
    }


def rebuild_model(payload: dict[str, Any]) -> torch.nn.Module:
    model = build_probe(
        payload["probe"],
        n_classes=payload["n_classes"],
        visual_dim=payload["visual_dim"] or 768,
        hidden=tuple(payload["hidden"]),
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate_with_features(
    model: torch.nn.Module,
    features: dict[str, torch.Tensor],
    labels: Sequence[int],
    cmap,
) -> dict[str, Any]:
    """Score a fixed model on a (possibly permuted) feature dict."""
    with torch.no_grad():
        logits = model({k: v for k, v in features.items()})
        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1).tolist()
    classes = usable_classes(labels, cmap.n_classes)
    report = summarise(
        labels,
        probs.tolist(),
        classes=classes,
        class_names=list(cmap.class_names),
        class_to_group={
            cmap.class_names[i]: hbt_group(min(cmap.members[i]))
            for i in range(cmap.n_classes)
        },
    )
    report["n_usable_classes"] = len(classes)
    report["predictions"] = predictions
    return report


def run_shuffle(
    probes_dir: Path,
    cache_root: Path,
    encoder: str,
    cache_split: str,
    cell: str,
    variants: Sequence[str],
    groupings: Sequence[str],
    n_repeats: int,
    seed: int,
) -> dict[str, Any]:
    from tools.ontology_probe.train_probes import _load_visual

    payload = load_cell(probes_dir, cell)
    level = payload["model"]["level"]
    cmap = level_map(level)
    model = rebuild_model(payload["model"])

    rows = list(payload["predictions"]["rows"])
    labels = payload["predictions"]["targets"].tolist()
    visual = _load_visual(cache_root, encoder, cache_split, rows)
    if visual is None:
        raise ValueError(f"no visual cache for split {cache_split}")

    inputs = payload["inputs"]
    features = {
        "labels_s": inputs["labels_s"],
        "labels_o": inputs["labels_o"],
        "geometry": inputs["geometry"],
        **visual,
    }

    base = evaluate_with_features(model, features, labels, cmap)
    base_correct = [int(p == t) for p, t in zip(base["predictions"], labels, strict=True)]

    results: list[dict[str, Any]] = []
    for variant in variants:
        keys = VARIANTS[variant]
        for grouping in groupings:
            groups = labels if grouping == "within_predicate" else None
            for repeat in range(n_repeats):
                perm = permutation(len(labels), seed + 1000 * repeat, groups)
                shuffled = dict(features)
                for key in keys:
                    shuffled[key] = features[key][perm]
                # The intervention must touch vision and nothing else: every
                # non-permuted block has to be the same tensor object, so a
                # difference can only come from the correspondence.
                for key in VARIANTS["all"]:
                    if key not in keys:
                        assert shuffled[key] is features[key], (
                            f"{key} must be untouched by the {variant} variant"
                        )
                for key in ("labels_s", "labels_o", "geometry"):
                    assert shuffled[key] is features[key], (
                        f"{key} must never be permuted"
                    )
                report = evaluate_with_features(model, shuffled, labels, cmap)
                correct = [
                    int(p == t) for p, t in zip(report["predictions"], labels, strict=True)
                ]
                delta = paired_accuracy_diff(base_correct, correct)
                results.append(
                    {
                        "variant": variant,
                        "grouping": grouping,
                        "repeat": repeat,
                        "seed": seed + 1000 * repeat,
                        "macro_recall": report["macro_recall"],
                        "accuracy": report["accuracy"],
                        "nll": report["nll"],
                        "delta_accuracy": delta["diff"],
                        "delta_accuracy_lo": delta["lo"],
                        "delta_accuracy_hi": delta["hi"],
                        "mcnemar_p": mcnemar_test(base_correct, correct)["p_value"],
                        "predictions": report["predictions"],
                    }
                )

    # Per-predicate shuffle drop, averaged over repeats, for the M10 table.
    by_predicate: list[dict[str, Any]] = []
    for class_id, class_name in sorted(
        ((k, cmap.class_names[k]) for k in sorted(set(labels)))
    ):
        members = [i for i in range(len(labels)) if labels[i] == class_id]
        if not members:
            continue
        base_hits = sum(base_correct[i] for i in members) / len(members)
        for variant in variants:
            repeated = [
                r
                for r in results
                if r["variant"] == variant and r["grouping"] == "global"
            ]
            if not repeated:
                continue
            shuffled_hits = sum(
                sum(1 for i in members if r["predictions"][i] == labels[i])
                / len(members)
                for r in repeated
            ) / len(repeated)
            by_predicate.append(
                {
                    "predicate": class_name,
                    "support": len(members),
                    "variant": variant,
                    "base_recall": base_hits,
                    "shuffled_recall": shuffled_hits,
                    "shuffle_drop": base_hits - shuffled_hits,
                }
            )

    summary: dict[str, Any] = {
        "cell": cell,
        "level": level,
        "n_eval": len(labels),
        "baseline": {
            "accuracy": base["accuracy"],
            "macro_recall": base["macro_recall"],
            "nll": base["nll"],
        },
        "runs": [
            {k: v for k, v in r.items() if k != "predictions"} for r in results
        ],
        "by_predicate": by_predicate,
    }
    # Aggregate per (variant, grouping).
    aggregate = {}
    for variant in variants:
        for grouping in groupings:
            chosen = [
                r for r in results if r["variant"] == variant and r["grouping"] == grouping
            ]
            if not chosen:
                continue
            drops = [r["delta_accuracy"] for r in chosen]
            macro = [r["macro_recall"] for r in chosen]
            aggregate[f"{variant}__{grouping}"] = {
                "mean_delta_accuracy": sum(drops) / len(drops),
                "min_delta_accuracy": min(drops),
                "max_delta_accuracy": max(drops),
                "mean_macro_recall": sum(macro) / len(macro),
                "n_repeats": len(chosen),
                "all_mcnemar_p_below_0.05": all(
                    r["mcnemar_p"] < 0.05 for r in chosen
                ),
            }
    summary["aggregate"] = aggregate
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes-dir", default=str(DEFAULT_OUTPUT_ROOT / "probes"))
    parser.add_argument("--cache-root", default=str(DEFAULT_OUTPUT_ROOT / "cache"))
    parser.add_argument("--encoder", default="clip_vit_b16")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "shuffle"))
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument(
        "--cache-split", default=None, help="defaults to the cell's probe split"
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument(
        "--groupings", nargs="+", default=["global", "within_predicate"]
    )
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for cell in args.cells:
        # cell id is {level}__{split}__{probe}__s{seed}
        parts = cell.split("__")
        probe_split = parts[1] if len(parts) > 1 else "pair_ood"
        cache_split = args.cache_split or ("val" if probe_split == "iid" else "train")
        summary = run_shuffle(
            Path(args.probes_dir),
            Path(args.cache_root),
            args.encoder,
            cache_split,
            cell,
            args.variants,
            args.groupings,
            args.n_repeats,
            args.seed,
        )
        summaries.append(summary)
        print(f"[shuffle] {cell}")
        for key, value in summary["aggregate"].items():
            print(
                f"[shuffle]   {key:34s} Δacc={value['mean_delta_accuracy']:+.4f} "
                f"[{value['min_delta_accuracy']:+.4f},{value['max_delta_accuracy']:+.4f}] "
                f"macro={value['mean_macro_recall']:.4f} "
                f"p<0.05={value['all_mcnemar_p_below_0.05']}"
            )

    write_json(
        output_dir / "shuffle_results.json",
        status_block(encoder=args.encoder, cells=summaries),
    )
    rows = [
        {"cell": s["cell"], **run}
        for s in summaries
        for run in s["runs"]
    ]
    if rows:
        write_csv(
            output_dir / "shuffle_runs.csv",
            list(rows[0].keys()),
            rows,
        )
    predicate_rows = [
        {"cell": s["cell"], **row} for s in summaries for row in s["by_predicate"]
    ]
    if predicate_rows:
        write_csv(
            output_dir / "shuffle_by_predicate.csv",
            list(predicate_rows[0].keys()),
            predicate_rows,
        )
    print(f"[shuffle] wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
