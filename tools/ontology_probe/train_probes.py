"""Train and evaluate one probe cell: {label space} x {split} x {probe}.

Every cell uses an identical optimisation protocol -- same loss, optimiser,
batch size, epoch budget, and early-stopping criterion -- because a difference
in *training* between label spaces would be indistinguishable from an ontology
effect.  ``selected_epoch`` is recorded per cell and reported so a systematic
divergence stays visible rather than hidden inside an average.

Usage::

    python tools/ontology_probe/train_probes.py --data-root data/VisualGenome \
        --probes B1_add B2 --levels vg50 L2_entail --splits iid pair_ood
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.canonical_map import (
    CanonicalMap,
    identity_map,
    load_canonical_map,
)
from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    hbt_group,
    status_block,
    write_json,
)
from tools.ontology_probe.eval_matrix import usable_classes
from tools.ontology_probe.probe_metrics import summarise
from tools.ontology_probe.probe_models import (
    PROBE_INPUTS,
    build_probe,
    geometry_from_boxes,
)
from tools.ontology_probe.splits import load_split
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    VGIndex,
    build_relation_table,
    load_vg_index,
    sub_object_boxes,
)

LEVELS = ("vg50", "L1_noise", "L2_entail")

DEFAULT_PROBE_CONFIG: dict[str, Any] = {
    "hidden": [512, 256],
    "dropout": 0.1,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 8192,
    "max_epochs": 100,
    "patience": 5,
    "seeds": [0, 1, 2],
    "monitor": "dev_nll",
}


@dataclass
class FeatureBundle:
    """Everything one probe needs for one set of relations."""

    labels_s: torch.Tensor
    labels_o: torch.Tensor
    geometry: torch.Tensor
    targets: torch.Tensor
    visual: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.targets.shape[0]

    def as_model_input(self) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {
            "labels_s": self.labels_s,
            "labels_o": self.labels_o,
            "geometry": self.geometry,
        }
        if self.visual is not None:
            payload.update(self.visual)
        return payload

    def to(self, device: torch.device) -> FeatureBundle:
        return FeatureBundle(
            labels_s=self.labels_s.to(device),
            labels_o=self.labels_o.to(device),
            geometry=self.geometry.to(device),
            targets=self.targets.to(device),
            visual=(
                {k: v.to(device) for k, v in self.visual.items()}
                if self.visual is not None
                else None
            ),
        )


def build_bundle(
    table: RelationTable,
    rows: Sequence[int],
    index: VGIndex,
    cmap: CanonicalMap,
    visual: dict[str, torch.Tensor] | None = None,
) -> FeatureBundle:
    """Assemble label / geometry (and optionally visual) features for ``rows``."""
    labels_s = torch.tensor([table.c_s[r] for r in rows], dtype=torch.long)
    labels_o = torch.tensor([table.c_o[r] for r in rows], dtype=torch.long)
    targets = torch.tensor(
        [cmap.remap(table.pred_id[r]) for r in rows], dtype=torch.long
    )

    sub_boxes = torch.zeros(len(rows), 4)
    obj_boxes = torch.zeros(len(rows), 4)
    sizes = torch.zeros(len(rows), 2)
    for i, r in enumerate(rows):
        sub, obj, (height, width) = sub_object_boxes(index, table, r)
        sub_boxes[i] = torch.tensor(sub)
        obj_boxes[i] = torch.tensor(obj)
        sizes[i] = torch.tensor([float(height), float(width)])
    geometry = geometry_from_boxes(sub_boxes, obj_boxes, sizes)

    return FeatureBundle(
        labels_s=labels_s,
        labels_o=labels_o,
        geometry=geometry,
        targets=targets,
        visual=visual,
    )


def _needs_visual(probe: str, inputs: Sequence[str]) -> bool:
    return any(name.startswith("visual") for name in inputs)


def train_one(
    probe_name: str,
    cmap: CanonicalMap,
    train: FeatureBundle,
    dev: FeatureBundle,
    eval_bundle: FeatureBundle,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    # Whole-bundle transfer: these feature sets fit in VRAM (233k x ~2.4k fp16
    # is ~1 GB), so per-batch host->device copies would only add overhead.
    train = train.to(device)
    dev = dev.to(device)
    eval_bundle = eval_bundle.to(device)

    model = build_probe(
        probe_name,
        n_classes=cmap.n_classes,
        visual_dim=(
            next(iter(train.visual.values())).shape[-1] if train.visual else 768
        ),
        hidden=tuple(config["hidden"]),
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_inputs = train.as_model_input()
    dev_inputs = dev.as_model_input()
    n = len(train)
    batch = config["batch_size"]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    best_dev_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    log: list[dict[str, Any]] = []

    for epoch in range(config["max_epochs"]):
        model.train()
        order = torch.randperm(n, generator=generator)
        total_loss = 0.0
        for start in range(0, n, batch):
            idx = order[start : start + batch].to(device)
            batch_inputs = {k: v[idx] for k, v in train_inputs.items()}
            logits = model(batch_inputs)
            loss = criterion(logits, train.targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * idx.shape[0]

        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_inputs)
            dev_nll = float(
                torch.nn.functional.cross_entropy(
                    dev_logits, dev.targets, reduction="mean"
                )
            )
        log.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "dev_nll": dev_nll,
            }
        )
        if dev_nll < best_dev_nll - 1e-6:
            best_dev_nll = dev_nll
            best_epoch = epoch
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        eval_logits = model(eval_bundle.as_model_input())
        eval_probs = torch.softmax(eval_logits, dim=-1).cpu()

    eval_labels = eval_bundle.targets.tolist()
    probs = eval_probs.tolist()
    classes = usable_classes(eval_labels, cmap.n_classes)
    report = summarise(
        eval_labels,
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
    report["predictor"] = model.describe()
    report["seed"] = seed

    return {
        "report": report,
        "train_log": log,
        "selected_epoch": best_epoch,
        "best_dev_nll": best_dev_nll,
        # If the cap was reached the run is truncated rather than converged, and
        # a cap that binds at different epochs for different label spaces would
        # masquerade as an ontology effect. Recorded so it can be checked.
        "hit_epoch_cap": len(log) >= config["max_epochs"],
        "probs": eval_probs,
        "targets": eval_bundle.targets,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def _load_visual(
    cache_root: Path | None, encoder: str, cache_split: str, rows: Sequence[int]
) -> dict[str, torch.Tensor] | None:
    """Load ``V_s, V_o, V_u`` from the cache for the split that owns ``rows``.

    ``cache_split`` is the *VG split* the rows come from (train or val), not the
    probe split name: pair_ood eval rows live in the train table, so they read
    the train cache, while IID rows read the val cache.
    """
    if cache_root is None:
        return None
    from tools.ontology_probe.feature_cache import load_visual_features, shard_dir

    directory = shard_dir(cache_root, encoder, cache_split)
    if not (directory / "shards").exists():
        raise FileNotFoundError(
            f"no visual cache at {directory}; run extract_features.py for "
            f"split={cache_split} first"
        )
    return load_visual_features(directory, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "probes"))
    parser.add_argument("--splits-dir", default=str(DEFAULT_OUTPUT_ROOT / "splits"))
    parser.add_argument("--probes", nargs="+", default=["B1_add", "B2"])
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--splits", nargs="+", default=["iid", "pair_ood"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None, help="root of the visual feature cache")
    parser.add_argument("--encoder", default="clip_vit_b16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = dict(DEFAULT_PROBE_CONFIG)
    if args.seeds:
        config["seeds"] = list(args.seeds)
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs

    print(f"[train_probes] device={device} config={ {k: config[k] for k in ('lr','batch_size','max_epochs','patience','seeds')} }")

    cache_root = Path(args.cache_dir) if args.cache_dir else None
    train_index = load_vg_index(data_root, "train", with_boxes=True)
    train_table = build_relation_table(train_index)
    val_index = load_vg_index(data_root, "val", with_boxes=True)
    val_table = build_relation_table(val_index)

    maps: dict[str, CanonicalMap] = {"vg50": identity_map(train_index.predicate_names)}
    for level in ("L1_noise", "L2_entail"):
        maps[level] = load_canonical_map(level=level)

    ood = load_split("pair_ood", args.splits_dir, table=train_table, verify=True)
    known = load_split("pair_known", args.splits_dir, table=train_table, verify=True)
    # (index, table, rows, cache_split) -- the cache split is the VG split the
    # relation rows belong to, which is not the probe split name.
    eval_sets = {
        "iid": (val_index, val_table, list(range(len(val_table))), "val"),
        "pair_ood": (train_index, train_table, ood.rows("eval"), "train"),
        "pair_known": (train_index, train_table, known.rows("eval"), "train"),
    }

    summaries: list[dict[str, Any]] = []
    for level in args.levels:
        cmap = maps[level]
        train_bundle = build_bundle(
            train_table,
            ood.rows("train"),
            train_index,
            cmap,
            visual=_load_visual(cache_root, args.encoder, "train", ood.rows("train")),
        )
        dev_bundle = build_bundle(
            train_table,
            ood.rows("dev"),
            train_index,
            cmap,
            visual=_load_visual(cache_root, args.encoder, "train", ood.rows("dev")),
        )
        for split_name in args.splits:
            index, table, rows, cache_split = eval_sets[split_name]
            eval_bundle = build_bundle(
                table,
                rows,
                index,
                cmap,
                visual=_load_visual(cache_root, args.encoder, cache_split, rows),
            )
            for probe in args.probes:
                if _needs_visual(probe, PROBE_INPUTS[probe]) and train_bundle.visual is None:
                    print(f"[train_probes] SKIP {probe}: needs a visual cache (--cache-dir)")
                    continue
                for seed in config["seeds"]:
                    result = train_one(
                        probe,
                        cmap,
                        train_bundle,
                        dev_bundle,
                        eval_bundle,
                        config,
                        seed,
                        device,
                    )
                    cell = f"{level}__{split_name}__{probe}__s{seed}"
                    cell_dir = output_dir / cell
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    report = result["report"]
                    write_json(
                        cell_dir / "metrics.json",
                        status_block(
                            cell=cell,
                            level=level,
                            split=split_name,
                            probe=probe,
                            seed=seed,
                            selected_epoch=result["selected_epoch"],
                            best_dev_nll=result["best_dev_nll"],
                            n_epochs_run=len(result["train_log"]),
                            hit_epoch_cap=result["hit_epoch_cap"],
                            config=config,
                            metrics=report,
                            training_subset="pair_ood.train",
                        ),
                    )
                    write_json(
                        cell_dir / "train_log.json",
                        status_block(cell=cell, log=result["train_log"]),
                    )
                    torch.save(
                        {
                            "probs": result["probs"],
                            "targets": result["targets"],
                            "rows": list(rows),
                        },
                        cell_dir / "predictions.pt",
                    )
                    # Weights are kept so the intervention stage (M7 shuffle,
                    # M8 rescue) can perturb a *fixed* trained probe instead of
                    # retraining, which is what isolates the visual evidence
                    # from any change in the optimisation trajectory.
                    torch.save(
                        {
                            "state_dict": result["state_dict"],
                            "probe": probe,
                            "level": level,
                            "n_classes": cmap.n_classes,
                            "visual_dim": (
                                next(iter(train_bundle.visual.values())).shape[-1]
                                if train_bundle.visual
                                else None
                            ),
                            "hidden": config["hidden"],
                            "dropout": config["dropout"],
                            "rows": list(rows),
                            "cache_split": cache_split,
                            "encoder": args.encoder,
                        },
                        cell_dir / "model.pt",
                    )
                    # Non-visual inputs are stored so the intervention stage can
                    # re-run a fixed model while permuting only the visual
                    # blocks, without rebuilding labels/geometry from scratch
                    # (and without risking a subtly different reconstruction).
                    torch.save(
                        {
                            "labels_s": eval_bundle.labels_s.cpu(),
                            "labels_o": eval_bundle.labels_o.cpu(),
                            "geometry": eval_bundle.geometry.cpu(),
                        },
                        cell_dir / "inputs.pt",
                    )
                    summaries.append(
                        {
                            "cell": cell,
                            "level": level,
                            "split": split_name,
                            "probe": probe,
                            "seed": seed,
                            "selected_epoch": result["selected_epoch"],
                            "n_epochs_run": len(result["train_log"]),
                            "hit_epoch_cap": result["hit_epoch_cap"],
                            "accuracy": report["accuracy"],
                            "macro_recall": report["macro_recall"],
                            "nll": report["nll"],
                            "mR@1": report["mR@1"],
                            "n_usable_classes": report["n_usable_classes"],
                        }
                    )
                    print(
                        f"[train_probes] {cell:44s} ep={result['selected_epoch']:3d} "
                        f"acc={report['accuracy']:.4f} macro={report['macro_recall']:.4f} "
                        f"nll={report['nll']:.4f}"
                    )

    write_json(
        output_dir / "probe_summary.json",
        status_block(
            config=config,
            controlled_training_subset="pair_ood.train",
            cells=summaries,
        ),
    )
    print(f"[train_probes] wrote {len(summaries)} cells to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
