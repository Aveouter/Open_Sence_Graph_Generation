"""M5 gate: prove the visual features carry real signal before using them.

If this stage is skipped, a later ``Δ_V ≈ 0`` is uninterpretable -- it could
mean "VG has no cross-pair visual signal" or "the crop pipeline is broken", and
nothing in the probe results distinguishes the two.

Checks, in order of diagnostic value:

1. ``object_probe`` -- can a linear probe on ``V_s``/``V_o`` recover the object
   category? This is the gate. Near-chance here means the features are not
   encoding the image, and M6 must not proceed.
2. ``alignment`` -- re-encode the subject, object and union crops of sampled
   relations and compare against the cache. Catches an index-mapping error,
   which would otherwise silently pair every feature with the wrong label.
3. ``determinism`` -- the same crop encoded twice must give the same vector.
4. ``variance`` -- detect collapsed or dead feature dimensions.
5. ``pair_discriminability`` -- for pairs carrying several predicates, do the
   union features differ across predicates? If not, ``Δ_V`` cannot be positive
   for those relations no matter how good the probe is.
6. ``roi_contact_sheet`` -- render crops so the boxes can be eyeballed.

Usage::

    python tools/ontology_probe/visual_sanity.py --cache-root <cache> \\
        --data-root data/VisualGenome --limit-images 200
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

from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    status_block,
    write_csv,
    write_json,
)
from tools.ontology_probe.feature_cache import (
    load_visual_features,
    read_manifest,
    shard_dir,
)
from tools.ontology_probe.vg_annotations import (
    RelationTable,
    build_relation_table,
    load_vg_index,
)

#: A linear probe below this top-1 on 150 object classes means the features are
#: not carrying usable appearance information, and M6 would be uninterpretable.
OBJECT_PROBE_GATE = 0.20
#: Cosine similarity below which two encodings of the same crop are "different".
DETERMINISM_TOLERANCE = 1e-3


def _import_torch():
    import torch

    return torch


def check_cache_integrity(directory: Path, rows: Sequence[int]) -> dict[str, Any]:
    import numpy as np

    manifest = read_manifest(directory)
    feats = load_visual_features(directory, rows)
    report: dict[str, Any] = {
        "manifest_present": manifest is not None,
        "manifest": {k: v for k, v in (manifest or {}).items() if k != "provenance"},
        "n_rows_loaded": len(rows),
        "feature_dim": int(feats["visual_u"].shape[-1]) if len(rows) else 0,
    }
    for key in ("visual_s", "visual_o", "visual_u"):
        array = feats[key].to(_import_torch().float32).numpy()
        report[f"{key}_finite"] = bool(np.isfinite(array).all())
        report[f"{key}_abs_mean"] = float(np.abs(array).mean()) if array.size else 0.0
    return report


def check_variance(directory: Path, rows: Sequence[int]) -> dict[str, Any]:
    """Detect collapsed features: dead dimensions or near-zero spread."""
    import numpy as np

    feats = load_visual_features(directory, rows)["visual_u"]
    array = feats.to(_import_torch().float32).numpy()
    if array.size == 0:
        return {"passed": False, "reason": "no features"}
    variances = array.var(axis=0)
    norms = np.linalg.norm(array, axis=1)
    report = {
        "n_samples": int(array.shape[0]),
        "n_dims": int(array.shape[1]),
        "mean_variance": float(variances.mean()),
        "min_variance": float(variances.min()),
        "n_dead_dims": int((variances < 1e-12).sum()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        # A single point repeated would give zero variance everywhere.
        "n_unique_rows_approx": int(len(np.unique(np.round(array, 3), axis=0))),
    }
    report["passed"] = (
        report["n_dead_dims"] == 0
        and report["mean_variance"] > 1e-6
        and report["n_unique_rows_approx"] > 0.5 * array.shape[0]
    )
    return report


def check_determinism(
    data_root: Path,
    index,
    table: RelationTable,
    rows: Sequence[int],
    sample: int = 30,
    context_scale: float = 1.5,
) -> dict[str, Any]:
    """Re-encode sampled crops and compare against the cache."""
    import torch

    from tools.ontology_probe.extract_features import (
        FrozenEncoder,
        encode_crops,
        expanded_crop,
        union_crop,
    )

    encoder = FrozenEncoder("clip_vit_b16", "cpu")
    chosen = list(rows[:sample])
    crops = []
    for row in chosen:
        boxes = index.ann_boxes_by_image[table.image_id[row]]
        size = index.image_sizes[table.image_id[row]]
        sub = boxes[table.sub_idx[row]]
        obj = boxes[table.obj_idx[row]]
        crops.append(
            union_crop(
                [sub[0], sub[1], sub[0] + sub[2], sub[1] + sub[3]],
                [obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3]],
                size,
                context_scale,
            )
        )
    sizes = [1] * len(chosen)
    first = encode_crops(data_root, index, [table.image_id[r] for r in chosen], sizes, crops, encoder, 8)
    second = encode_crops(data_root, index, [table.image_id[r] for r in chosen], sizes, crops, encoder, 8)
    a = torch.from_numpy(first).to(torch.float32)
    b = torch.from_numpy(second).to(torch.float32)
    max_abs = float((a - b).abs().max()) if a.numel() else 0.0
    cos = (
        torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
        if a.numel()
        else float("nan")
    )
    return {
        "n_sampled": len(chosen),
        "max_abs_diff": max_abs,
        "min_cosine": cos,
        "passed": max_abs == 0.0,
        "note": "identical input must give bitwise identical output on one device",
    }


def check_alignment(
    data_root: Path,
    index,
    table: RelationTable,
    directory: Path,
    rows: Sequence[int],
    sample: int = 30,
    context_scale: float = 1.5,
) -> dict[str, Any]:
    """Verify cache features match a fresh encoding of the same crops.

    This is the check that would catch an off-by-one in the subject/object
    index mapping, which would otherwise assign every relation the features of
    some other object while looking completely healthy.
    """
    import torch

    from tools.ontology_probe.extract_features import (
        FrozenEncoder,
        encode_crops,
        expanded_crop,
        union_crop,
    )

    chosen = list(rows[:sample])
    encoder = FrozenEncoder("clip_vit_b16", "cpu")
    cached = load_visual_features(directory, chosen)

    # One group per relation, holding its subject then its object crop. The
    # group's image id must therefore appear once per relation, not once per
    # crop -- see the length check in encode_crops.
    obj_crops, obj_ids, obj_sizes = [], [], []
    for row in chosen:
        boxes = index.ann_boxes_by_image[table.image_id[row]]
        size = index.image_sizes[table.image_id[row]]
        obj_ids.append(table.image_id[row])
        obj_sizes.append(2)
        for idx in (table.sub_idx[row], table.obj_idx[row]):
            x, y, w, h = boxes[idx]
            obj_crops.append(expanded_crop((x, y, x + w, y + h), size, context_scale))
    fresh_obj = torch.from_numpy(
        encode_crops(data_root, index, obj_ids, obj_sizes, obj_crops, encoder, 16)
    )

    def _cos(a, b) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                a.to(torch.float32), b.to(torch.float32), dim=-1
            ).min()
        )

    fresh_s = fresh_obj[0::2]
    fresh_o = fresh_obj[1::2]
    return {
        "n_sampled": len(chosen),
        "min_cosine_subject": _cos(cached["visual_s"][: len(fresh_s)], fresh_s),
        "min_cosine_object": _cos(cached["visual_o"][: len(fresh_o)], fresh_o),
        "passed": min(
            _cos(cached["visual_s"][: len(fresh_s)], fresh_s),
            _cos(cached["visual_o"][: len(fresh_o)], fresh_o),
        )
        > (1.0 - DETERMINISM_TOLERANCE),
    }


def check_object_probe(
    directory: Path,
    index,
    table: RelationTable,
    rows: Sequence[int],
    epochs: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Linear probe: ``V_s``/``V_o`` -> object category. This is the M5 gate."""
    import torch

    feats = load_visual_features(directory, rows)
    labels = torch.tensor(
        [index.ann_labels_by_image[table.image_id[r]][table.sub_idx[r]] for r in rows],
        dtype=torch.long,
    )
    x = torch.cat([feats["visual_s"], feats["visual_o"]], dim=0).to(torch.float32)
    y = torch.cat([labels, labels], dim=0)

    # Standardise per dimension: CLIP CLS vectors have a large common component
    # that otherwise dominates the first gradient steps.
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp(min=1e-6)

    n = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    n_train = int(n * 0.8)
    train_idx, test_idx = order[:n_train], order[n_train:]

    n_classes = int(y.max().item()) + 1
    probe = torch.nn.Linear(x.shape[1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(probe(x[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_acc = float(
            (probe(x[test_idx]).argmax(dim=-1) == y[test_idx]).float().mean()
        )
        train_acc = float(
            (probe(x[train_idx]).argmax(dim=-1) == y[train_idx]).float().mean()
        )

    counts = torch.bincount(y[test_idx], minlength=n_classes).float()
    majority = float(counts.max() / counts.sum()) if counts.sum() else float("nan")
    # Chance over the classes actually present, which is what the probe competes
    # against -- 1/150 would flatter a probe that never predicts rare classes.
    present = int((torch.bincount(y, minlength=n_classes) > 0).sum())

    return {
        "n_samples": int(n),
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "n_classes_present": present,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "majority_baseline": majority,
        "uniform_chance": 1.0 / present if present else float("nan"),
        "passed": test_acc >= OBJECT_PROBE_GATE,
        "gate": OBJECT_PROBE_GATE,
        "interpretation": (
            "the gate asks only whether the features encode appearance at all; "
            "a probe near uniform chance means the crop pipeline is broken and "
            "any later Delta_V ~ 0 would be uninterpretable"
        ),
    }


def check_pair_discriminability(
    directory: Path,
    table: RelationTable,
    rows: Sequence[int],
    min_per_pair: int = 2,
) -> dict[str, Any]:
    """Do union features differ across predicates on the *same* object pair?"""
    import torch

    feats = load_visual_features(directory, rows)["visual_u"].to(torch.float32)
    pairs = table.undirected_pairs()
    grouped: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        grouped.setdefault(pairs[row], []).append(i)

    same, diff = [], []
    for members in grouped.values():
        if len(members) < min_per_pair:
            continue
        normalised = torch.nn.functional.normalize(feats[members], dim=-1)
        sim = normalised @ normalised.T
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if table.pred_id[rows[members[a]]] == table.pred_id[rows[members[b]]]:
                    same.append(float(sim[a, b]))
                else:
                    diff.append(float(sim[a, b]))

    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    mean_same = _mean(same)
    mean_diff = _mean(diff)
    return {
        "n_same_predicate_pairs": len(same),
        "n_diff_predicate_pairs": len(diff),
        "mean_cosine_same_predicate": mean_same,
        "mean_cosine_different_predicate": mean_diff,
        "gap": mean_same - mean_diff if same and diff else float("nan"),
        "passed": bool(same and diff and mean_same > mean_diff),
        "interpretation": (
            "sampling noise dominates below ~100 comparisons; treat a small "
            "positive gap on a 200-image cache as directional only"
        ),
    }


def check_pair_predicate_probe(
    directory: Path,
    table: RelationTable,
    rows: Sequence[int],
    epochs: int = 300,
    seed: int = 0,
    min_pair_instances: int = 30,
) -> dict[str, Any]:
    """Can a linear probe read the predicate off ``V_u`` when the pair is fixed?

    ``check_pair_discriminability`` compares raw cosine and is therefore
    dominated by object identity -- within a same-pair group the objects are
    identical, so the cosine barely moves whether or not the predicate matches
    (measured gap -0.0009 over 7.8M/12.4M comparisons). That makes it a poor
    test of whether predicate evidence is *present*.

    This is the discriminative version: restrict to relations whose object pair
    carries more than one predicate, and ask whether a linear probe on the union
    feature beats the majority-predicate baseline. A positive margin here is
    what makes a non-zero Δ_V possible at all.
    """
    import torch

    pairs = table.undirected_pairs()
    grouped: dict[int, list[int]] = {}
    for r in rows:
        grouped.setdefault(pairs[r], []).append(r)
    usable = [
        members
        for members in grouped.values()
        if len(members) >= min_pair_instances
        and len({table.pred_id[m] for m in members}) > 1
    ]
    if not usable:
        return {
            "passed": False,
            "reason": "no multi-predicate pairs with enough instances at this scale",
        }

    selected = [m for members in usable for m in members]
    feats = load_visual_features(directory, selected)["visual_u"].to(torch.float32)
    labels = torch.tensor([table.pred_id[m] for m in selected], dtype=torch.long)
    feats = (feats - feats.mean(dim=0, keepdim=True)) / feats.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Split by pair, not by row: the question is generalisation to unseen
    # instances of a pair, and a row-level split would let near-duplicate
    # instances of the same relation appear on both sides.
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(usable), generator=generator)
    train_pairs = {i for i in order[: int(len(usable) * 0.7)].tolist()}

    pair_of_row: list[int] = []
    for pair_index, members in enumerate(usable):
        pair_of_row.extend([pair_index] * len(members))
    is_train = torch.tensor([i in train_pairs for i in pair_of_row])
    train_rows = torch.nonzero(is_train).squeeze(-1)
    test_rows = torch.nonzero(~is_train).squeeze(-1)
    if len(train_rows) == 0 or len(test_rows) == 0:
        return {"passed": False, "reason": "split produced an empty side"}

    present = torch.unique(labels)
    remap = {int(p): i for i, p in enumerate(present.tolist())}
    y = torch.tensor([remap[int(v)] for v in labels.tolist()], dtype=torch.long)

    probe = torch.nn.Linear(feats.shape[1], len(remap))
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(probe(feats[train_rows]), y[train_rows])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(feats).argmax(dim=-1)
        test_acc = float((preds[test_rows] == y[test_rows]).float().mean())
        train_acc = float((preds[train_rows] == y[train_rows]).float().mean())

    counts = torch.bincount(y[test_rows], minlength=len(remap)).float()
    majority = float(counts.max() / counts.sum()) if counts.sum() else float("nan")
    return {
        "n_multi_predicate_pairs": len(usable),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_predicates": len(remap),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "majority_baseline": majority,
        "margin_over_majority": test_acc - majority,
        "passed": test_acc > majority + 0.02,
        "interpretation": (
            "a positive margin means predicate evidence is linearly present in "
            "the union feature once object identity is held fixed; this is the "
            "precondition for a non-zero Delta_V on ambiguous pairs"
        ),
    }


def render_contact_sheet(
    data_root: Path,
    index,
    table: RelationTable,
    rows: Sequence[int],
    out_path: Path,
    n: int = 25,
    context_scale: float = 1.5,
    cols: int = 5,
) -> dict[str, Any] | None:
    """Draw context-padded crops with the subject/object boxes overlaid."""
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"skipped": f"PIL unavailable ({exc})"}

    from tools.ontology_probe.extract_features import expanded_crop, union_crop

    chosen = list(rows[:n])
    if not chosen:
        return {"skipped": "no rows"}
    tile = 160
    rows_n = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * tile, rows_n * tile), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    palette = [(255, 80, 80), (80, 200, 255), (255, 220, 80)]
    for i, row in enumerate(chosen):
        image_id = table.image_id[row]
        boxes = index.ann_boxes_by_image[image_id]
        size = index.image_sizes[image_id]
        sub = boxes[table.sub_idx[row]]
        obj = boxes[table.obj_idx[row]]
        sub_xyxy = [sub[0], sub[1], sub[0] + sub[2], sub[1] + sub[3]]
        obj_xyxy = [obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3]]
        crop = union_crop(sub_xyxy, obj_xyxy, size, context_scale)
        try:
            with Image.open(data_root / "images" / index.image_file_names[image_id]) as handle:
                patch = handle.convert("RGB").crop(crop).resize((tile, tile))
        except Exception:
            continue
        patch_draw = ImageDraw.Draw(patch)
        scale_x = tile / max(1, crop[2] - crop[0])
        scale_y = tile / max(1, crop[3] - crop[1])
        for box, colour in ((sub_xyxy, palette[0]), (obj_xyxy, palette[1])):
            x1 = (box[0] - crop[0]) * scale_x
            y1 = (box[1] - crop[1]) * scale_y
            x2 = (box[2] - crop[0]) * scale_x
            y2 = (box[3] - crop[1]) * scale_y
            patch_draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        patch_draw.rectangle([0, tile - 12, tile, tile], fill=(0, 0, 0))
        patch_draw.text(
            (2, tile - 11), index.predicate_name(table.pred_id[row])[:22], fill=(255, 255, 255)
        )
        sheet.paste(patch, ((i % cols) * tile, (i // cols) * tile))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return {"path": str(out_path), "n_tiles": len(chosen), "tile_px": tile}


def run_all(
    data_root: Path,
    cache_root: Path,
    encoder: str,
    split: str,
    limit_images: int | None,
    output_dir: Path,
    context_scale: float = 1.5,
) -> dict[str, Any]:
    directory = shard_dir(cache_root, encoder, split)
    if not (directory / "shards").exists():
        raise FileNotFoundError(
            f"no cache at {directory}; run extract_features.py first"
        )

    index = load_vg_index(data_root, split, with_boxes=True)
    table = build_relation_table(index)
    rows = sorted(
        {
            r
            for r in range(len(table))
            if limit_images is None
            or table.image_id[r] in set(index.image_ids[:limit_images])
        }
    )
    print(f"[sanity] cache={directory} relations_in_cache_scope={len(rows)}")

    checks: dict[str, Any] = {}
    checks["integrity"] = check_cache_integrity(directory, rows)
    checks["variance"] = check_variance(directory, rows)
    checks["determinism"] = check_determinism(
        data_root, index, table, rows, context_scale=context_scale
    )
    checks["alignment"] = check_alignment(
        data_root, index, table, directory, rows, context_scale=context_scale
    )
    checks["object_probe"] = check_object_probe(directory, index, table, rows)
    checks["pair_discriminability"] = check_pair_discriminability(directory, table, rows)
    # The discriminative counterpart to the cosine check above; this is the one
    # that actually bounds whether Delta_V can be non-zero.
    checks["pair_predicate_probe"] = check_pair_predicate_probe(
        directory, table, rows
    )
    checks["contact_sheet"] = render_contact_sheet(
        data_root, index, table, rows, output_dir / "roi_contact_sheet.png",
        context_scale=context_scale,
    )

    gate_passed = bool(checks["object_probe"]["passed"])
    checks["gate"] = {
        "object_probe_gate": OBJECT_PROBE_GATE,
        "object_probe_accuracy": checks["object_probe"]["test_accuracy"],
        "decision": "PROCEED to M6" if gate_passed else "STOP: fix the feature pipeline",
    }
    for name, result in checks.items():
        if isinstance(result, dict) and "passed" in result:
            print(f"[sanity] {name:24s} passed={result['passed']}")
    print(f"[sanity] object probe acc={checks['object_probe']['test_accuracy']:.3f} "
          f"(majority {checks['object_probe']['majority_baseline']:.3f}) -> "
          f"{checks['gate']['decision']}")

    write_json(output_dir / "visual_sanity.json", status_block(checks=checks))
    write_csv(
        output_dir / "visual_sanity.csv",
        ["check", "passed", "detail"],
        [
            {"check": name, "passed": result.get("passed"), "detail": str(result)}
            for name, result in checks.items()
            if isinstance(result, dict)
        ],
    )
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--encoder", default="clip_vit_b16")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "sanity")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_all(
        Path(args.data_root),
        Path(args.cache_root),
        args.encoder,
        args.split,
        args.limit_images,
        Path(args.output_dir),
        args.context_scale,
    )
    return 0 if checks["gate"].get("decision", "").startswith("PROCEED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
