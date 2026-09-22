"""M5/M6: extract frozen visual features into a sharded cache.

Crops are encoded directly (crop-and-encode) rather than ROI-pooled from a
patch grid: CLIP ViT-B/16 at 224 gives a 14x14 grid, and a small VG object
covers so few patches that ROI pooling would discard most of the evidence that
this experiment is trying to measure.

Calibrate before committing GPU hours::

    # 200-image dry run that measures throughput and projects the full cost
    python tools/ontology_probe/extract_features.py --measure --limit-images 200

The full run is refused unless the projection is under ``--max-projected-hours``
(or ``--accept-projection`` is passed), because a mis-estimated 12-hour job is
much worse than a two-minute calibration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ontology_probe.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SAMPLE_ROOT,
    status_block,
    write_json,
)
from tools.ontology_probe.feature_cache import (
    build_manifest,
    is_shard_done,
    plan_shards,
    write_shard,
)
from tools.ontology_probe.vg_annotations import (
    build_relation_table,
    load_vg_index,
)

#: CLIP's own preprocessing constants, pinned here so the cache is reproducible
#: without depending on a downloaded preprocessor config.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 224

DEFAULT_ENCODER = "clip_vit_b16"
ENCODER_REPOS = {
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_b32": "openai/clip-vit-base-patch32",
}


def expanded_crop(
    box_xyxy: Sequence[float],
    image_size: tuple[int, int],
    context_scale: float,
) -> tuple[int, int, int, int]:
    """Grow a box about its centre by ``context_scale`` and clamp to the image.

    Some context around the box matters for relations that are defined by
    contact or support (``on``, ``holding``) -- a tight crop of a hand and a cup
    looks the same whether the cup is held or resting on the table.
    """
    height, width = image_size
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * context_scale / 2.0
    half_h = (y2 - y1) * context_scale / 2.0
    return (
        max(0, int(round(cx - half_w))),
        max(0, int(round(cy - half_h))),
        min(width, int(round(cx + half_w))),
        min(height, int(round(cy + half_h))),
    )


def union_crop(
    sub_xyxy: Sequence[float],
    obj_xyxy: Sequence[float],
    image_size: tuple[int, int],
    context_scale: float,
) -> tuple[int, int, int, int]:
    """Context-padded crop of the box containing both subject and object."""
    x1 = min(sub_xyxy[0], obj_xyxy[0])
    y1 = min(sub_xyxy[1], obj_xyxy[1])
    x2 = max(sub_xyxy[2], obj_xyxy[2])
    y2 = max(sub_xyxy[3], obj_xyxy[3])
    return expanded_crop((x1, y1, x2, y2), image_size, context_scale)


class FrozenEncoder:
    """A frozen image encoder returning one pooled feature vector per crop."""

    def __init__(self, name: str, device: str) -> None:
        import torch

        self.name = name
        self.torch = torch
        self.device = torch.device(device)
        if name not in ENCODER_REPOS:
            raise ValueError(f"unknown encoder {name!r}; known: {sorted(ENCODER_REPOS)}")
        self.repo = ENCODER_REPOS[name]

        if name.startswith("clip_"):
            from transformers import CLIPVisionModel

            self.model = CLIPVisionModel.from_pretrained(self.repo)
        else:  # pragma: no cover - reserved for a timm back end
            raise ValueError(f"unsupported encoder {name!r}")
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.dim = int(self.model.config.hidden_size)

    def encode(self, images) -> "Any":
        """``images`` is a float32 tensor [B, 3, 224, 224] already normalised."""
        import torch

        with torch.inference_mode():
            outputs = self.model(pixel_values=images.to(self.device))
            # pooled_output is CLIP's own CLS projection target; the raw CLS
            # hidden state is what the repo's FlowSG encoder also exposes.
            return outputs.pooler_output.float().cpu()


def load_image_crops(
    path: Path,
    crops: Sequence[tuple[int, int, int, int]],
):
    """Decode one JPEG once and return all its crop tensors.

    Decoding is the dominant CPU cost (~6 ms per image), and a VG image
    contributes ~12 crops on average, so opening the file once per crop rather
    than once per image multiplies that cost by the number of crops.
    """
    import numpy as np
    import torch
    from PIL import Image

    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    out = []
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        for x1, y1, x2, y2 in crops:
            if x2 <= x1 or y2 <= y1:
                x2, y2 = x1 + 1, y1 + 1
            patch = image.crop((x1, y1, x2, y2))
            patch = patch.resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
            array = np.asarray(patch, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            out.append((tensor - mean) / std)
    return out


def extract_split(
    data_root: Path,
    directory: Path,
    encoder: FrozenEncoder,
    split: str,
    context_scale: float,
    shard_images: int,
    batch_size: int,
    limit_images: int | None,
    workers: int = 4,
    log_every: int = 10,
) -> dict[str, Any]:
    """Extract one split's features into ``directory``, resuming finished shards."""
    import numpy as np
    import torch

    index = load_vg_index(data_root, split, with_boxes=True)
    table = build_relation_table(index)

    rel_rows_by_image: dict[int, list[int]] = {}
    for row in range(len(table)):
        rel_rows_by_image.setdefault(table.image_id[row], []).append(row)

    image_ids = index.image_ids if limit_images is None else index.image_ids[:limit_images]
    shards = plan_shards(image_ids, shard_images)
    print(
        f"[extract] split={split} images={len(image_ids)} shards={len(shards)} "
        f"encoder={encoder.name} dim={encoder.dim}"
    )

    started = time.time()
    n_crops = 0
    peak_vram_gb = 0.0
    skipped = 0
    for shard_id, shard_images_ids in enumerate(shards):
        if is_shard_done(directory, shard_id):
            skipped += 1
            continue

        object_rows: list[tuple[int, int]] = []
        crops: list[tuple[int, int, int, int]] = []
        for image_id in shard_images_ids:
            labels = index.ann_labels_by_image.get(image_id, [])
            boxes = index.ann_boxes_by_image.get(image_id, [])
            size = index.image_sizes[image_id]
            for obj_idx in range(len(labels)):
                x, y, w, h = boxes[obj_idx]
                object_rows.append((image_id, obj_idx))
                crops.append(
                    expanded_crop((x, y, x + w, y + h), size, context_scale)
                )

        rel_rows: list[int] = []
        rel_sub_obj: list[tuple[int, int]] = []
        rel_crops: list[tuple[int, int, int, int]] = []
        local_of = {key: i for i, key in enumerate(object_rows)}
        for image_id in shard_images_ids:
            boxes = index.ann_boxes_by_image.get(image_id, [])
            size = index.image_sizes[image_id]
            for row in rel_rows_by_image.get(image_id, []):
                sub_xyxy = table_box(boxes, table.sub_idx[row])
                obj_xyxy = table_box(boxes, table.obj_idx[row])
                rel_rows.append(row)
                rel_sub_obj.append(
                    (
                        local_of[(image_id, table.sub_idx[row])],
                        local_of[(image_id, table.obj_idx[row])],
                    )
                )
                rel_crops.append(union_crop(sub_xyxy, obj_xyxy, size, context_scale))

        obj_sizes = [len(index.ann_labels_by_image.get(i, [])) for i in shard_images_ids]
        rel_sizes = [len(rel_rows_by_image.get(i, [])) for i in shard_images_ids]

        obj_feats = encode_crops(
            data_root, index, shard_images_ids, obj_sizes, crops, encoder, batch_size, workers
        )
        rel_feats = encode_crops(
            data_root, index, shard_images_ids, rel_sizes, rel_crops, encoder, batch_size, workers
        )
        n_crops += int(obj_feats.shape[0]) + int(rel_feats.shape[0])

        write_shard(
            directory,
            shard_id,
            np.asarray([r[0] for r in object_rows], dtype=np.int64),
            np.asarray([r[1] for r in object_rows], dtype=np.int32),
            obj_feats,
            np.asarray(rel_rows, dtype=np.int64),
            np.asarray(rel_sub_obj, dtype=np.int32).reshape(-1, 2),
            rel_feats,
            meta={"context_scale": context_scale, "encoder": encoder.name},
        )
        if (shard_id + 1) % log_every == 0 or shard_id == len(shards) - 1:
            elapsed = time.time() - started
            done = shard_id + 1 - skipped
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (len(shards) - shard_id - 1) / rate if rate else float("inf")
            print(
                f"[extract]   shard {shard_id + 1}/{len(shards)} "
                f"objects={len(object_rows)} relations={len(rel_rows)} "
                f"crops={n_crops} {rate:.2f} shard/s eta={remaining / 60:.1f} min"
            )

    if torch.cuda.is_available():
        peak_vram_gb = max(
            peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9
        )
    elapsed = time.time() - started
    return {
        "split": split,
        "images": len(image_ids),
        "shards": len(shards),
        "shards_skipped": skipped,
        "crops": n_crops,
        "elapsed_seconds": elapsed,
        "images_per_second": len(image_ids) / elapsed if elapsed else None,
        "peak_vram_gb": peak_vram_gb,
    }


def table_box(boxes: Sequence[Sequence[float]], idx: int) -> list[float]:
    """COCO [x, y, w, h] at positional ``idx`` -> absolute xyxy."""
    x, y, w, h = boxes[idx]
    return [x, y, x + w, y + h]


def encode_crops(
    data_root: Path | str,
    index: Any,
    image_ids: Sequence[int],
    sizes: Sequence[int],
    crops: Sequence[tuple[int, int, int, int]],
    encoder: FrozenEncoder,
    batch_size: int,
    workers: int = 4,
) -> Any:
    """Encode crops that were built in shard-image order, grouped by image.

    ``sizes[i]`` is how many crops ``image_ids[i]`` contributed, and they occupy
    ``crops[cursor : cursor + sizes[i]]`` -- so each JPEG is opened once and all
    of its crops are read before moving on.  Decoding per crop instead would
    re-read the same file up to ~12 times per image.

    Decoding runs on a small thread pool because it is CPU-bound (PIL releases
    the GIL) and the GPU sits idle during it otherwise; ``executor.map`` yields
    in order, so the feature order still matches ``crops`` exactly.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    data_root = Path(data_root)
    if not crops:
        return np.zeros((0, encoder.dim), dtype=np.float16)

    # Both checks are load-bearing. `zip` below would silently truncate to the
    # shorter of the two, and a caller that passes one image_id per *crop*
    # instead of per *group* would then cut later crops out of the wrong image
    # while producing a full-length, plausible-looking result.
    if len(image_ids) != len(sizes):
        raise ValueError(
            f"image_ids has {len(image_ids)} entries but sizes has {len(sizes)}; "
            "image_ids must hold one entry per group, not per crop"
        )
    if sum(sizes) != len(crops):
        raise ValueError(
            f"crop bookkeeping mismatch: sizes sum to {sum(sizes)} but there are "
            f"{len(crops)} crops; features would be misaligned with their objects"
        )

    tasks: list[tuple[int, Sequence[tuple[int, int, int, int]]]] = []
    cursor = 0
    for image_id, count in zip(image_ids, sizes, strict=True):
        image_crops = crops[cursor : cursor + count]
        cursor += count
        if image_crops:
            tasks.append((image_id, image_crops))

    def _decode(task):
        image_id, image_crops = task
        path = data_root / "images" / index.image_file_names[image_id]
        return load_image_crops(path, image_crops)

    features: list[Any] = []
    pending: list[Any] = []

    def _drain() -> None:
        nonlocal pending
        # Exactly batch_size per forward, so peak VRAM is a function of
        # --batch-size rather than of how many crops one image happens to have.
        while len(pending) >= batch_size:
            chunk, pending = pending[:batch_size], pending[batch_size:]
            features.append(encoder.encode(torch.stack(chunk)))

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for tensors in pool.map(_decode, tasks):
            pending.extend(tensors)
            _drain()
    finally:
        pool.shutdown()
    if pending:
        features.append(encoder.encode(torch.stack(pending)))

    stacked = torch.cat(features, dim=0).to(torch.float16).numpy()
    if stacked.shape[0] != len(crops):
        raise AssertionError(
            f"encoded {stacked.shape[0]} crops for {len(crops)} crop boxes"
        )
    return stacked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "cache"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--shard-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4, help="CPU decode threads")
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="only the first N images of each split (use 200 to calibrate)",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="write a throughput calibration and project the full-run cost",
    )
    parser.add_argument("--max-projected-hours", type=float, default=4.0)
    parser.add_argument("--accept-projection", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="use the 10-image sample")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    data_root = Path(SAMPLE_ROOT if args.dry_run else args.data_root)
    output_root = Path(args.output_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    encoder = FrozenEncoder(args.encoder, device)
    print(f"[extract] encoder={encoder.name} repo={encoder.repo} device={device}")

    from tools.ontology_probe.feature_cache import shard_dir as resolve_dir

    results = []
    for split in args.splits:
        directory = resolve_dir(output_root, args.encoder, split)
        result = extract_split(
            data_root,
            directory,
            encoder,
            split,
            args.context_scale,
            args.shard_images,
            args.batch_size,
            args.limit_images,
            args.workers,
        )
        result["cache_dir"] = str(directory)
        results.append(result)

        full_images = len(load_vg_index(data_root, split).image_ids)
        projected = None
        if result["images_per_second"]:
            projected = full_images / result["images_per_second"] / 3600.0
        result["full_split_images"] = full_images
        result["projected_full_hours"] = projected
        if projected is None:
            print(f"[extract] {split}: no throughput measurement")
        else:
            # `full_split_images` is the size of the data root actually read, so
            # a --dry-run against the 10-image fixture projects a meaningless
            # (tiny) number. Only a --limit-images run against the real root
            # gives a usable projection.
            basis = "" if args.limit_images else " (no limit applied)"
            print(
                f"[extract] {split}: {result['images']} images, "
                f"{result['crops']} crops, {result['elapsed_seconds']:.1f}s, "
                f"projected full split of {full_images} images = "
                f"{projected:.2f} h{basis}"
            )

        if (args.measure or args.limit_images) and projected is not None:
            if projected > args.max_projected_hours and not args.accept_projection:
                print(
                    f"[extract] REFUSING large run: projected {projected:.2f} h exceeds "
                    f"--max-projected-hours {args.max_projected_hours}. Re-run with "
                    f"--accept-projection to proceed anyway."
                )
                write_json(
                    Path(output_root) / f"calibration_{args.encoder}_{split}.json",
                    status_block(calibration=results),
                )
                return 3

    write_json(
        Path(output_root) / f"extraction_{args.encoder}.json",
        status_block(
            encoder=args.encoder,
            encoder_repo=encoder.repo,
            feature_dim=encoder.dim,
            device=device,
            context_scale=args.context_scale,
            batch_size=args.batch_size,
            limit_images=args.limit_images,
            runs=results,
        ),
    )
    if args.measure or args.limit_images:
        for result in results:
            build_manifest(
                Path(result["cache_dir"]),
                args.encoder,
                result["split"],
                result["images"],
                result["shards"],
                0,
                0,
                encoder.dim,
                {"context_scale": args.context_scale, "size": CLIP_SIZE},
                {"partial": True, "limit_images": args.limit_images},
            )
    print(f"[extract] wrote {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
