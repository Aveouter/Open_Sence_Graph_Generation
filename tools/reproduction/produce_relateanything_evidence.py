#!/usr/bin/env python3
"""Produce the external attestations the RelateAnything input gate consumes.

``check_relateanything_official_inputs.py`` verifies *supplied* attestations and
reruns nothing; this is the other half that runs the audit and writes the JSON.
It recomputes every hash from the files on disk, so a stale or copied attestation
cannot pass by accident.

Two independent phases, so a runtime without torch still produces the split
audit (and vice versa):

  --skip-load    do not run the strict-load phase
  --skip-split   do not run the full-split audit

An existing ``--output`` is merged, never silently replaced: re-running one
phase refreshes only the sections it recomputed, and the sections kept from the
previous file are named on stderr. Artifacts stay outside the repository
(reproduction/USAGE.md): pass external ``--output`` and evidence paths.

Exit 0 only when every section produced in this run satisfies the numeric
requirements the gate enforces; exit 2 otherwise, with the evidence still
written so the failure is inspectable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "benchmark/eval_zeroshot.py", "relsgg/checkpoint.py", "relsgg/eval/evaluator.py",
    "relsgg/data/dataset.py", "pyproject.toml", "LICENSE",
)
SNAPSHOT_FILES = ("model.pth", "text_student.pt", "tokenizer.json", "tokenizer_config.json")
PACK_FILES = ("meta.json", "file_names.json", "img_meta.npy", "boxes.npy", "box_cats.npy", "rels.npy")
CONSUMER_FILES = ("src/relateanything.py", "src/relateanything_cli.py", "src/relateanything_evaluation.py")
APACHE_SOURCE_COMMIT = "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
EXPECTED_IMAGES = 26404
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_revision(directory: Path) -> tuple[str | None, str | None]:
    """The commit a Hugging Face ``local_dir`` download was pinned to.

    ``snapshot_download`` records it as the first line of every
    ``.cache/huggingface/download/**/*.metadata`` sidecar; the value is read
    from the file rather than the Hub so it describes what is on disk.
    """
    for metadata in sorted((directory / ".cache" / "huggingface" / "download").rglob("*.metadata")):
        first = metadata.read_text(encoding="utf-8").splitlines()
        if first and FULL_REVISION.fullmatch(first[0].strip()):
            return first[0].strip(), str(metadata.relative_to(directory))
    return None, None


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


def strict_load_section(snapshot: Path, official_root: Path) -> tuple[dict, list[str]]:
    """Attest a strict load of the released checkpoint.

    The key/shape diff is computed before loading, against a model built by the
    loader's own code path, because ``load_state_dict`` raises on shape
    mismatches and would otherwise hide them behind one exception. The
    attestation itself is the loader's documented strict call, which must not
    raise. ``remapped_keys`` stays empty by definition: the loader resizes
    vocabulary-sized buffers to the checkpoint's shape rather than remapping
    keys, and those resizes are recorded separately.
    """
    sys.path.insert(0, str(REPO))
    from src.modules.relateanything.checkpoint import (
        OBSOLETE_KEYS,
        VOCAB_SIZED,
        build_model_from_ckpt,
        hub_id_for_backbone,
        load_checkpoint,
        materialize_backbone_config,
    )
    from src.modules.relateanything.config import config_from_args
    from src.modules.relateanything.model import RelSGG

    problems: list[str] = []
    ckpt = load_checkpoint(str(snapshot / "model.pth"))
    raw_args = ckpt["args"]
    args = dict(raw_args if isinstance(raw_args, dict) else vars(raw_args))
    offline = materialize_backbone_config(ckpt, args)
    if not offline:
        args["backbone_model"] = hub_id_for_backbone(args.get("backbone_model"))
    probe = RelSGG(config_from_args(args, backbone_pretrained=not offline))

    used_ema = bool(ckpt.get("ema_model"))
    sd = ckpt["ema_model"] if used_ema else ckpt["model"]
    obsolete_present = sorted(key for key in sd if key in OBSOLETE_KEYS)
    sd = {key: value for key, value in sd.items() if key not in OBSOLETE_KEYS}
    model_sd = probe.state_dict()
    missing = sorted(key for key in model_sd if key not in sd and key != "W_obj")
    unexpected = sorted(key for key in sd if key not in model_sd)
    shape_mismatches = sorted(
        key for key in sd
        if key in model_sd and tuple(sd[key].shape) != tuple(model_sd[key].shape)
        and key not in VOCAB_SIZED)
    resized = sorted(
        key for key in VOCAB_SIZED
        if key in sd and key in model_sd and tuple(sd[key].shape) != tuple(model_sd[key].shape))
    del probe

    build_model_from_ckpt(ckpt, weights="ema", strict=True)  # must not raise

    if missing or unexpected or shape_mismatches:
        problems.append(
            f"strict load diff is not empty: missing={missing[:6]} "
            f"unexpected={unexpected[:6]} shape_mismatches={shape_mismatches[:6]}")
    section = {
        "strict": True,
        "missing_keys": missing, "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches, "remapped_keys": [],
        "model_sha256": sha256_file(snapshot / "model.pth"),
        "text_student_sha256": sha256_file(snapshot / "text_student.pt"),
        "source_sha256": {name: sha256_file(official_root / name) for name in SOURCE_FILES},
        "loader": "src/modules/relateanything/checkpoint.py",
        "loader_sha256": sha256_file(REPO / "src/modules/relateanything/checkpoint.py"),
        "loader_upstream_commit": APACHE_SOURCE_COMMIT,
        "weights_requested": "ema", "used_ema_model": used_ema,
        "resized_vocab_buffers": resized, "obsolete_keys_present": obsolete_present,
        "w_obj_in_checkpoint": "W_obj" in ckpt["model"],
    }
    return section, problems


def probe_image(job: tuple[str, str]) -> tuple[str, bool, str]:
    path, name = job
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:  # any decode failure is a failed input, named below
        return name, False, f"{type(error).__name__}: {error}"[:160]
    return name, True, ""


def split_section(data_root: Path, workers: int, max_objects: int,
                  failure_samples: int) -> tuple[dict, list[str]]:
    """Decode every packed image and count the inputs the runner would score.

    The arrays are read directly with numpy rather than through
    ``RelationDataset``: the audit must not depend on the loader it is meant to
    check, and that module imports torch at module level, which no phase of the
    audit otherwise needs. When torch is importable the loader is consulted as
    an additional cross-check and its count recorded.

    ``valid_input_images`` counts images that decode and carry at least two
    boxes -- the minimum for a supplied-region relation. Zero-relation images
    are counted separately rather than folded into validity: they are valid
    inputs whose ground truth is empty, which the official evaluator skips and
    the OpenSGG adapter refuses (recorded here so that outcome is predicted,
    not discovered mid-run).
    """
    import numpy as np

    problems: list[str] = []
    split_dir = data_root / "test"
    meta = json.loads((split_dir / "meta.json").read_text(encoding="utf-8"))
    declared = meta["img_dir"]
    image_root = Path(declared).resolve()  # same resolution the loader performs at decode time
    if not image_root.is_dir():
        raise SystemExit(
            f"pack declares img_dir {declared!r}, which resolves to {image_root} from "
            f"cwd {os.getcwd()} and does not exist; run from the directory the declared "
            "path is relative to (the runner resolves it the same way)")
    names = json.loads((split_dir / "file_names.json").read_text(encoding="utf-8"))
    img_meta = np.load(split_dir / "img_meta.npy")
    if img_meta.ndim != 2 or img_meta.shape[1] < 7:
        raise SystemExit(f"img_meta.npy has shape {img_meta.shape}; expected [N, 7]")
    if len(names) != len(img_meta):
        problems.append(f"file_names.json lists {len(names)} names, img_meta has {len(img_meta)} rows")
    boxes_per_image = img_meta[:, 4].astype(np.int64)
    rels_per_image = img_meta[:, 6].astype(np.int64)
    rels = np.load(split_dir / "rels.npy", mmap_mode="r")

    jobs = ((str(image_root / name), name) for name in names)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcome = list(pool.map(probe_image, jobs, chunksize=64))

    failures = [(name, error) for name, ok, error in outcome if not ok]
    decoded = len(outcome) - len(failures)
    valid = sum(1 for (_, ok, _), nb in zip(outcome, boxes_per_image, strict=True) if ok and nb >= 2)
    dropped_by_cap = 0
    over_cap = np.nonzero(boxes_per_image > max_objects)[0]
    for index in over_cap:
        start, count = int(img_meta[index, 5]), int(img_meta[index, 6])
        rows = rels[start:start + count]
        if len(rows):
            # Box indices are image-local (see RelationDataset.__getitem__).
            dropped_by_cap += int(((rows[:, 0] >= max_objects) | (rows[:, 1] >= max_objects)).sum())

    loaded = len(names)
    loader_images, loader_note = None, "loader cross-check unavailable: torch not importable"
    try:
        sys.path.insert(0, str(REPO))
        from src.modules.relateanything.data import RelationDataset

        loader_images = len(RelationDataset(str(data_root), "test",
                                            resolution=448, max_objects=max_objects))
        loader_note = "loaded through RelationDataset"
    except Exception as error:  # torch blocked or absent; the raw-array counts stand alone
        loader_note = f"loader cross-check unavailable: {type(error).__name__}"
    if loader_images is not None and loader_images != loaded:
        problems.append(f"RelationDataset reads {loader_images} images, img_meta has {loaded}")

    if "num_images" not in meta:
        problems.append("pack meta.json declares no num_images; cannot cross-check the count")
    elif int(meta["num_images"]) != loaded:
        problems.append(f"pack declares {meta['num_images']!r} images, img_meta has {loaded}")
    if loaded != EXPECTED_IMAGES:
        problems.append(f"official VG150 test pack has {EXPECTED_IMAGES} images, img_meta has {loaded}")
    if valid != loaded:
        problems.append(f"valid_input_images {valid} != loaded_images {loaded}")
    if failures:
        problems.append(f"{len(failures)} images failed to decode")
    section = {
        "dataset": "vg150", "split": "test",
        "expected_images": int(meta.get("num_images", loaded)),
        "loaded_images": int(loaded), "valid_input_images": int(valid),
        "limit": 0, "failed_images": int(len(failures)),
        "pack_sha256": {name: sha256_file(split_dir / name) for name in PACK_FILES},
        "audit": {
            "image_root_declared": declared, "image_root_resolved": str(image_root),
            "cwd": os.getcwd(), "decoded_images": int(decoded),
            "zero_box_images": int((boxes_per_image == 0).sum()),
            "single_box_images": int((boxes_per_image == 1).sum()),
            "zero_relation_images": int((rels_per_image == 0).sum()),
            "total_boxes": int(boxes_per_image.sum()),
            "total_relations": int(rels_per_image.sum()),
            "max_boxes_per_image": int(boxes_per_image.max()) if loaded else 0,
            "images_over_object_cap": int(over_cap.size),
            "relations_dropped_by_object_cap": int(dropped_by_cap),
            "loader_cross_check": loader_note,
            "failure_samples": [{"name": name, "error": error}
                                for name, error in failures[:failure_samples]],
        },
    }
    return section, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Directory containing test/<pack files>; img_dir resolves from cwd")
    parser.add_argument("--output", required=True, type=Path,
                        help="External evidence JSON; existing sections are merged, not replaced")
    parser.add_argument("--official-commit", default=APACHE_SOURCE_COMMIT)
    parser.add_argument("--snapshot-revision", help="40-hex Hub commit; derived from .cache when omitted")
    parser.add_argument("--data-revision", help="40-hex Hub commit; derived from .cache when omitted")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-objects", type=int, default=100)
    parser.add_argument("--failure-samples", type=int, default=10)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    evidence: dict = {}
    previous_keys: list[str] = []
    if args.output.exists():
        try:
            evidence = json.loads(args.output.read_text(encoding="utf-8"))
            if not isinstance(evidence, dict):
                evidence = {}
        except ValueError:
            evidence = {}
        previous_keys = sorted(evidence)

    problems: list[str] = []
    actual_commit = git_output(args.official_root, "rev-parse", "HEAD")
    if actual_commit != args.official_commit:
        problems.append(f"official checkout HEAD {actual_commit} != {args.official_commit}")
    if git_output(args.official_root, "status", "--porcelain"):
        problems.append("official checkout is not clean")
    evidence["official_commit"] = actual_commit

    for key, directory, given in (("snapshot_revision", args.snapshot, args.snapshot_revision),
                                  ("data_revision", args.data_root, args.data_revision)):
        revision = given or derive_revision(directory)[0]
        if not revision or not FULL_REVISION.fullmatch(revision):
            problems.append(f"{key} unavailable; pass it explicitly")
            revision = given
        evidence[key] = revision

    evidence["snapshot_sha256"] = {name: sha256_file(args.snapshot / name) for name in SNAPSHOT_FILES}
    evidence["consumer_sha256"] = {name: sha256_file(REPO / name) for name in CONSUMER_FILES}

    refreshed = {"official_commit", "snapshot_revision", "data_revision",
                 "snapshot_sha256", "consumer_sha256"}
    if not args.skip_load:
        section, section_problems = strict_load_section(args.snapshot, args.official_root)
        evidence["strict_load"] = section
        problems += section_problems
        refreshed.add("strict_load")
        print(f"strict_load: missing/unexpected/shape_mismatches all empty: "
              f"{not section_problems}")
    if not args.skip_split:
        section, section_problems = split_section(
            args.data_root, args.workers, args.max_objects, args.failure_samples)
        evidence["full_split"] = section
        problems += section_problems
        refreshed.add("full_split")
        print(f"full_split: loaded={section['loaded_images']} valid={section['valid_input_images']} "
              f"failed={section['failed_images']} zero_relation={section['audit']['zero_relation_images']}")
    kept = [name for name in previous_keys if name not in refreshed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if kept:
        print(f"kept unchanged from previous output: {', '.join(kept)}", file=sys.stderr)
    print(f"wrote {args.output} sha256={sha256_file(args.output)}")
    if problems:
        print("evidence problems (the gate will reject these):", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
