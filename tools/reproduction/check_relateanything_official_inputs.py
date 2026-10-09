#!/usr/bin/env python3
"""Offline RelateAnything input audit. Does not load weights or run metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import zipfile
from functools import cache
from pathlib import Path

SOURCE_FILES = (
    "benchmark/eval_zeroshot.py", "relsgg/checkpoint.py", "relsgg/eval/evaluator.py",
    "relsgg/data/dataset.py", "pyproject.toml", "LICENSE", "NOTICE",
)
SNAPSHOT_FILES = ("model.pth", "text_student.pt", "tokenizer.json", "tokenizer_config.json")
PACK_FILES = ("meta.json", "file_names.json", "img_meta.npy", "boxes.npy", "box_cats.npy", "rels.npy")
OFFICIAL_COMMIT = "06766fdf56752ca535fc9b971fca99ce563676d0"
APACHE_SOURCE_COMMIT = "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
CONSUMER_FILES = ("src/relateanything.py", "src/relateanything_cli.py", "src/relateanything_evaluation.py")


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--official-commit", default=OFFICIAL_COMMIT,
                        help="Expected full source commit; revision changes require a recorded decision")
    parser.add_argument("--evidence", type=Path, help="External strict-load and full-split audit JSON")
    parser.add_argument("--protocol", choices=("A1", "A3"), default="A1")
    parser.add_argument("--consumer", choices=("official", "opensgg"), default="official",
                        help="Explicit runner contract; the port reads the released names/W bank")
    parser.add_argument("--tau-calibration", type=Path,
                        help="A3 synonym-matcher calibration, not deployment calibration.json")
    args = parser.parse_args()
    # Per invocation: large weights and pack arrays are read only once.
    digest = cache(file_hash)
    source_files = tuple(name for name in SOURCE_FILES
                         if name != "NOTICE" or args.official_commit != APACHE_SOURCE_COMMIT)
    paths = [args.official_root / name for name in source_files]
    paths += [args.snapshot / name for name in SNAPSHOT_FILES]
    paths += [args.data_root / "test" / name for name in PACK_FILES]
    if args.protocol == "A3":
        paths.append(args.snapshot / "predicate_embeddings.npz")
    missing = [str(path.resolve()) for path in paths if not path.is_file() or path.stat().st_size == 0]
    blockers = (["missing_required_assets"] if missing else [])
    blockers += ["strict_load_unverified", "full_split_unverified"]
    evidence = {}
    if args.evidence:
        try:
            evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
            if not isinstance(evidence, dict):
                raise ValueError("evidence must be an object")
        except (OSError, ValueError):
            evidence = {}
            blockers.append("evidence_invalid")
    consumer_hashes = {}
    if args.consumer == "opensgg":
        repo = Path(__file__).resolve().parents[2]
        consumer_hashes = {name: digest(repo / name) for name in CONSUMER_FILES}
        if evidence.get("consumer_sha256") != consumer_hashes or not all(consumer_hashes.values()):
            blockers.append("consumer_identity_unverified")
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(args.official_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(args.official_root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        actual_commit, dirty = None, None
    if (not re.fullmatch(r"[0-9a-f]{40}", args.official_commit)
            or actual_commit != args.official_commit or dirty
            or evidence.get("official_commit") != actual_commit):
        blockers.append("official_source_identity_unverified")
    revisions = (evidence.get("snapshot_revision"), evidence.get("data_revision"))
    if not all(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions):
        blockers.append("asset_revisions_unverified")
    snapshot_hashes = evidence.get("snapshot_sha256")
    if (not isinstance(snapshot_hashes, dict) or not all(
            digest(args.snapshot / name) is not None
            and snapshot_hashes.get(name) == digest(args.snapshot / name) for name in SNAPSHOT_FILES)):
        blockers.append("snapshot_identity_unverified")
    load = evidence.get("strict_load", {})
    if isinstance(load, dict) and load:
        if (load.get("model_sha256") != digest(args.snapshot / "model.pth")
                or load.get("text_student_sha256") != digest(args.snapshot / "text_student.pt")):
            blockers.append("strict_load_artifact_mismatch")
        elif (not isinstance(load.get("source_sha256"), dict)
              or not all(digest(args.official_root / name) is not None
                         and load["source_sha256"].get(name) == digest(args.official_root / name)
                         for name in source_files)):
            blockers.append("strict_load_source_mismatch")
        elif (load.get("strict") is True and all(load.get(key) == [] for key in
                ("missing_keys", "unexpected_keys", "shape_mismatches", "remapped_keys"))):
            blockers.remove("strict_load_unverified")
    split = evidence.get("full_split", {})
    if isinstance(split, dict) and split:
        counts_match = (split.get("dataset") == "vg150" and split.get("split") == "test"
                        and all(type(split.get(key)) is int and split[key] == 26404 for key in
                                ("expected_images", "loaded_images", "valid_input_images"))
                        and type(split.get("limit")) is int and split["limit"] == 0
                        and type(split.get("failed_images")) is int and split["failed_images"] == 0)
        if not counts_match:
            blockers.append("full_split_count_mismatch")
        elif (isinstance(split.get("pack_sha256"), dict)
              and all(digest(args.data_root / "test" / name) is not None
                      and split["pack_sha256"].get(name) == digest(args.data_root / "test" / name)
                      for name in PACK_FILES)):
            blockers.remove("full_split_unverified")
        else:
            blockers.append("full_split_artifact_mismatch")
    fields = []
    if args.protocol == "A3":
        embeddings = args.snapshot / "predicate_embeddings.npz"
        if embeddings.is_file():
            try:
                with zipfile.ZipFile(embeddings) as archive:
                    fields = sorted(name[:-4] for name in archive.namelist() if name.endswith(".npy"))
                required_fields = {"names", "W"} if args.consumer == "opensgg" else {"predicates", "embeddings"}
                if not required_fields.issubset(fields):
                    blockers.append("a3_embedding_layout_incompatible")
            except (OSError, zipfile.BadZipFile):
                blockers.append("a3_embedding_archive_invalid")
        if args.tau_calibration is None:
            blockers.append("a3_matcher_calibration_missing")
        else:
            try:
                calibration = json.loads(args.tau_calibration.read_text(encoding="utf-8"))
                calibrated_path = Path(calibration["pred_embeds"])
                # Relative identities are anchored beside the calibration, never cwd.
                if not calibrated_path.is_absolute():
                    calibrated_path = args.tau_calibration.resolve().parent / calibrated_path
                if calibrated_path.resolve() != embeddings.resolve():
                    blockers.append("a3_calibration_text_space_mismatch")
                tau = calibration["chosen"]["tau"]
                if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or not -1 <= tau <= 1:
                    blockers.append("a3_matcher_threshold_invalid")
            except (OSError, ValueError, KeyError, TypeError):
                blockers.append("a3_matcher_calibration_invalid")
        a3 = evidence.get("a3", {})
        if (not isinstance(a3, dict)
                or a3.get("checkpoint_pred_embeds") != str(embeddings.resolve())
                or a3.get("vocabulary_order_verified") is not True
                or a3.get("runner_layout_verified") is not True
                or not digest(embeddings)
                or a3.get("embedding_sha256") != digest(embeddings)
                or not args.tau_calibration or not digest(args.tau_calibration)
                or a3.get("calibration_sha256") != digest(args.tau_calibration)):
            blockers.append("a3_runner_contract_unverified")
    student = (args.snapshot / "text_student.pt").resolve()
    command = ["python", str((args.official_root / "benchmark/eval_zeroshot.py").resolve()),
               "--checkpoint", str((args.snapshot / "model.pth").resolve()),
               "--text_student", str(student), "--data_roots", str(args.data_root.resolve()),
               "--split", "test", "--graph_constraint", "--limit", "0"]
    if args.protocol == "A3":
        command += ["--open_vocab"]
        if args.tau_calibration:
            command += ["--tau_calibration", str(args.tau_calibration.resolve())]
    report = {"status": "BLOCKED" if blockers else "PASS", "outcome": "implementation_audit", "can_evaluate": not blockers,
              "reproduction_ready": False,
              "missing_paths": missing, "blockers": blockers,
              "text_student": str(student), "reference_command": command,
              "protocol": args.protocol, "embedding_fields": fields,
              "consumer": args.consumer,
              "expected_official_commit": args.official_commit, "actual_official_commit": actual_commit,
              "snapshot_revision": revisions[0], "data_revision": revisions[1],
              "artifact_sha256": {
                  "source": {name: digest(args.official_root / name) for name in source_files},
                  "snapshot": {name: digest(args.snapshot / name) for name in SNAPSHOT_FILES},
                  "pack": {name: digest(args.data_root / "test" / name) for name in PACK_FILES},
                  "consumer": consumer_hashes,
              },
              "claim_boundary": "Input gate only; external load/split attestations are not independently rerun. No method, paper or metric parity is established."}
    if args.protocol == "A3":
        report["artifact_sha256"]["snapshot"]["predicate_embeddings.npz"] = digest(embeddings)
        report["artifact_sha256"]["tau_calibration"] = digest(args.tau_calibration) if args.tau_calibration else None
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
