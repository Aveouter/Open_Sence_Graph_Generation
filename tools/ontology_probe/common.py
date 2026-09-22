"""Shared paths, VG150 constants, provenance helpers and report status block."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: VG150 lives outside the repo in some checkouts; callers override with --data-root.
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "VisualGenome"
#: 10-image tracked fixture used by --dry-run and unit tests.
SAMPLE_ROOT = REPO_ROOT / "data" / "VisualGenome_sample"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "ontology_probe"

# rel.json's rel_categories[0] is __background__; real predicates are 1..50.
BACKGROUND_PREDICATE_ID = 0
NUM_VG_PREDICATES = 50
NUM_VG_OBJECTS = 150
NUM_VG_OBJECT_SLOTS = NUM_VG_OBJECTS + 1  # 151: ids 0..150, 0 unused

SPLITS = ("train", "val", "test")

#: Stamped into every artifact.  See AGENTS.md -- these files must never be
#: mistaken for benchmark results or reproduction evidence.
STATUS_BLOCK: dict[str, Any] = {
    "status": "diagnostic_experiment",
    "not_a_reproduction": True,
    "scope": "vg150_predicate_ontology_cross_pair_generalization",
}

#: The seven predicates singled out for per-predicate analysis and contact sheets.
STUDY_PREDICATES = ("on", "has", "of", "in", "holding", "wearing", "riding")

# Official VG150 frequency strata.  These mirror the constants in
# src/models/ra_sgg.py, which cannot be imported here because it pulls in torch
# and this module must stay importable in the dependency-free CI job.
# tests/analysis/test_ontology_probe_stats.py asserts the two stay in sync by
# parsing ra_sgg.py with ast.
VG_HEAD_IDS = (8, 20, 22, 29, 30, 31, 48)
VG_BODY_IDS = (1, 5, 6, 7, 9, 11, 16, 19, 21, 23, 25, 33, 35, 38, 40, 41, 43, 46, 47, 49, 50)
VG_TAIL_IDS = (2, 3, 4, 10, 12, 13, 14, 15, 17, 18, 24, 26, 27, 28, 32, 34, 36, 37, 39, 42, 44, 45)


def hbt_group(pred_id: int) -> str:
    """Map a predicate id to its official VG150 head/body/tail stratum."""
    if pred_id in VG_HEAD_IDS:
        return "head"
    if pred_id in VG_BODY_IDS:
        return "body"
    if pred_id in VG_TAIL_IDS:
        return "tail"
    return "unknown"


def status_block(**extra: Any) -> dict[str, Any]:
    """STATUS_BLOCK plus caller-supplied provenance fields."""
    return {**STATUS_BLOCK, **extra}


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Minimal stdlib CSV writer so the core stays dependency-free."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
