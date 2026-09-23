"""Shared constants and artifact helpers for the relational-emergence tool tree.

Stdlib-only by design: the dependency-free CI job imports this module.

Scope note -- every artifact written by this tree is a research instrument
output. It is not evidence for any baseline, and :data:`STATUS_BLOCK` says so in
the artifact itself rather than in a surrounding document.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "relational_emergence"

SCOPE = "synthetic_relational_emergence_phase1a"

STATUS_BLOCK: dict[str, Any] = {
    "status": "research_experiment",
    "not_a_reproduction": True,
    "scope": SCOPE,
}


def status_block(**extra: Any) -> dict[str, Any]:
    """Return the artifact stamp, with caller-supplied fields merged in.

    Callers are expected to add ``world``, ``config_sha256``, ``git_sha``,
    ``schema_version`` and ``seed`` where those apply; this function deliberately
    does not invent them, because a stamp with a placeholder hash is worse than
    no stamp.
    """
    block = dict(STATUS_BLOCK)
    block.update(extra)
    return block


def git_sha() -> str:
    """The current commit, or ``"unknown"`` outside a repository.

    Never raises: an artifact stamped ``"unknown"`` is honest about what it is,
    whereas a missing stamp is what the protocol validator rejects.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
