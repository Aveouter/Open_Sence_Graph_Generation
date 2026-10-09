"""Source identity for the Apache-2.0 port; stdlib-only and offline."""
import hashlib
import json
from pathlib import Path


def verify_source(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8"))
    changed = []
    for name, expected in manifest["files"].items():
        path = root / name
        try:
            canonical = path.read_bytes().replace(b"\r\n", b"\n")
        except OSError:
            changed.append(name)
            continue
        if hashlib.sha256(canonical).hexdigest() != expected:
            changed.append(name)
    if changed:
        raise RuntimeError("RelateAnything source identity mismatch: " + ", ".join(changed))
    return {key: manifest[key] for key in ("source_repo", "source_commit", "reference_commit", "license")}
