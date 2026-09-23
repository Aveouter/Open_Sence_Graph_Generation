"""Phase IA protocol validator.

Checks the invariants that ADR 0005 and ADR 0006 declare but that nothing else
enforces. None of these are statistical; they are the conditions under which the
statistics mean anything.

1. **World separation is by construction.** Development and audit seed pools are
   disjoint, and the disjointness is re-derived here rather than trusted from the
   config's own assertion.
2. **Every artifact carries its provenance stamp.** ``status``,
   ``not_a_reproduction``, ``scope``, ``world``, ``schema_version``,
   ``config_sha256`` and ``git_sha`` are all present. A missing stamp is an
   error, not a warning: an unlabelled artifact is exactly the thing the
   repository's claim guardrails exist to prevent.
3. **An audit-world artifact may not exist before its freeze.** If a run claims
   ``world: audit_world`` while no freeze record is present, this fails closed.
   The audit world is meant to be instantiated once, after the protocol is
   frozen, and "we ran it early" must not be silently indistinguishable from
   "we ran it properly".
4. **Level-2 must describe the same config as Level-1.** The oracle utility and
   the data it was measured on have to come from one frozen configuration;
   otherwise the gate's verdict combines two different experiments.

Exit codes: ``0`` clean, ``2`` violations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common import DEFAULT_OUTPUT_ROOT, status_block, write_json
from ..config import WORLD_AUDIT, Phase1AConfig, config_digest

REQUIRED_STAMP_FIELDS = (
    "status",
    "not_a_reproduction",
    "scope",
    "world",
    "schema_version",
    "config_sha256",
    "git_sha",
)

FREEZE_RECORD = "audit_freeze.json"


def validate(root: Path, config: Phase1AConfig) -> list[str]:
    violations: list[str] = []

    if set(config.development_seeds) & set(config.audit_seeds):
        violations.append("development and audit seed pools overlap")

    if not root.exists():
        return violations

    freeze = root / FREEZE_RECORD
    # The digest every artifact must agree on is the one the artifact set itself
    # declares, not this process's default: a run with overridden parameters is
    # legitimate, and what matters is that the oracle utility and the gate
    # describe the same frozen configuration rather than two different ones.
    expected = _recorded_digest(root) or config_digest(config)

    for path in sorted(root.rglob("*.json")):
        if path.name == FREEZE_RECORD or path.name == "manifest.json":
            continue
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            violations.append(f"{path.name}: not valid JSON ({error})")
            continue
        if not isinstance(payload, dict):
            continue

        missing = [field for field in REQUIRED_STAMP_FIELDS if field not in payload]
        if missing:
            violations.append(f"{path.name}: missing stamp fields {missing}")
            continue

        if payload["world"] == WORLD_AUDIT and not freeze.exists():
            violations.append(
                f"{path.name}: claims the audit world but no {FREEZE_RECORD} is present"
            )

        if path.name in ("gate.json", "level2.json", "config.json"):
            if payload["config_sha256"] != expected:
                violations.append(
                    f"{path.name}: config_sha256 does not match the current frozen config; "
                    "the artifact describes a different experiment"
                )

    level1_dir = root / "phase1a"
    level1_config = _read_stamp(level1_dir / "gate.json", "config_sha256")
    dataset = level1_dir / "oracle_dataset.jsonl"

    # Every Level-2 artifact, not one fixed filename: the oracle has more than one
    # parameterisation, and which one produced a verdict is recorded in the
    # artifact rather than implied by where it was written. Checking only
    # ``level2.json`` would leave a second measurement unchecked -- and an
    # unchecked measurement is the failure this whole module exists to prevent.
    for path in sorted(level1_dir.glob("level2*.json")) if level1_dir.exists() else []:
        recorded_config = _read_stamp(path, "config_sha256")
        if level1_config and recorded_config and level1_config != recorded_config:
            violations.append(
                f"{path.name} and gate.json disagree on config_sha256; the oracle "
                "utility was measured on a different configuration than the one it "
                "is folded into"
            )

        # The oracle is required to consume the frozen dataset rather than rebuild
        # it, so the artifact has to name the dataset it actually read. Without
        # this check a regenerated dataset silently orphans the measurement taken
        # on the old one: every number would still look valid, and would describe
        # data that no longer exists.
        recorded = _read_stamp(path, "dataset_sha256")
        if recorded and dataset.exists():
            from ..common import sha256_file

            actual = sha256_file(dataset)
            if actual != recorded:
                violations.append(
                    f"{path.name} was measured on a different oracle_dataset.jsonl "
                    f"than the one on disk (recorded {recorded[:12]}..., found "
                    f"{actual[:12]}...)"
                )
        elif recorded and not dataset.exists():
            violations.append(f"{path.name} cites a dataset that is not present")

    return violations


def _recorded_digest(root: Path) -> str | None:
    """The config digest the artifact set declares for itself."""
    for candidate in (root / "phase1a" / "config.json", root / "config.json"):
        digest = _read_stamp(candidate, "config_sha256")
        if digest:
            return digest
    return None


def _read_stamp(path: Path, field: str) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get(field)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--world", default=Phase1AConfig().world)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    config = Phase1AConfig(world=args.world)
    violations = validate(args.root, config)

    report = status_block(
        world=config.world,
        schema_version=config.schema_version,
        config_sha256=config_digest(config),
        root=str(args.root),
        violations=violations,
        clean=not violations,
    )
    if args.report is not None:
        write_json(args.report, report)

    if violations:
        for violation in violations:
            print(f"[protocol_validator] VIOLATION: {violation}")
        return 2
    print(f"[protocol_validator] clean ({args.root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
