"""Ontology audit: enumerate ``S_0_realizable`` and check the compatibility table.

This is the M0 step that has to pass before any data is generated. It answers
only structural questions -- which tuples exist, which mechanisms are eligible at
each, and whether the admissibility constraints hold -- and writes the result as
an artifact so that a later reader re-checks the rule rather than trusting this
summary.

Exit codes: ``0`` admissible, ``2`` constraint violation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common import DEFAULT_OUTPUT_ROOT, git_sha, status_block, write_json
from ..config import DEFAULT_CONFIG, WORLDS, Phase1AConfig, config_digest
from ..simulator.compatibility import (
    ACTIVE_MECHANISMS,
    MECHANISMS,
    analytic_baseline_by_tuple,
    audit_compatibility,
    compatible_mechanisms,
    compatibility_table,
)
from ..simulator.factors import derived_readouts, enumerate_realizable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default=DEFAULT_CONFIG.world, choices=WORLDS)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "audits" / "ontology_audit.json",
    )
    return parser


def build_report(config: Phase1AConfig) -> dict:
    table = compatibility_table()
    realizable = enumerate_realizable()
    violations = audit_compatibility()

    return status_block(
        world=config.world,
        schema_version=config.schema_version,
        config_sha256=config_digest(config),
        git_sha=git_sha(),
        factor_domains={
            "containment": ["inside", "outside"],
            "contact": ["touching", "non_touching"],
            "pose": ["on_top", "other"],
        },
        n_realizable_tuples=len(realizable),
        realizable_tuples=[
            {**s0.as_dict(), "readouts": derived_readouts(s0), "compatible": list(table[s0.key()])}
            for s0 in realizable
        ],
        mechanisms=list(MECHANISMS),
        active_mechanisms=list(ACTIVE_MECHANISMS),
        analytic_baseline_by_tuple=analytic_baseline_by_tuple(),
        constraint_violations=violations,
        admissible=not violations,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = Phase1AConfig(world=args.world)
    report = build_report(config)

    print(f"[ontology_audit] world={config.world} tuples={report['n_realizable_tuples']}")
    for s0 in enumerate_realizable():
        mechanisms = compatible_mechanisms(s0)
        baseline = report["analytic_baseline_by_tuple"][s0.key()]
        print(
            f"  {s0.key():<34s} compatible={len(mechanisms)} "
            f"({', '.join(mechanisms)}) baseline={baseline:.4f}"
        )

    if report["constraint_violations"]:
        for violation in report["constraint_violations"]:
            print(f"[ontology_audit] VIOLATION: {violation}")
        write_json(args.out, report)
        print(f"[ontology_audit] wrote {args.out}")
        return 2

    write_json(args.out, report)
    print(f"[ontology_audit] admissible; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
