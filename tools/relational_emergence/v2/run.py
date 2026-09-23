"""Run v2 development prerequisites; never open the reserved audit world."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..common import DEFAULT_OUTPUT_ROOT, git_sha, write_json
from .data import build_batch, build_scenes
from .protocol import PROTOCOL, Protocol, code_digest, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "v2" / "development"
    )
    parser.add_argument("--groups-per-tuple", type=int, default=24)
    parser.add_argument("--calibration-repeats", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--probe-epochs", type=int, default=120)
    args = parser.parse_args(argv)
    import torch

    from .leakage import AuditConfig, records_from_scenes, run_audit
    from .validation import chronology_audit, evaluator_controls, world_audit

    torch.set_num_threads(1)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty; evidence is never overwritten")
    protocol = Protocol(groups_per_tuple=args.groups_per_tuple)
    config = AuditConfig(
        epochs=args.probe_epochs,
        permutations=args.permutations,
        calibration_repeats=args.calibration_repeats,
    )
    rows = build_scenes(protocol)
    batch, _ = build_batch(rows, protocol)
    stamp = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol.digest(),
        "code_sha256": code_digest(),
        "git_sha": git_sha(),
        "audit_config_sha256": digest(asdict(config)),
        "dataset_sha256": digest(rows),
        "audit_config": asdict(config),
        "config": asdict(protocol),
        "status": "implementation_audit",
        "not_a_reproduction": True,
        "world": "development",
    }
    write_json(args.out / "manifest.json", stamp)
    print(
        f"[v2] {len(rows)} episodes; {len({r['group'] for r in rows})} base groups",
        flush=True,
    )
    world = world_audit(rows, protocol)
    write_json(args.out / "world.json", {**stamp, "world_audit": world})
    chronology = chronology_audit(rows, batch)
    write_json(args.out / "chronology.json", {**stamp, "chronology": chronology})
    controls = evaluator_controls()
    write_json(args.out / "controls.json", {**stamp, "evaluator_controls": controls})
    print(
        f"[v2] world={world['status']} chronology={chronology['status']} controls={controls['status']}",
        flush=True,
    )
    leakage = run_audit(records_from_scenes(rows), protocol, config)
    write_json(args.out / "m0.json", {**stamp, "leakage": leakage})
    blocked = [
        name
        for name, passed in (
            ("world", world["status"] == "PASS"),
            ("chronology", chronology["status"] == "PASS"),
            ("M0", leakage["verdict"] == "PASS_NO_DETECTABLE_LEAKAGE"),
            ("synthetic_evaluators", controls["status"] == "PASS"),
        )
        if not passed
    ]
    # Level-2, learned-decoder controls and the representation factorial require
    # a validated world/control definition. This driver makes no substitute GO.
    result = {
        **stamp,
        "verdict": "STOP_DATA" if blocked else "INCOMPLETE",
        "blocked": blocked,
        "m0": leakage["verdict"],
        "multi_object": world["status"],
        "chronology_control": chronology["status"],
        "evaluator_controls": controls["status"],
        "selected_horizon": world["selected_horizon"],
        "training_seeds": list(protocol.seeds),
        "level2": "PENDING",
        "representation_factorial": "NOT_RUN",
        "fresh_audit": "SEALED",
        "issue_107_complete": False,
    }
    write_json(args.out / "gate.json", result)
    print(
        f"[v2] {result['verdict']}; blocked={blocked}; fresh audit remains SEALED",
        flush=True,
    )
    return 3 if blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
