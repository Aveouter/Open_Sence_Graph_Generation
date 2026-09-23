"""Phase IA data generation, M0 audits, and the Level-1 identifiability gate.

Everything in this driver is pure stdlib, so the gate runs in the
dependency-free CI job rather than only on a machine with torch. The Level-2
oracle utility is a separate torch step reading the dataset this writes; the
verdict stays ``INCOMPLETE`` until that has run and been folded in.

Writes, under ``outputs/analysis/relational_emergence/phase1a/``:

``config.json``      the frozen settings this run is reproducible from
``leakage.json``     the four M0 probes, their baselines, and the must-fire control
``j_ab.json``        the mechanism-pair x regime x tuple separability matrix
``floors.json``      the numerical and null floors, kept separate
``horizon.json``     the saturation curve and the selected horizon
``gate.json``        the verdict, its reasons, and the dispositions for any failures
``oracle_dataset.jsonl``  frozen episodes for the torch oracles to consume
``manifest.json``    artifact hashes, so a report can cite what it read

Exit codes: ``0`` GO, ``1`` INCOMPLETE, ``3`` STOP or STOP_DATA.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ..audits import identifiability as ident
from ..audits import leakage as leak
from ..audits import protocol
from ..common import DEFAULT_OUTPUT_ROOT, git_sha, status_block, write_json
from ..config import (
    DEFAULT_CONFIG,
    WORLDS,
    Phase1AConfig,
    config_digest,
    world_families,
    world_seed,
)
from ..simulator.counterfactuals import CounterfactualGroup, build_group
from ..simulator.dataset import build_rows, write_rows
from ..simulator.factors import enumerate_realizable
from ..simulator.fillers import build_filler_pool
from ..simulator.splits import (
    DEFAULT_SPLIT_SEED,
    PROBE_IDENTITY_BUCKETS,
    PROBE_IDENTITY_RANGE,
    SPLIT_KINDS,
    SUPPORTER_IDENTITY_BUCKETS,
    SUPPORTER_IDENTITY_RANGE,
    build_split,
)

EXIT_CODES = {"GO_PHASE_IA": 0, "INCOMPLETE": 1, "STOP": 3, "STOP_DATA": 3}


def build_groups(config: Phase1AConfig) -> list[CounterfactualGroup]:
    """Every base context for one world.

    Both worlds run the identical generator; only the filler pool it draws from
    changes. That is the whole content of the seal -- a different *sample*, not a
    different process -- so a result that holds in the audit world is a result
    about the physics, and a result that fails is not explained away by a changed
    simulator.
    """
    rng = random.Random(world_seed(config.world))
    pool = build_filler_pool(config.filler_pool_size, rng, families=world_families(config.world))
    groups: list[CounterfactualGroup] = []
    for tuple_index, s0 in enumerate(enumerate_realizable()):
        for group_index in range(config.groups_per_tuple):
            filler = pool[(tuple_index * 5 + group_index * 3) % len(pool)]
            groups.append(build_group(s0, group_index, filler, config.horizon))
    return groups


def check_level2_dataset(level2: dict, current_sha256: str) -> None:
    """Refuse a Level-2 result measured against a different dataset.

    The oracle's artifact records the SHA-256 of the dataset it read, and the
    whole point of that field is to be checked. Folding in a result measured
    against *another* dataset would put a number in the gate that describes an
    experiment this run did not perform -- and it would look exactly like a valid
    one, because both artifacts are well-formed.

    This was not hypothetical. After a schema change to the episode file the
    artifact on disk still carried the old digest, and the discrepancy was found
    by comparing two hex strings by hand. A check that runs on every fold-in is
    what makes that automatic rather than dependent on someone looking.
    """
    recorded = level2.get("dataset_sha256")
    if recorded is None:
        raise SystemExit(
            "the Level-2 artifact records no dataset digest, so it cannot be "
            "shown to describe this dataset; re-run run_level2"
        )
    if recorded != current_sha256:
        raise SystemExit(
            f"the Level-2 artifact was measured against dataset {recorded}, but "
            f"this run wrote {current_sha256}. Re-run run_level2 before folding it "
            "in; a stale oracle result folded into a fresh gate is a verdict about "
            "an experiment that did not happen."
        )


def curves_to_json(curves: dict) -> dict:
    return {
        tuple_key: [
            {
                "regime": entry.regime,
                "pair": entry.pair,
                "auc": entry.auc(),
                "mean_curve": list(entry.mean_curve()),
                "per_group_auc": list(entry.per_group_auc()),
            }
            for entry in entries
        ]
        for tuple_key, entries in sorted(curves.items())
    }


def write_oracle_dataset(
    path: Path, groups: list[CounterfactualGroup], config: Phase1AConfig
) -> int:
    """Freeze the episodes the torch oracles train on.

    The oracles are required to consume this rather than re-sample: rebuilding
    twins on the model side would let the split, the compatibility table or the
    pairing quietly diverge from the audited ones, and every number downstream
    would then describe a different experiment from the one that was checked.
    Phase IB reads the same file for the same reason.
    """
    rows = build_rows(groups, config.horizon, group_limit=config.oracle_group_limit)
    return write_rows(path, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default=DEFAULT_CONFIG.world, choices=WORLDS)
    parser.add_argument("--groups-per-tuple", type=int, default=DEFAULT_CONFIG.groups_per_tuple)
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.horizon)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase1a")
    parser.add_argument(
        "--level2",
        type=Path,
        default=None,
        help="Level-2 oracle result JSON, if the torch step has already run",
    )
    args = parser.parse_args(argv)

    config = Phase1AConfig(
        world=args.world,
        groups_per_tuple=args.groups_per_tuple,
        horizon=args.horizon,
    )
    stamp = status_block(
        world=config.world,
        schema_version=config.schema_version,
        config_sha256=config_digest(config),
        git_sha=git_sha(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", {**stamp, "config": config.as_dict()})

    started = time.time()
    groups = build_groups(config)
    print(f"[phase1a] groups={len(groups)} horizon={config.horizon}")

    scale = ident.robust_scale(ident.collect_calibration_vectors(groups, config.horizon, ident.DEFAULT_CONSTANTS))
    curves = ident.pair_curves(groups, scale, config.horizon)
    floor_num = ident.numerical_floor(groups, scale, config.horizon)
    null_curves = ident.null_floor_curves(groups, scale, config.horizon)
    floor = protocol.floor_level(null_curves)
    floor_mean = sum(e.auc() for t in null_curves for e in null_curves[t]) / sum(
        len(null_curves[t]) for t in null_curves
    )
    print(
        f"[phase1a] floor_num max={floor_num['max']:.3e} mean={floor_num['mean']:.3e} "
        f"| floor_null 95={floor:.4f} mean={floor_mean:.4f}"
    )

    write_json(
        args.out / "floors.json",
        {
            **stamp,
            "numerical_floor": floor_num,
            "null_floor_quantile_95": floor,
            "null_floor_mean": floor_mean,
            "frozen_scale": list(scale),
            "null_per_cell": curves_to_json(null_curves),
        },
    )
    write_json(args.out / "j_ab.json", {**stamp, "curves": curves_to_json(curves)})

    horizon = protocol.select_horizon(curves, "rich")
    write_json(args.out / "horizon.json", {**stamp, **horizon})
    print(
        f"[phase1a] selected H={horizon['selected_horizon']} "
        f"(saturated={horizon['saturated']})"
    )

    records = leak.build_records(groups)
    # Both estimators run, and the verdict takes the stronger result per feature
    # set: a leak a linear probe misses is still a leak, and reporting only the
    # weaker probe would make a false negative look like a clean sampler.
    leakage_results: dict[str, dict] = {}
    for feature_set in leak.FEATURE_SETS:
        for probe in ("linear", "random_feature"):
            result = leak.evaluate_feature_set(
                records,
                feature_set,
                n_permutations=config.leakage_permutations,
                seed=7,
                probe=probe,
            )
            result["significantly_above_baseline"] = (
                result["permutation_p_value"] < 0.05
                and result["accuracy"] > result["compatibility_aware_baseline"]
            )
            leakage_results[f"{feature_set}::{probe}"] = result
    for name, result in leakage_results.items():
        if not name.endswith("::linear"):
            continue
        stronger = leakage_results[name.replace("::linear", "::random_feature")]
        result["nonlinear_accuracy"] = stronger["accuracy"]
        result["nonlinear_p_value"] = stronger["permutation_p_value"]
        result["nonlinear_flagged"] = stronger["significantly_above_baseline"]
        result["significantly_above_baseline"] = (
            result["significantly_above_baseline"] or stronger["significantly_above_baseline"]
        )
    # The control validates the detector *suite*: it fires if either estimator
    # catches the deliberate leak. Requiring a specific one would fail the gate
    # when a more expressive probe happens to overfit a small sample -- an
    # artefact of the probe, not evidence about the sampler.
    control_records = leak.leaky_sampler_control(groups)
    control_by_probe = {
        probe: leak.evaluate_feature_set(
            control_records,
            "x0_only",
            n_permutations=config.leakage_permutations,
            seed=7,
            probe=probe,
        )
        for probe in ("linear", "random_feature")
    }
    for result in control_by_probe.values():
        result["fired"] = (
            result["accuracy"] > result["compatibility_aware_baseline"]
            and result["permutation_p_value"] < 0.05
        )
    control = control_by_probe["linear"]
    control["fired"] = any(result["fired"] for result in control_by_probe.values())
    control["by_probe"] = {
        probe: {"accuracy": r["accuracy"], "p_value": r["permutation_p_value"], "fired": r["fired"]}
        for probe, r in control_by_probe.items()
    }
    leakage_results["__controls_fired__"] = control["fired"]

    # Pre-registered power: every probe must be able to detect the declared
    # effect size, or a null result says nothing about the sampler.
    achieved_mde = max(
        result["minimum_detectable_effect"]
        for name, result in leakage_results.items()
        if not name.startswith("__")
    )
    powered = achieved_mde <= config.leakage_mde_target
    leakage_results["__adequately_powered__"] = powered
    write_json(
        args.out / "leakage.json",
        {
            **stamp,
            "n_records": len(records),
            "probes": {k: v for k, v in leakage_results.items() if not k.startswith("__")},
            "must_fire_control": control,
            "controls_fired": control["fired"],
            "mde_target": config.leakage_mde_target,
            "mde_achieved": achieved_mde,
            "adequately_powered": powered,
        },
    )
    if not powered:
        print(
            f"[phase1a] WARNING: achieved MDE {achieved_mde:.4f} exceeds the "
            f"pre-registered target {config.leakage_mde_target:.4f}; the leakage "
            "verdict is underpowered and must be reported as such"
        )
    print(
        "[phase1a] leakage (linear / random-feature, against baseline "
        f"{next(iter(leakage_results.values()))['compatibility_aware_baseline']:.3f}):"
    )
    for name, result in leakage_results.items():
        if not name.endswith("::linear"):
            continue
        marker = " LEAK" if result["significantly_above_baseline"] else ""
        print(
            f"[phase1a]   {name.split('::')[0]:16s} "
            f"{result['accuracy']:.3f} / {result['nonlinear_accuracy']:.3f}{marker}"
        )
    print(f"[phase1a]   must-fire control fired={control['fired']}")

    separability = protocol.separability_verdict(curves, floor, horizon["selected_horizon"])
    precondition = {
        "passed": floor_mean < 0.1 * max(
            entry.auc() for t in curves for entry in curves[t]
        ),
        "floor_null_mean": floor_mean,
        "floor_null_quantile_95": floor,
    }
    level2 = None
    if args.level2 is not None and args.level2.exists():
        level2 = json.loads(args.level2.read_text(encoding="utf-8"))
        check_level2_dataset(level2, _digest(args.out / "oracle_dataset.jsonl")["sha256"])
    decision = protocol.gate_decision(leakage_results, separability, precondition, level2)
    write_json(
        args.out / "gate.json",
        {
            **stamp,
            "separability": separability,
            "precondition": precondition,
            "decision": decision,
        },
    )

    # Emitted rather than left as machinery: a split that is never materialised
    # cannot be audited, and the audit is the point. All four of the plan's kinds
    # are expressible here. An earlier version of this comment recorded
    # pair-recombination as blocked on the sampler; that was wrong, and ADR 0006
    # carries the correction -- `_structural` draws the probe's and the
    # supporter's attributes from independent uniforms, so the pair already
    # exists and it was the split that was missing.
    split_report = {}
    for kind in SPLIT_KINDS:
        split = build_split(groups, kind, seed=DEFAULT_SPLIT_SEED)
        split_report[kind] = {
            "train": list(split.train),
            "test": list(split.test),
            "held_out": list(split.held_out),
            "n_train": len(split.train),
            "n_test": len(split.test),
        }
    write_json(
        args.out / "splits.json",
        {
            **stamp,
            "split_seed": DEFAULT_SPLIT_SEED,
            "unit": "base_context",
            "kinds": split_report,
            "identity_edges": {
                "probe": [PROBE_IDENTITY_RANGE, PROBE_IDENTITY_BUCKETS],
                "supporter": [SUPPORTER_IDENTITY_RANGE, SUPPORTER_IDENTITY_BUCKETS],
            },
        },
    )

    written = write_oracle_dataset(args.out / "oracle_dataset.jsonl", groups, config)
    manifest = {
        "config.json": _digest(args.out / "config.json"),
        "floors.json": _digest(args.out / "floors.json"),
        "j_ab.json": _digest(args.out / "j_ab.json"),
        "horizon.json": _digest(args.out / "horizon.json"),
        "leakage.json": _digest(args.out / "leakage.json"),
        "gate.json": _digest(args.out / "gate.json"),
        "oracle_dataset.jsonl": _digest(args.out / "oracle_dataset.jsonl"),
    }
    write_json(
        args.out / "manifest.json",
        {
            **stamp,
            "artifacts": manifest,
            "oracle_episodes": written,
            "wall_seconds": time.time() - started,
        },
    )

    print(
        f"[phase1a] verdict={decision['verdict']} "
        f"inseparable_important={decision['inseparable_important_pairs']}"
    )
    for reason in decision["reasons"]:
        print(f"[phase1a]   reason: {reason}")
    return EXIT_CODES.get(decision["verdict"], 1)


def _digest(path: Path) -> dict:
    from ..common import sha256_file

    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


if __name__ == "__main__":
    raise SystemExit(main())
