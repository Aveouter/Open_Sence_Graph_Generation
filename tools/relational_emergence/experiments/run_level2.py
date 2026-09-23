"""Level-2 oracle utility: train the three arms and write the gate's input.

Reads the frozen episodes written by
:mod:`tools.relational_emergence.experiments.run_phase1a` and writes
``level2.json`` next to them. The JSON carries the two numbers
:func:`tools.relational_emergence.audits.protocol.gate_decision` reads --
``passed``, ``delta_M`` and ``shuffled_vs_base`` -- so the Level-1 gate can fold
this in as

    python -m tools.relational_emergence.experiments.run_phase1a \\
        --level2 outputs/analysis/relational_emergence/phase1a/level2.json

without either driver importing the other.

This module is intentionally thin and imports no torch at module scope: all of
the science lives in :mod:`tools.relational_emergence.models.oracles`, and this
driver only parses arguments, loads the dataset, calls the run and stamps the
artifact. That keeps the CLI importable in the dependency-free CI job even though
the run it describes needs torch.

Exit codes: ``0`` the Level-2 criterion passed, ``1`` the run completed and it
did not, ``3`` the run could not be started (no dataset, or no torch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..common import DEFAULT_OUTPUT_ROOT, git_sha, status_block, write_json
from ..config import DEFAULT_CONFIG, SCHEMA_VERSION, WORLDS, Phase1AConfig, config_digest
from ..models import oracles

MODULE = "tools.relational_emergence.experiments.run_level2"
DEFAULT_DATASET = DEFAULT_OUTPUT_ROOT / "phase1a" / "oracle_dataset.jsonl"
DEFAULT_OUT = DEFAULT_OUTPUT_ROOT / "phase1a" / "level2.json"

PASSED = 0
FAILED = 1
CANNOT_RUN = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Frozen episodes from run_phase1a (JSONL).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where the Level-2 result JSON is written.",
    )
    parser.add_argument("--world", default=DEFAULT_CONFIG.world, choices=WORLDS)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(oracles.OracleConfig.seeds),
        help="One seed per paired replicate; at least two, and a gate verdict "
        "quotes a confidence interval, so three or more is the intended use.",
    )
    parser.add_argument("--epochs", type=int, default=oracles.OracleConfig.epochs)
    parser.add_argument("--hidden", type=int, default=oracles.OracleConfig.hidden)
    parser.add_argument("--depth", type=int, default=oracles.OracleConfig.depth)
    parser.add_argument("--batch-size", type=int, default=oracles.OracleConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=oracles.OracleConfig.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=oracles.OracleConfig.weight_decay
    )
    parser.add_argument(
        "--val-fraction", type=float, default=oracles.OracleConfig.val_fraction
    )
    parser.add_argument("--device", default=oracles.OracleConfig.device)
    parser.add_argument(
        "--oracle",
        # Rollout is the default because ADR 0007 makes it the primary
        # parameterisation: the direct head is retained, and its measurement is
        # reported, but it is not what a verdict is taken from.
        default=oracles.ROLLOUT,
        choices=oracles.ORACLE_MODES,
        help="Which parameterisation to measure. Both train the same three arms "
        "from the same split under the same checkpoint rule and reduce to the same "
        "paired delta_M, but the two loss scales are not comparable, so the choice "
        "is recorded in the artifact and must be named wherever the result is "
        "cited.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset.exists():
        print(
            f"[level2] no oracle dataset at {args.dataset}; generate one with "
            f"python -m tools.relational_emergence.experiments.run_phase1a"
        )
        return CANNOT_RUN

    try:
        import torch
    except ImportError:
        print(
            "[level2] torch is not importable; the Level-2 oracle needs the "
            "environment that has it (the Level-1 gate does not)"
        )
        return CANNOT_RUN

    config = oracles.OracleConfig(
        hidden=args.hidden,
        depth=args.depth,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        weight_decay=args.weight_decay,
        seeds=tuple(args.seeds),
        device=args.device,
    )
    dataset = oracles.load_oracle_dataset(args.dataset)
    result = oracles.run_level2(dataset, config, args.oracle)

    write_json(
        args.out,
        {
            **status_block(
                world=args.world,
                schema_version=SCHEMA_VERSION,
                # Two digests, because two different things have to be pinned and
                # conflating them is a real hazard. `config_sha256` is the *Phase
                # IA* configuration, the same value the gate stamps -- it is what
                # lets the protocol validator confirm that the oracle utility and
                # the gate describe one experiment rather than two. The oracle's
                # own hyperparameters are a separate object and get their own
                # field, so a change to the training setup cannot masquerade as a
                # change to the world, or the reverse.
                config_sha256=config_digest(Phase1AConfig(world=args.world)),
                oracle_config_sha256=config.config_sha256(),
                git_sha=git_sha(),
                # The artifact aggregates over replicates, so ``seed`` carries the
                # seed set rather than one member of it.
                seed=list(config.seeds),
                dataset=str(args.dataset),
                dataset_sha256=oracles.oracle_dataset_sha256(args.dataset),
                torch_version=torch.__version__,
                command=" ".join(["python", "-m", MODULE, *_argv(argv)]),
                oracle_config=config.as_dict(),
                oracle_mode=args.oracle,
            ),
            **result.as_dict(),
        },
    )

    print(f"[level2] oracle mode: {args.oracle}")
    print(
        f"[level2] arms: trueM={result.mean_losses[oracles.TRUE_MODE]:.6f} "
        f"shuffledM={result.mean_losses[oracles.SHUFFLED_MODE]:.6f} "
        f"base={result.mean_losses[oracles.BASE_MODE]:.6f}"
    )
    print(
        f"[level2] delta_M={result.delta_m.mean:+.6f} "
        f"ci=[{result.delta_m.lo:+.6f}, {result.delta_m.hi:+.6f}] "
        f"| shuffled_vs_base={result.shuffled_vs_base.mean:+.6f} "
        f"ci=[{result.shuffled_vs_base.lo:+.6f}, {result.shuffled_vs_base.hi:+.6f}]"
    )
    print(f"[level2] wrote {args.out}")
    if result.passed:
        print("[level2] passed: TrueM beats ShuffledM, and Base is not beaten by it")
        return PASSED
    print(f"[level2] passed: False blocked_by={list(result.blocked_by)}")
    return FAILED


def _argv(argv: Sequence[str] | None) -> list[str]:
    """The arguments this run was given, for the artifact's command line."""
    return list(argv) if argv is not None else sys.argv[1:]


if __name__ == "__main__":
    raise SystemExit(main())
