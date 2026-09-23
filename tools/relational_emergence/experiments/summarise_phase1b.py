"""Emit the Phase IB report tables straight from the artifacts.

The reports in this repository are hand-written, and the rule they follow is that
every number in them is read from a frozen artifact rather than recomputed. A
formatting pass is how that rule is kept honest at this many numbers: retyping a
table is where a digit changes, and a report that disagrees with its own artifact
is worse than one with no table at all.

Prints Markdown to stdout. It computes nothing the artifacts do not already
contain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import DEFAULT_OUTPUT_ROOT


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact {path}; run run_phase1b first")
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: object, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:+.{places}f}" if isinstance(value, float) else str(value)
    return str(value)


def spread(runs: list[Path]) -> None:
    """Across-seed spread of the headline numbers, over several run directories.

    The plan does not require replication, and a single seed is what the frozen
    protocol runs, so this is reported as a limitation rather than as the result.
    It exists because a between-arm difference is only readable against an
    across-seed one, and comparing them afterwards -- when the seed runs are gone
    -- is not possible.
    """
    per_arm: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        arms = _load(run / "endpoints.json")["arms"]
        for arm, record in arms.items():
            bucket = per_arm.setdefault(arm, {"ccgp": [], "role_gain": [], "val_loss": []})
            bucket["ccgp"].append(record["ccgp_unseen_family"]["above_baseline"])
            bucket["role_gain"].append(record["role_equivariance"]["role_gain"])
            bucket["val_loss"].append(record["val_loss"])
    print(f"### Across-seed spread over {len(runs)} runs\n")
    print("| arm | CCGP mean | min | max | role gain mean |")
    print("|---|---|---|---|---|")
    for arm in sorted(per_arm):
        values = per_arm[arm]["ccgp"]
        gains = per_arm[arm]["role_gain"]
        print(
            f"| `{arm}` | {sum(values) / len(values):+.4f} | {min(values):+.4f} | "
            f"{max(values):+.4f} | {sum(gains) / len(gains):+.4f} |"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase1b")
    parser.add_argument(
        "--spread",
        type=Path,
        nargs="*",
        default=None,
        help="run directories to aggregate across seeds instead of tabulating one run",
    )
    args = parser.parse_args(argv)

    if args.spread is not None:
        spread(list(args.spread))
        return 0

    endpoints = _load(args.run / "endpoints.json")
    transplant = _load(args.run / "transplant.json")
    splits = _load(args.run / "splits.json")
    arms = endpoints["arms"]

    print(f"<!-- world={endpoints.get('world')} code={endpoints.get('code', 'learned')} "
          f"split={splits.get('kind')} held_out={splits.get('held_out')} -->")
    print()

    print("### Training\n")
    print("| arm | parameters | val loss | best epoch |")
    print("|---|---|---|---|")
    for arm, record in arms.items():
        print(f"| `{arm}` | {record['n_parameters']} | {record['val_loss']:.6f} | {record['best_epoch']} |")
    print()

    print("### CCGP on unseen appearance families (headline)\n")
    print("| arm | accuracy | baseline | above baseline |")
    print("|---|---|---|---|")
    for arm, record in arms.items():
        c = record["ccgp_unseen_family"]
        print(f"| `{arm}` | {_f(c['accuracy'])} | {_f(c['compatibility_aware_baseline'])} | "
              f"{_f(c['above_baseline'])} |")
    print()

    print("### CCGP by intervention regime\n")
    print("| arm | passive | weak | rich |")
    print("|---|---|---|---|")
    for arm, record in arms.items():
        cells = []
        for regime in ("passive", "weak", "rich"):
            entry = record.get("ccgp_by_regime", {}).get(regime)
            cells.append("n/a" if not entry or "above_baseline" not in entry else _f(entry["above_baseline"]))
        print(f"| `{arm}` | " + " | ".join(cells) + " |")
    print()

    print("### Role equivariance on unseen families\n")
    print("| arm | E_role | E_shuffled | gain | E_inv |")
    print("|---|---|---|---|---|")
    for arm, record in arms.items():
        r = record["role_equivariance"]
        print(f"| `{arm}` | {_f(r['e_role'])} | {_f(r['e_role_shuffled'])} | "
              f"{_f(r['role_gain'])} | {_f(r['e_inv'])} |")
    print()

    print("### Label efficiency (CCGP above baseline, labels per class)\n")
    budgets = ["1", "5", "10", "50", "None"]
    print("| arm | " + " | ".join(budgets) + " |")
    print("|---" * (len(budgets) + 1) + "|")
    for arm, record in arms.items():
        curve = record.get("label_efficiency", {})
        cells = []
        for budget in budgets:
            entry = curve.get(budget)
            cells.append("n/a" if not entry or "above_baseline" not in entry else _f(entry["above_baseline"]))
        print(f"| `{arm}` | " + " | ".join(cells) + " |")
    print()

    print("### Code ablation (one-step loss with and without the bottleneck)\n")
    print("| arm | with code | without code | relative increase |")
    print("|---|---|---|---|")
    for arm, record in arms.items():
        a = record.get("code_ablation", {})
        print(f"| `{arm}` | {_f(a.get('one_step_loss_with_code'))} | "
              f"{_f(a.get('one_step_loss_without_code'))} | {_f(a.get('relative_increase'))} |")
    print()

    print("### Transplant\n")
    print("| arm | CFError self | correct | wrong | gain | CI | positive |")
    print("|---|---|---|---|---|---|---|")
    for arm, record in transplant["arms"].items():
        s = record.get("summary")
        if not s:
            print(f"| `{arm}` | n/a | n/a | n/a | n/a | n/a | {record.get('reason', '')} |")
            continue
        m = s["arm_mean"]
        print(f"| `{arm}` | {_f(m['self'])} | {_f(m['correct'])} | {_f(m['wrong'])} | "
              f"{_f(s['gain_wrong_minus_correct'])} | "
              f"[{_f(s['ci_low'])}, {_f(s['ci_high'])}] | {s['positive']} |")
    print()

    print("### Transplant alignment, tested against zero, by regime\n")
    print("`confounded` is true when the control's mean falls inside the pairing's")
    print("interval -- i.e. when this statistic cannot separate a matched code from")
    print("an arbitrary one, and must not be read. See the report.\n")
    print("| arm | passive | weak | rich | overall | control mean | confounded |")
    print("|---|---|---|---|---|---|---|")
    for arm, record in transplant["arms"].items():
        sens = record.get("sensitivity") or {}
        if "overall" not in sens:
            print(
                f"| `{arm}` | n/a | n/a | n/a | {sens.get('unavailable', 'n/a')} "
                f"(code moved nothing: {sens.get('pairs_the_code_did_not_move', 'n/a')}; "
                f"law inactive: {sens.get('pairs_with_an_inactive_law', 'n/a')}) | n/a | n/a |"
            )
            continue
        cells = []
        for regime in ("passive", "weak", "rich"):
            entry = sens["per_regime"].get(regime)
            cells.append("n/a" if not entry else _f(entry["mean_alignment"]))
        cells.append(_f(sens["overall"]["mean_alignment"]))
        cells.append(_f(sens["overall"].get("mean_control_alignment")))
        cells.append(str(sens["overall"].get("confounded")))
        print(f"| `{arm}` | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
