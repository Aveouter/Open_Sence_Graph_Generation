"""Horizon selection, the separability verdict, and the gate decision.

Three things live here, in the order they depend on each other.

**Horizon.** ``M`` is dispositional, so a law may do nothing at ``X_0`` and only
separate from its neighbours once something has been excited or has fallen into
contact. The horizon therefore has to cover the time to that divergence, and it
is read off the measured curve rather than chosen: ``H`` is the first point where
the mean rich-regime curve has reached its saturated fraction.

**Separability verdict.** A pair is separable in a cell when its ``J_ab`` clears
the null floor's upper quantile rather than a hand-set constant. The floor is
the same-mechanism distance under re-drawn noise, so this reads "the laws differ
by more than noise does".

**The gate.** ``important`` pairs are those that share at least one legal cell --
pairs excluded by the compatibility mask never compete and their inseparability
would mean nothing. When an important pair fails to separate even under the
richest regime, the disposition follows the trend across regimes: rising but not
yet clear means the excitation is insufficient, while flat at the floor across
all three means the two laws are behaviourally the same law and the ontology
should be merged. Neither is available silently -- both require a new ADR.
"""

from __future__ import annotations

from ..simulator.compatibility import ACTIVE_MECHANISMS, compatible_mechanisms
from ..simulator.factors import enumerate_realizable

REGIME_ORDER: tuple[str, ...] = ("passive", "weak", "rich")
SATURATION_FRACTION = 0.9
MIN_HORIZON = 10


def select_horizon(curves: dict, regime: str = "rich") -> dict:
    """Choose ``H`` to cover the *slowest* separation, not the average one.

    The mean curve saturates almost immediately, because most pairs contain a
    law that diverges on the first step under gravity. Reading the horizon off
    that mean would say "one step is enough" while the pairs this study is
    actually about -- the ones that need excitation to separate -- are still
    flat. So the horizon is taken from the per-pair saturation times instead,
    at the upper quantile, and the mean curve is reported beside it for context.
    """
    samples: list[list[float]] = []
    per_pair: list[dict] = []
    for tuple_key in sorted(curves):
        for entry in curves[tuple_key]:
            if entry.regime != regime:
                continue
            curve = list(entry.mean_curve())
            samples.append(curve)
            final = curve[-1] if curve else 0.0
            if final <= 0.0:
                continue
            target = SATURATION_FRACTION * final
            reach = next((h for h in range(len(curve)) if curve[h] >= target), len(curve) - 1)
            per_pair.append(
                {"tuple": tuple_key, "pair": entry.pair, "saturation_horizon": reach, "final": final}
            )
    if not samples:
        raise ValueError(f"no curves for regime {regime!r}")

    horizon = min(len(curve) for curve in samples)
    mean_curve = [sum(curve[h] for curve in samples) / len(samples) for h in range(horizon)]

    if not per_pair:
        return {
            "regime": regime,
            "n_curves": len(samples),
            "final_mean": mean_curve[-1] if mean_curve else 0.0,
            "saturated": False,
            "selected_horizon": horizon,
            "curve": mean_curve,
        }

    reaches = sorted(entry["saturation_horizon"] for entry in per_pair)
    upper = reaches[min(len(reaches) - 1, int(0.9 * len(reaches)))]
    return {
        "regime": regime,
        "n_curves": len(samples),
        "final_mean": mean_curve[-1],
        "saturated": True,
        "mean_curve_saturation_horizon": next(
            (
                h
                for h in range(horizon)
                if mean_curve[h] >= SATURATION_FRACTION * mean_curve[-1]
            ),
            horizon - 1,
        ),
        "per_pair_saturation_quantile_90": upper,
        "per_pair_saturation_max": reaches[-1],
        "per_pair_saturation_min": reaches[0],
        # Clamped to the measured length: a 5-step curve must not report a
        # 10-step horizon, or the verdict would describe more data than exists.
        "selected_horizon": min(max(upper, MIN_HORIZON), horizon),
        "per_pair": sorted(per_pair, key=lambda entry: -entry["saturation_horizon"])[:12],
        "curve": mean_curve,
    }


def important_pairs() -> set[str]:
    """Pairs that share at least one legal cell, so they actually compete."""
    shared: set[str] = set()
    for s0 in enumerate_realizable():
        mechanisms = sorted(compatible_mechanisms(s0))
        for index, first in enumerate(mechanisms):
            for second in mechanisms[index + 1 :]:
                shared.add("/".join(sorted((first, second))))
    return shared


def _pair_of(entry) -> str:
    if "/" in entry.pair:
        return entry.pair
    return "/".join(sorted((entry.pair, "free"))) if entry.pair != "free" else entry.pair


def floor_level(null_curves: dict, quantile: float = 0.95) -> float:
    """Upper quantile of the null-floor AUCs: the bar a ``J_ab`` must clear."""
    scores = sorted(
        entry.auc() for tuple_key in null_curves for entry in null_curves[tuple_key]
    )
    if not scores:
        raise ValueError("no null-floor curves")
    return scores[min(len(scores) - 1, int(quantile * len(scores)))]


def separability_verdict(curves: dict, floor: float, horizon: int) -> dict:
    """Per cell, is the pair's AUC at ``horizon`` above the floor?"""
    cells: dict[str, dict[str, dict]] = {}
    for tuple_key in sorted(curves):
        for entry in curves[tuple_key]:
            curve = entry.mean_curve()[:horizon]
            auc = sum(curve) / len(curve) if curve else 0.0
            cells.setdefault(entry.pair, {})[entry.regime] = {
                "auc": auc,
                "separated": auc > floor,
            }
    return {
        pair: {
            "regimes": regimes,
            "separated_rich": regimes.get("rich", {}).get("separated", False),
            "trend": _trend(regimes),
        }
        for pair, regimes in sorted(cells.items())
    }


def _trend(regimes: dict[str, dict]) -> str:
    values = [regimes.get(name, {}).get("auc", 0.0) for name in REGIME_ORDER]
    if all(values[index] <= values[index + 1] + 1e-9 for index in range(len(values) - 1)):
        return "non_decreasing"
    if values[-1] > values[0] + 1e-9:
        return "rising_with_reversals"
    return "flat_or_falling"


def gate_decision(
    leakage: dict,
    separability: dict,
    precondition: dict,
    level2: dict | None = None,
) -> dict:
    """Combine the four checks into one verdict, with reasons attached.

    Level-2 is optional here so the stdlib gate can report a preliminary verdict
    before the torch oracles have run; ``GO_PHASE_IA`` is never returned while it
    is missing, because the utility check is one of the three the plan requires.
    """
    important = important_pairs()

    # Reasons that block the gate on their own, before Level-2 is even consulted.
    blocking: list[str] = []
    premise_failed = not precondition.get("passed", False)
    if premise_failed:
        blocking.append(
            "null floor is comparable to the between-mechanism signal, so the "
            "deterministic/shared-noise premise does not hold"
        )
    leaked = [
        name
        for name, result in leakage.items()
        if not name.startswith("__") and result.get("significantly_above_baseline", False)
    ]
    if leaked:
        blocking.append(f"leakage probes above baseline: {sorted(leaked)}")
    if not leakage.get("__controls_fired__", False):
        blocking.append(
            "the leaky-sampler control did not fire, so the leakage probe has not "
            "been shown able to detect a leak it is looking for"
        )

    inseparable = sorted(
        pair
        for pair, result in separability.items()
        if pair in important
        and pair != "free"
        and not result["separated_rich"]
        and any(
            result["regimes"].get(regime, {}).get("auc", 0.0) > 0.0
            for regime in REGIME_ORDER
        )
    )

    # Taxonomy: STOP_DATA is for failures of the *data*, STOP for failures of the
    # *learned* check. A leakage probe above baseline, a control that never
    # fired, and a null floor swamping the signal are all statements that the
    # world cannot support the question, and no amount of oracle training would
    # change any of them -- so they are STOP_DATA, not STOP. Only a Level-2
    # utility failure is STOP, because that one is about a model.
    if blocking:
        verdict = "STOP_DATA"
        reasons = blocking
    elif level2 is None:
        verdict = "INCOMPLETE"
        reasons = [
            "Level-2 oracle utility has not been measured yet; the gate requires "
            "identifiability, learned utility and the leakage controls together"
        ]
    elif not level2.get("passed", False):
        verdict = "STOP"
        # Name the parameterisation: the two oracle modes have incomparable loss
        # scales, so a verdict that does not say which one produced it cannot be
        # checked against the artifact later (ADR 0007).
        blocked = ", ".join(level2.get("blocked_by", []) or ["unspecified"])
        reasons = [
            f"Level-2 utility failed ({level2.get('oracle_mode', 'unspecified')} oracle): "
            f"delta_M={level2.get('delta_M')} (ci={level2.get('delta_M_ci')}), "
            f"shuffled_vs_base={level2.get('shuffled_vs_base')}, blocked_by={blocked}"
        ]
    else:
        verdict = "GO_PHASE_IA"
        reasons = []

    return {
        "verdict": verdict,
        "reasons": reasons,
        "important_pairs": sorted(important),
        "inseparable_important_pairs": inseparable,
        "inseparable_dispositions": {
            pair: _disposition(separability[pair]) for pair in inseparable
        },
        "level2_measured": level2 is not None,
    }


def _disposition(result: dict) -> str:
    trend = result.get("trend", "flat_or_falling")
    if trend == "flat_or_falling":
        return "merge_ontology_or_redefine_M"
    return "increase_excitation"


def oracle_mechanisms() -> tuple[str, ...]:
    return ACTIVE_MECHANISMS
