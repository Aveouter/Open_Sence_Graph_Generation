"""Trained nonlinear M0, conditional held-out permutation max test and calibration.

Test labels are permuted jointly within each compatible base group, identically
across its regime records. Models are fitted without the test labels and held
fixed for this conditional randomization test. All effects average base groups,
not correlated mechanism/regime records.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn

from .protocol import MECHANISMS, Protocol

FEATURES = {
    "S0": ("s0",),
    "F": ("filler",),
    "S0+F": ("s0", "filler"),
    "X0": ("x0",),
    "X0+F": ("x0", "filler"),
    "X0+S0+F": ("x0", "s0", "filler"),
}


@dataclass(frozen=True)
class AuditConfig:
    epochs: int = 120
    hidden: int = 32
    permutations: int = 199
    calibration_repeats: int = 20
    seed: int = 109

    def __post_init__(self):
        if (
            min(self.epochs, self.hidden, self.permutations, self.calibration_repeats)
            < 1
        ):
            raise ValueError("audit counts must be positive")


def records_from_scenes(rows):
    """Deduplicate regimes: there is only one initial-observable audit per twin."""
    tuples = sorted({r["tuple"] for r in rows})
    records = {}
    for row in rows:
        record = {
            "group": row["group"],
            "stratum": tuple(row["compatible"]),
            "label": MECHANISMS.index(row["mechanism"]),
            "s0": [float(row["tuple"] == key) for key in tuples],
            "filler": [v for obj in row["structural"] for v in obj] + row["nuisance"],
            "x0": [v for obj in row["states"][0] for v in obj],
        }
        key = (row["group"], row["mechanism"])
        if key in records and records[key] != record:
            raise ValueError("initial observables differ across regimes")
        records[key] = record
    return list(records.values())


def group_split(records, seed):
    strata = {}
    for record in records:
        strata.setdefault(record["stratum"], set()).add(record["group"])
    train, test = set(), set()
    rng = random.Random(seed)
    for groups in strata.values():
        groups = sorted(groups)
        if len(groups) < 4:
            raise ValueError(
                "M0 requires at least four base groups per compatible stratum"
            )
        rng.shuffle(groups)
        test.update(groups[: len(groups) // 2])
        train.update(groups[len(groups) // 2 :])
    if train & test:
        raise ValueError("base group appears in multiple strata or partitions")
    return (
        [i for i, r in enumerate(records) if r["group"] in train],
        [i for i, r in enumerate(records) if r["group"] in test],
    )


def permute_labels(records, seed):
    """One random label bijection per whole base group, confined to its stratum."""
    rng, maps = random.Random(seed), {}
    for record in records:
        key = record["group"]
        if key not in maps:
            original = [MECHANISMS.index(m) for m in record["stratum"]]
            shuffled = original[:]
            rng.shuffle(shuffled)
            maps[key] = dict(zip(original, shuffled, strict=True))
    return torch.tensor([maps[r["group"]][r["label"]] for r in records])


def _group_average(values, records):
    groups = sorted({r["group"] for r in records})
    return torch.stack(
        [
            values[[i for i, r in enumerate(records) if r["group"] == g]].mean(0)
            for g in groups
        ]
    )


def evaluate(records, protocol: Protocol, config: AuditConfig, *, features=FEATURES):
    train, test = group_split(records, config.seed)
    labels = torch.tensor([r["label"] for r in records])
    mask = torch.tensor([[m in r["stratum"] for m in MECHANISMS] for r in records])
    baseline = mask.sum(1).double().log()
    reports, heldout = {}, []
    test_records = [records[i] for i in test]
    groups = sorted({r["group"] for r in test_records})
    membership = torch.tensor(
        [[float(r["group"] == g) for r in test_records] for g in groups]
    ).double()
    membership /= membership.sum(1, keepdim=True)
    # Equal weight per base group, even when compatibility permits 3 vs 4 laws.
    weights = membership.mean(0)
    bootstrap_rng = torch.Generator().manual_seed(config.seed + 2909)
    bootstrap_indices = torch.randint(
        len(groups), (1000, len(groups)), generator=bootstrap_rng
    )
    for name, columns in features.items():
        x = torch.tensor(
            [[v for column in columns for v in r[column]] for r in records],
            dtype=torch.float32,
        )
        mean, std = x[train].mean(0), x[train].std(0, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / std
        for estimator in ("linear", "mlp"):
            torch.manual_seed(config.seed)
            if estimator == "linear":
                model = nn.Linear(x.shape[1], len(MECHANISMS))
            else:
                model = nn.Sequential(
                    nn.Linear(x.shape[1], config.hidden),
                    nn.Tanh(),
                    nn.Linear(config.hidden, config.hidden),
                    nn.Tanh(),
                    nn.Linear(config.hidden, len(MECHANISMS)),
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(config.epochs):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[train]).masked_fill(~mask[train], -1e9)
                loss = nn.functional.cross_entropy(logits, labels[train])
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logp = (
                    model(x[test])
                    .double()
                    .masked_fill(~mask[test], -1e9)
                    .log_softmax(1)
                )
            if not torch.isfinite(logp).all():
                raise ValueError("nonfinite leakage probe")
            ce = -logp[torch.arange(len(test)), labels[test]]
            effects = membership @ (baseline[test] - ce)
            effect = float((baseline[test] - ce) @ weights)
            se = float(effects.std(unbiased=True) / len(effects) ** 0.5)
            bootstrap = effects[bootstrap_indices].mean(1)
            name_key = f"{name}/{estimator}"
            reports[name_key] = {
                "delta_ce": effect,
                "cluster_se": se,
                "cluster_bootstrap_ci": bootstrap.quantile(
                    torch.tensor([0.025, 0.975], dtype=torch.float64)
                ).tolist(),
                "ci_scope": "marginal diagnostic; decisions use family-wise permutation p",
                "accuracy": float((logp.argmax(1) == labels[test]).float().mean()),
                "accuracy_effect": float(
                    (
                        (logp.argmax(1) == labels[test]).float() - 1 / mask[test].sum(1)
                    ).mean()
                ),
            }
            heldout.append(logp)
    observed = torch.tensor([r["delta_ce"] for r in reports.values()])
    maxima = []
    log_probabilities = torch.stack(heldout)
    for permutation in range(config.permutations):
        labels_null = permute_labels(test_records, config.seed + 10000 + permutation)
        effects = (
            baseline[test] + log_probabilities[:, torch.arange(len(test)), labels_null]
        ) @ weights
        maxima.append(float(effects.max()))
    for report in reports.values():
        report["p_fwer"] = (
            # A numerical tie is an exceedance, never evidence of leakage.
            # This tolerance is far below the 0.05-nat practical resolution.
            1 + sum(value >= report["delta_ce"] - 1e-12 for value in maxima)
        ) / (1 + len(maxima))
    detected = any(
        r["delta_ce"] > 1e-12 and r["p_fwer"] <= protocol.alpha
        for r in reports.values()
    )
    return {
        "probes": reports,
        "detected": detected,
        "max_delta_ce": float(observed.max()),
        "train_groups": len({records[i]["group"] for i in train}),
        "test_groups": len({records[i]["group"] for i in test}),
        "permutation_unit": "compatible base-group label bijection",
        "test": "conditional held-out family maximum; trained models fixed",
    }


def inject_nonlinear(records, information_ce, seed):
    """Erasure-channel control with exactly `information_ce` nats of Bayes signal.

    Revealed labels require multiplying a random sign by a signed one-hot vector.
    No single channel carries the class. A group-level Bernoulli flag reveals all
    of that group's compatible labels with probability information_ce/log(K).
    The flag, random sign and signed vector are appended to F, preserving groups.
    """
    rng, revealed = random.Random(seed), {}
    output = []
    for record in records:
        probability = information_ce / math.log(len(record["stratum"]))
        if not 0 <= probability <= 1:
            raise ValueError("injected CE exceeds the compatible-label entropy")
        if record["group"] not in revealed:
            revealed[record["group"]] = rng.random() < probability
        sign = rng.choice((-1.0, 1.0))
        vector = [sign * float(k == record["label"]) for k in range(len(MECHANISMS))]
        if not revealed[record["group"]]:
            vector = [0.0] * len(MECHANISMS)
        output.append({**record, "filler": [*record["filler"], sign, *vector]})
    return output


def run_audit(records, protocol, config):
    observed = evaluate(records, protocol, config)
    controls = {}
    for name, strength in (
        ("weak", 0.5),
        ("mde", 1.0),
        ("strong", 4.0),
        ("random_labels", 4.0),
    ):
        detected, effects = 0, []
        for repeat in range(config.calibration_repeats):
            seed = config.seed + 1000 * (repeat + 1)
            sample = inject_nonlinear(records, protocol.leakage_mde_ce * strength, seed)
            if name == "random_labels":
                labels = permute_labels(sample, seed + 1).tolist()
                sample = [
                    {**r, "label": label}
                    for r, label in zip(sample, labels, strict=True)
                ]
            result = evaluate(sample, protocol, config)
            detected += int(result["detected"])
            effects.append(result["max_delta_ce"])
        controls[name] = {
            "detections": detected,
            "trials": config.calibration_repeats,
            "rate": detected / config.calibration_repeats,
            "effects_ce": effects,
            "injected_information_ce": protocol.leakage_mde_ce * strength,
        }
        print(f"[v2 M0] {name}: {detected}/{config.calibration_repeats}", flush=True)
    # A tiny calibration can debug code but cannot certify the declared power.
    resolution = 1 / (config.permutations + 1)
    adequate = (
        config.calibration_repeats >= 20
        and resolution <= protocol.alpha
        and controls["mde"]["rate"] >= protocol.power
        and controls["strong"]["rate"] >= protocol.power
        and controls["random_labels"]["rate"] <= protocol.alpha
    )
    verdict = (
        "FAIL_LEAKAGE"
        if observed["detected"]
        else "PASS_NO_DETECTABLE_LEAKAGE"
        if adequate
        else "FAIL_AUDIT_POWER"
    )
    return {
        "verdict": verdict,
        "observed": observed,
        "controls": controls,
        "preregistered_mde_ce": protocol.leakage_mde_ce,
        "target_power": protocol.power,
        "power_is_empirical": True,
        "family": list(FEATURES),
        "estimators": ["linear", "trained_mlp"],
    }
