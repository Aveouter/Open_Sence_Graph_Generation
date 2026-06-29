#!/usr/bin/env python3
"""EXP-012 Step 4: Aggregate prior baseline on selected splits.

Reads:
  outputs/gen_sgg/exp012_split_audit/audit_C_seed{42,123,2027}.json
  outputs/gen_sgg/exp012_split_audit/audit_E_seed{42,123,2027}.json

Writes:
  outputs/gen_sgg/exp012_split_audit/prior_baseline_on_selected_splits.json

This is the prior baseline that TPVB must beat on the selected splits.
"""
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = _PROJECT_ROOT / "outputs" / "gen_sgg" / "exp012_split_audit"

SEEDS = [42, 123, 2027]
SPLITS = ["C", "E"]  # selected main + auxiliary


def main():
    summary = {}
    for split in SPLITS:
        entries = []
        for seed in SEEDS:
            with open(OUT / f"audit_{split}_seed{seed}.json") as f:
                entries.append(json.load(f))
        n_list = [e["hidden_positive_count"] for e in entries]
        ec_r1 = [e["ec_r1"] for e in entries]
        ec_r3 = [e["ec_r3"] for e in entries]
        ec_r5 = [e["ec_r5"] for e in entries]
        ec_mr = [e["ec_mean_rank"] for e in entries]
        pf_r1 = [e["pf_r1"] for e in entries]
        pf_r3 = [e["pf_r3"] for e in entries]
        pf_r5 = [e["pf_r5"] for e in entries]
        leak = [e["conditional_leak"] for e in entries]
        family = entries[0]["family_breakdown"]
        # Check 3-seed agreement on the family breakdown
        for e in entries[1:]:
            for k, v in e["family_breakdown"].items():
                family[k] = family.get(k, 0) + v
        family = {k: round(v / len(SEEDS), 1) for k, v in family.items()}
        summary[split] = {
            "split_full_name": entries[0]["split"],
            "hidden_positive_count_per_seed": dict(zip(SEEDS, n_list)),
            "hidden_positive_count_seed_mean": sum(n_list) / len(n_list),
            "ec_r1_seed_mean": sum(ec_r1) / len(ec_r1),
            "ec_r3_seed_mean": sum(ec_r3) / len(ec_r3),
            "ec_r5_seed_mean": sum(ec_r5) / len(ec_r5),
            "ec_mean_rank_seed_mean": sum(ec_mr) / len(ec_mr),
            "pf_r1_seed_mean": sum(pf_r1) / len(pf_r1),
            "pf_r3_seed_mean": sum(pf_r3) / len(pf_r3),
            "pf_r5_seed_mean": sum(pf_r5) / len(pf_r5),
            "conditional_leak_seed_mean": sum(leak) / len(leak),
            "family_breakdown_seed_mean": family,
        }

    # TPVB success thresholds on selected splits.
    # Required: TPVB hidden R@1 > EC R@1 + 0.05; TPVB hidden R@3 > EC R@3 + 0.20.
    summary["tpvb_success_threshold"] = {
        "C": {
            "required_r1": summary["C"]["ec_r1_seed_mean"] + 0.05,
            "required_r3": summary["C"]["ec_r3_seed_mean"] + 0.20,
            "required_r5": summary["C"]["ec_r5_seed_mean"] + 0.30,
            "main_R1_floor": 0.625,  # vs TokenSingleSoftmax 0.644
        },
        "E": {
            "required_r1": summary["E"]["ec_r1_seed_mean"] + 0.05,
            "required_r3": summary["E"]["ec_r3_seed_mean"] + 0.20,
            "required_r5": summary["E"]["ec_r5_seed_mean"] + 0.30,
            "main_R1_floor": 0.625,
        },
    }

    with open(OUT / "prior_baseline_on_selected_splits.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
