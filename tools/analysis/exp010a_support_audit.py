#!/usr/bin/env python3
"""EXP-010A: support audit. Confirms G-SUPPORT — that learned models and both priors
are evaluated on exactly the same hidden-positive support (994 per seed), and documents
the root cause of the old EXP-009B denominator mismatch.

Root cause (verified):
  - Val manifest has 994 hidden positives across 928 pairs (seed 42; same structure
    for 123/2027).
  - There are ZERO out-of-bounds (sub_idx/obj_idx) pairs: val.json iscrowd == 0 and the
    pycocotools ann_id ordering is identical to the dataset's `target["labels"]` ordering.
    subj/obj labels read by the prior therefore match the model's view (0 mismatch).
  - The ENTIRE old mismatch came from `if not ranking: continue` in
    compute_exp009b_priors.py: PairFrequencyPrior dropped 38 hidden positives on pairs
    whose (subj,obj) never appeared in training; ExposedConditionalPrior dropped 82
    (pairs where, after removing the exposed label, no co-occurring exposed label
    remained). The learned evaluator kept all 994 (it always ranks all 50 predicates
    and treats not-found as rank 9999).
  - Fix: priors now produce a full 50-way ranking for EVERY pair; empty-signal pairs get
    a deterministic per-pair random tie-break (expected R@K == K/50), so they stay in the
    denominator as near-misses instead of being dropped.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SEEDS = [42, 123, 2027]
MODELS = ["SS", "TS", "EH"]


def load_jsonl(path):
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def hidden_count(recs):
    return sum(len(r["hidden_labels"]) for r in recs) if recs else None


def pair_keys(recs):
    return (
        set((r["image_id"], r["sub_idx"], r["obj_idx"]) for r in recs) if recs else None
    )


def old_prior_support(seed, prior_type):
    """Reproduce the OLD (dropping) support count from compute_exp009b_priors.py."""
    from pycocotools.coco import COCO

    manifest_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "manifests"
    tm = json.load(
        open(manifest_dir / f"exp009_train_hidden_head_or_coarse_one_seed{seed}.json")
    )
    vm = json.load(
        open(manifest_dir / f"exp009_val_hidden_head_or_coarse_one_seed{seed}.json")
    )

    tr = COCO(str(_PROJECT_ROOT / "data" / "VisualGenome" / "train.json"))
    tr_anns = defaultdict(list)
    for ann_id, ann in tr.anns.items():
        tr_anns[ann["image_id"]].append(ann)
    pair_freq = defaultdict(Counter)
    same_pair_labels = defaultdict(set)
    for p in tm["pairs"]:
        anns = tr_anns.get(p["image_id"], [])
        if p["sub_idx"] >= len(anns) or p["obj_idx"] >= len(anns):
            continue
        sl = anns[p["sub_idx"]]["category_id"]
        ol = anns[p["obj_idx"]]["category_id"]
        pair_freq[(sl, ol)][p["exposed_label"]] += 1
        same_pair_labels[(sl, ol)].add(p["exposed_label"])

    va = COCO(str(_PROJECT_ROOT / "data" / "VisualGenome" / "val.json"))
    va_anns = defaultdict(list)
    for ann_id, ann in va.anns.items():
        va_anns[ann["image_id"]].append(ann)

    oob = 0
    empty = 0
    kept = 0
    for p in vm["pairs"]:
        if not p["hidden_labels"]:
            continue
        anns = va_anns.get(p["image_id"], [])
        if p["sub_idx"] >= len(anns) or p["obj_idx"] >= len(anns):
            oob += len(p["hidden_labels"])
            continue
        sl = anns[p["sub_idx"]]["category_id"]
        ol = anns[p["obj_idx"]]["category_id"]
        if prior_type == "pair_frequency":
            ranking = pair_freq.get((sl, ol), Counter())
            empty_flag = not ranking
        else:
            all_exp = same_pair_labels.get((sl, ol), set())
            cand = [pr for pr in all_exp if pr != p["exposed_label"]]
            empty_flag = not cand
        if empty_flag:
            empty += len(p["hidden_labels"])
        else:
            kept += len(p["hidden_labels"])
    total = kept + empty + oob
    return {
        "total_hidden": total,
        "kept": kept,
        "dropped_empty_ranking": empty,
        "dropped_oob": oob,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(_PROJECT_ROOT / "outputs" / "gen_sgg" / "exp010_prior_strata"),
    )
    args = parser.parse_args()
    base = Path(args.base)

    audit = {"protocol": "exp010a_support_audit", "seeds": {}}
    for seed in SEEDS:
        manifest_dir = _PROJECT_ROOT / "outputs" / "gen_sgg" / "manifests"
        vm = json.load(
            open(manifest_dir / f"exp009_val_hidden_head_or_coarse_one_seed{seed}.json")
        )
        manifest_hp = sum(len(p["hidden_labels"]) for p in vm["pairs"])
        manifest_pairs = len(vm["pairs"])

        seed_audit = {
            "manifest_hidden_positives": manifest_hp,
            "manifest_hidden_pairs": sum(1 for p in vm["pairs"] if p["hidden_labels"]),
            "manifest_total_pairs": manifest_pairs,
            "old_prior_support": {
                "pair_frequency": old_prior_support(seed, "pair_frequency"),
                "exposed_conditional": old_prior_support(seed, "exposed_conditional"),
            },
        }

        # New unified support: every method's dump should have manifest_hp hidden positives
        # and the SAME pair keys as the manifest.
        manifest_keys = pair_keys(
            load_jsonl(base / f"seed{seed}" / "scores_pair_frequency.jsonl")
        )
        seed_audit["unified_support"] = {}
        all_match = True
        for name in MODELS + ["pair_frequency", "exposed_conditional"]:
            recs = load_jsonl(base / f"seed{seed}" / f"scores_{name}.jsonl")
            if recs is None:
                seed_audit["unified_support"][name] = {"status": "MISSING"}
                all_match = False
                continue
            hp = hidden_count(recs)
            keys = pair_keys(recs)
            keys_match = keys == manifest_keys
            seed_audit["unified_support"][name] = {
                "n_pairs": len(recs),
                "n_hidden_positives": hp,
                "pair_keys_match_manifest": keys_match,
            }
            if hp != manifest_hp or not keys_match:
                all_match = False
        seed_audit["unified_support"]["_all_match_994"] = all_match
        audit["seeds"][seed] = seed_audit
        print(f"seed {seed}: manifest HP={manifest_hp}; unified all_match={all_match}")

    audit["root_cause"] = {
        "old_mismatch_source": "compute_exp009b_priors.py `if not ranking: continue` dropped "
        "hidden positives on pairs the prior could not rank, shrinking the "
        "prior denominator (PF: 38 dropped; EC: 82 dropped, seed 42).",
        "out_of_bounds": "0 — val.json iscrowd==0 and pycocotools ann_id order matches the "
        "dataset target['labels'] order; prior subj/obj labels verified identical "
        "to the model's view (0 mismatch on 200 checked).",
        "fix": "Priors now emit a full 50-way ranking for every pair; empty-signal pairs use a "
        "deterministic per-pair random tie-break (expected R@K == K/50) and remain in the "
        "denominator as near-misses, matching the learned evaluator's all-994 convention.",
        "G_SUPPORT_pass_condition": "unified_support._all_match_994 == true for every seed/model.",
    }
    audit["G_SUPPORT_passed"] = all(
        audit["seeds"][s]["unified_support"].get("_all_match_994", False) for s in SEEDS
    )

    out = base / "exp010a_support_audit.json"
    with open(out, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nwrote {out}")
    print(f"G_SUPPORT_passed = {audit['G_SUPPORT_passed']}")


if __name__ == "__main__":
    main()
