# ADR 0011: pair-known cross-split leakage and the clean-cohort protocol

## Status

Accepted as an audit record for [issue #110](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/110).
Not a protocol freeze, and not a replacement for the historical reports: those
are preserved unmodified, as #110 requires. This record states what was
measured, what is only inferred, and what remains undone.

## Context

The ontology-probe pipeline fits every arm on one fitting set and scores three
eval sets. `eval_matrix.run_split` and `train_probes` both pass
`pair_ood.train` (prior counts, probe fitting) and `pair_ood.dev` (calibration)
for the `iid`, `pair_ood` **and** `pair_known` cells alike; `split_name` changes
only the label an artifact carries. The `pair_ood` arm is held out by
construction under that choice. The `pair_known` arm is not, and nothing checked
it.

`build_pair_known_split` is self-consistent by design: it filters its eval rows
to those whose pair still occurs in its *own* `known.train`, and `audit_split`
confirms `known.train` and `known.eval` are disjoint. Both facts are true and
both are irrelevant to how the split was actually consumed. Two independently
drawn image holdouts over the same 57,723-image pool are disjoint from each
other only by luck, and the split's own audit is blind to the difference.

This is the failure mode that matters: a split can audit itself clean while
being evaluated on rows the model was fitted on, and every recorded provenance
field still reads plausibly.

## Decision

1. Every historical `pair_known` value is labelled **NON_HELD_OUT**. It is not
   held-out performance under the fitting protocol the pipeline uses, and it
   must not be cited as such.
2. A held-out cohort is recovered without retraining, by restricting
   `pair_known.eval` to rows whose image is absent from the actual fitting set.
   This keeps the shared-training-subset design intact, which is what the
   controlled comparison depends on; giving `pair_known` its own
   `known.train`/`known.dev` would be held out by construction but would move
   training volume from 211,219 to 227,869 relations and break that control.
3. That cohort is labelled **CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED**. It
   satisfies the isolation requirement checked here and nothing more.
4. Metric recomputation on it proceeds from saved predictions first. No training
   run, no GPU work, and no new SGG data are authorized by this record.

The historical reports are left exactly as they are. The correction lives here
and in the evidence JSON, not as an edit to a published number.

## Evidence

Reproduce with, from a checkout holding the generated splits and VG150 inputs:

```bash
python -m tools.ontology_probe.audit_cross_split \
  --data-root data/VisualGenome \
  --splits-dir outputs/analysis/ontology_probe/splits \
  --output reproduction/evidence/ontology_probe/cross_split_leakage.json \
  --expect-rel-json-sha256 ab5d435745d5683b4f672311a2ec8618d9126909d291b663deaf406c6497d32f \
  --expect-train-json-sha256 a40fdfdc3894d16349f2e9955e2cbaa199d511c36e25ba46dd6ad8a60043b440 \
  --expect-relation-order-sha256 0996001baec61348f5dda1a7828a851420819a02e2f26882acc923deb95abfe3
```

Inputs verified before measuring: `rel.json` and `train.json` digests, both
split-file digests against `splits_index.json`, and `relation_order_sha256`
against the split artifacts (315,642 relations). A mismatch is fatal, not a
warning.

**Measured.** `pair_ood.train ∪ pair_ood.dev` ∩ `pair_known.eval` =
**46,406 of 62,428 rows (74.34%)** and **9,172 of 11,520 images (79.62%)**.
Contamination is uniform across predicates (0.667–0.819 over the 15 largest).
Control channels read zero: `pair_ood.eval` vs `pair_ood.train`, `pair_known.eval`
vs `pair_known.train`, and VG val images vs the fitting images — the last of
which is what keeps the `iid` arm out of this finding.

**Measured.** The shipped `C_prior_conflict` mask (unordered key, no confidence
floor) counts 27,368 on the historical cohort, matching the published value; a
direction-sensitive `P(r | c_s → c_o)` counts 19,372, so the two definitions
differ by 29% and are reported separately. Confidence floors applied to the
distribution that selected each row give τ≥0.5 → 12,749, τ≥0.7 → 968, **τ≥0.9
→ 0**.

**Measured.** The clean cohort is nonempty: 16,022 rows, 2,348 images, 3,128
unordered pairs, with 6,079 unordered and 4,377 ordered conflict rows. 77.3% of
its rows sit on a pair with support in `pair_ood.train`, so the prior does not
degenerate to the global distribution there.

**Inferred, not measured.** That the historical run used the current code path
rests on the artifacts' own recorded provenance — `training_subset:
"pair_ood.train"` in `probes/vg50__pair_known__B2__s0/metrics.json` and
`controlled_training_subset: "pair_ood.train"` in `matrix/prior_cells.json`,
both consistent with the source at `44c21cc`. No captured invocation was
recovered, so this is inference from artifacts.

**Independently cross-checked.** A second implementation of the isolation check,
written separately, was run against the same inputs and reproduces every figure
above — the two intersections, both wrong-argmax counts, and all six confidence
tiers — and exits with its own "contaminated" status
(`evidence/ontology_probe/independent_cross_check.json`). Two implementations
agreeing is what lets these be treated as measured rather than as one script's
arithmetic.

### Phase 2 — the clean cohort, re-scored

`evidence/ontology_probe/clean_cohort_metrics.json`, produced by
`tools.ontology_probe.rescore_clean_cohort` (torch + numpy; analysis
environment only). It hashes 152 inputs, verifies all twelve `pair_known`
prediction tensors against their split rows **using the `rows` payload stored
inside each tensor rather than the assumed order** (0 misaligned), and
recomputes the metrics on the 16,022-row clean cohort.

Fidelity is the licence to read it: on the *full* cohort the script reproduces
the published `pair_known` values exactly — all nine micro accuracies and all
nine macro recalls, to the four decimals those reports print. It is the same
pipeline on other rows, not a different one.

Measured on the clean cohort, B4 − B2 on micro accuracy is **−0.0110 (vg50),
−0.0090 (L1_noise), −0.0149 (L2_entail)**; on the contaminated cohort it is
+0.0237, +0.0227 and +0.0106. The sign differs in all three label spaces. On
vg50 the clean cohort gives VRR 0.1289, harm 0.1004, net −0.0110 against the
contaminated cohort's 0.1767 / 0.0470 / +0.0237, and VRR at τ>0.9 is 0.0000 in
every cohort and every label space. Intervals are 2000-draw percentile
bootstraps resampling images, not rows.

**This does not retract the historical reading.** The paired difference on a
fixed cohort is internally valid — the same rows, neither arm fitted on the
clean ones — but the clean cohort is the greedy holdout's complement, so it is
`CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED` and its *levels* are not population
estimates. Selection bias is quantified rather than asserted: predicate-mix TVD
is 0.0894, and every one of the fifteen largest predicates is harder there
(`behind` 0.509 against 0.738, `near` 0.421 against 0.633). The two confounds
that would separate a capacity effect from a visual-evidence effect are both
absent: B2 and B4 differ in parameter count, and there is a single seed, so the
McNemar p-values test rows within one trained pair and say nothing about
seed-to-seed variance. Recording the flip is warranted; calling it the true
effect is not.

**Still not done.** No Δ_ontology, shuffling or per-predicate intervention
figure has been recomputed on the clean cohort — only the prior-free probe arms
above. The published Δ_ontology and VRR values in reports 04 and 05 stand as
published values on a non-held-out cohort.

## Consequences

Two limits bound what this record supports, and both are load-bearing.

**The conflict-rate gap is not established as an effect of contamination.**
The contaminated portion shows 45.87% and the clean remainder 37.94%, a gap of
7.93 points. The two portions are not comparable samples: the clean rows are
those the greedy pair-OOD holdout did not commit, which is a selection on the
same image statistics that drive conflict. The gap therefore indicates that the
historical aggregate carries selection-bias risk; it does not show that
contamination causes the difference, and no causal reading is licensed.

**The clean subset is not an unbiased test set.** It meets the isolation
condition checked here, but it inherits whatever selection effect the pair-OOD
greedy construction induces, and it is 26% of the original eval rows. Results on
it are labelled `CLEAN_SUBSET / SELECTION_BIAS_NOT_EXCLUDED`, never "clean" or
"held-out" without that qualifier.

What may be claimed: that the historical `pair_known` cells are not held-out,
as a measurement; that a nonempty cohort satisfying isolation exists; and the
cohort sizes and conflict counts above. What may not be claimed: corrected
accuracies, that any visual-evidence conclusion changes, or that the `iid` and
`pair_ood` cells are affected — they are not, and the control channels show it.

Next: recompute the Δ_ontology and intervention figures on the clean cohort the
same way, and decide separately whether the greedy holdout needs a construction
fix. #112 remains deferred; no GPU work is authorized here.

## Review checklist

Carried over from the transcription PR this record supersedes, so that closing
it loses nothing:

- [x] Full raw-input SHA256 manifest attached, not abbreviated hashes —
      `evidence/ontology_probe/inputs_sha256.json`, 152 files.
- [x] The committed verifier re-run on the real inputs, with its complete output
      committed (`independent_cross_check.json`), raw VG data still untracked.
- [x] Every reported cell cross-checked, including confidence-threshold
      inclusivity. The threshold convention differs between the two scripts and
      is recorded in both rather than reconciled silently: `audit_cross_split`
      bins with `>=`, `rescue_rate` (and therefore the phase-2 block) uses
      strictly `>`.
- [x] Saved B2/B3/B4 predictions verified against the clean row/image indices
      before any corrected metric was computed; 0 misaligned of 12 cells.
- [x] Source provenance limitations preserved: the `training_subset` stamps and
      HEAD code agree, but no historical command invocation was reconstructed.
- [x] Selection bias investigated rather than assumed away — predicate-mix TVD,
      per-predicate difficulty, and image-cluster intervals are all reported.
- [x] All historical reports preserved unchanged.

Not covered here, and left open: the Δ_ontology and shuffle/intervention
figures, and any construction change to the greedy holdout.
