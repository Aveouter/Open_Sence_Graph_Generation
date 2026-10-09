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

**Not done.** No metric on the clean cohort has been recomputed. The historical
`pair_known` accuracies, macro recalls, Δ_ontology and VRR figures stand as
published values on a non-held-out cohort; nothing here replaces them.

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

Next: recompute the `pair_known` cells on the clean cohort from saved
predictions under #110, and decide separately whether the greedy holdout needs a
construction fix. #112 remains deferred; no GPU work is authorized here.
