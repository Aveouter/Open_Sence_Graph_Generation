# ADR 0007: Emergence Evidence Criteria

## Status

Accepted

## Context

Phase I is at risk of a specific failure mode: producing a pipeline that runs,
yields numbers, and is then described with language the evidence does not
support. The plan names three primary endpoints — cross-filler generalization,
role-structured geometry, and counterfactually portable predictive use — and
requires all three before the phrase "abstract relational representation
emerged" may be used. It does not say what makes the underlying gates falsifiable,
and it leaves open how the first delivery's own verdict is decided.

Two further hazards are concrete in this repository. A gate that can only pass is
not a gate; each judgement here can fail silently if its detector is too weak, and
"not significant" is routinely read as "clean" when it means "underpowered". And
the repository's claim guardrail scans text for language that overstates evidence,
so the vocabulary used in reports is part of the protocol, not decoration.

## Decision

**Pre-register the claim ladder, the gate criteria, and the reporting contract
before any data is generated.**

### Claim ladder

The strongest phrase permitted is indexed by what is actually demonstrated:

| Demonstrated | Permitted description |
|---|---|
| Ordinary probe decodes mechanism | relation information is decodable |
| Cross-condition generalization only | relation coding generalizes across conditions |
| Cross-condition + role structure, transplant fails | no causal claim on the relation variable |
| Cross-condition + role structure + counterfactual transplant | abstract relational representation emerged |

Relation probes, embedding clustering, `Gamma`, and UMAP are **diagnostics only**
and may never be cited as primary evidence for the last row.

### Gate criteria (first delivery)

- **M0 leakage.** `X0-only -> M`, `X0+F -> M` and `S0-only -> M` are each compared
  against the analytic compatibility-aware baseline, defined as the conditional
  majority class **per realizable joint `S_0` tuple** over the enumerated
  `C(S0_tuple, M)` table. No fixed 25% or 33% chance figure is used, because the
  compatible mechanism count varies by tuple. Neither `X0-only -> M` nor
  `X0+F -> M` may significantly exceed that baseline.
  Detector: a flexible MLP of the same family and capacity as the Level-2 oracle;
  significance by counterfactual-group-blocked permutation test with family-wise
  control. Thresholds are pre-registered in the frozen config.
- **Level-1 identifiability.** `J_ab(H)` per mechanism pair and intervention
  regime, primary scalar `J_ab^AUC`, with `floor_num` (same seed, different
  compute path) and `floor_null` (same mechanism, independent seed) reported
  separately. `floor_null` comparable in magnitude to cross-mechanism `J_ab` is
  its own **precondition gate** and yields `STOP_DATA`.
- **Level-2 oracle utility.** `Delta_M = L_ShuffledM - L_TrueM` with three
  capacity-identical oracles. `TrueM` must significantly beat `ShuffledM`, and
  `ShuffledM` must not systematically beat `Base`. **This ADR originally left the
  oracle's parameterisation unspecified, which was a mistake** — see the
  amendment below. The parameterisation is part of the criterion, not an
  implementation detail of it, and must be named wherever a verdict is cited.
- **Verdict.** `GO_PHASE_IA` requires Level-1, Level-2 and the leakage controls
  to pass together. Otherwise `STOP`, `STOP_DATA`, or `not_reproduction_ready` as
  appropriate.

Terminology to use, in this order of preference: `implementation_audit`,
`deferred_reproduction`, `protocol_mismatch`, `not_reproduction_ready`,
`pipeline_smoke_only`. None of these may be upgraded without the evidence above.

### Amendment: the Level-2 oracle parameterisation

**The gap this closes.** The criterion above fixed the comparison -- paired
`delta_M` over capacity-identical arms -- but not what the arms *are*. The first
implementation chose a **direct** head, predicting `H * 8 = 480` values from a
single snapshot. Measured, it overfits across base contexts: training loss
0.49–0.66, validation 0.80–0.97, best epoch 2–11 of 200, against a
mean-predictor reference of 1.05–1.16. All three arms sat near the trivial
predictor and `delta_M` came out at `+0.0019`, CI [−0.0023, +0.0061]. The gate
returned `STOP` on a measurement that could not distinguish "the mechanism
carries no utility" from "this oracle cannot see it", which is precisely the
distinction M0 closes with a pre-registered minimum detectable effect.

**Decision.** The oracle is **rollout mode**: trained on
`(X_t, F, A_t) -> delta X_{t+1}` and scored by rolling the transition out
autoregressively for **10 steps** — the horizon Level-1 selected from the `J_ab`
saturation curve, so the oracle is judged over the same interval the gate treats
as covering the interaction consequence. The direct head is retained as a
secondary mode and its result is reported beside, not instead of, the rollout's.

Four implementation decisions are load-bearing and each was found by measuring
rather than by reasoning:

1. **Residual targets.** Predicting `delta X` rather than `X` makes the identity
   map mean "nothing moved", which is a sane baseline for a one-step physical
   transition and keeps an autoregressive rollout on the data manifold.
2. **A bounded evaluation horizon.** An autoregressive rollout of all 60 steps
   compounds its own error until the loss measures divergence rather than
   information — measured, it exploded to O(10^3) with a trivial predictor
   scoring O(1).
3. **Every input block standardised, including the filler block.** It holds the
   masses, which run to about 18 where every other block is order 1. Left raw it
   dominates the first layer and the outputs diverge.
4. **State statistics pooled across the horizon.** `dataset.y` is the whole
   trajectory flattened, so a state dimension lives at `step * w_y + dim`.
   Taking the first `w_y` columns scales by the release state alone, which is far
   quieter than the rest of the trajectory.

**Why the compounding argument does not block this.** `oracles.OracleConfig`
argues against a rollout, on the grounds that compounded early error would mix
"knows more about `M`" with "accumulated error faster". That is right about an
absolute loss and wrong about `delta_M`: the three arms are paired, share weights
and batch order, and differ only in the label, so compounding affects them alike
— and where a better `M` yields a better one-step prediction, compounding
*amplifies* the difference being measured rather than confounding it. What the
mode gives up is reading `delta_M` as "information per step". What it buys is a
comparison that is measurable at all. The two modes' loss scales are **not**
comparable, and the artifact records which one produced a verdict.

### Pre-registered witness for the intervention claim

The claim "identifiable interaction experience is what makes relational
abstraction possible" is only testable if some mechanism pair is *passively
indistinguishable and actively separable*. That pair is pre-registered here as
the load-bearing cell:

> At least one important pair is **passively inseparable and actively
> separable**: its ``J_ab`` under the passive regime is below the null floor's
> 95th percentile, while its ``J_ab`` under the rich regime is above it.

The criterion is stated against the floor rather than as "the three laws produce
identical passive trajectories", because a *strict* equality is a claim about the
implementation rather than about the world, and the measured passive separation
at the ``inside`` tuples is small but not zero (drift accumulates over a
horizon). What makes the intervention claim testable is the ratio and the
floor-crossing, not exact equality.

If no pair crosses, the intervention claim is reported as **unresolved**, not as
negative: an experiment that never built the witness cannot testify about it
either way.

### Falsifiability requirements

- Every gate ships with a **must-fire** and a **must-pass** control built through
  the same code path: a deliberately leaky sampler for M0; definitionally
  identical mechanism pairs and known-large-divergence pairs for Level-1; a
  synthetic task in which `M` fully determines the future for Level-2.
- Every significance judgement pre-registers a **minimum detectable effect**.
  Effects below it are reported as *undetectable at this N*, never as absent.
- When a mechanism pair is inseparable under the richest regime, the disposition
  is chosen by the `J_ab` trend across regimes: rising but not separable means
  increase excitation; flat at the floor across all three regimes means the two
  laws are behaviourally equivalent and the ontology should be merged. Both
  dispositions require a new ADR.

### Reporting contract

- Reports separate **Measured** / **Interpretation** / **Not claimed**.
- A report may only quote numbers that exist in an artifact, and must record that
  artifact's hash and the command that produced it.
- Every artifact carries `status`, `not_a_reproduction: true`, `scope`, `world`,
  `config_sha256`, `git_sha`, `schema_version`, `seed`. Generated artifacts stay
  untracked; the frozen config, schema validator, ADRs, artifact manifest and
  hand-written reports are tracked.
- **PR scope for this work:** simulator, M0 audits and the M1 gates only. No
  Predictive-Global or Predictive-Relational model, and no CCGP, role
  equivariance or transplant evaluator, is started under this ADR.

## Evidence

- Plan/source: Phase I plan text supplied 2026-09-23 (not tracked in this
  repository). Section references: primary endpoints, the claim gate, the
  decision matrix, the milestone order.
- Local standard for claim discipline: `AGENTS.md`,
  `reproduction/adr/0001-reproduction-claim-standard.md`,
  `reproduction/adr/0002-random-init-is-not-reproduction.md`.
- Local standard for thresholds written into artifacts: `tools/ontology_probe/README.md`,
  `tools/ontology_probe/diagnostics_table.py` (`VERDICT_RULES`).

## Consequences

- **Can be claimed:** that the criteria for every verdict in the first delivery
  were fixed before the data existed, and that each gate has a demonstrated
  failure mode.
- **Cannot be claimed:** any representation-learning finding. The first delivery
  is an instrument check. `CCGP`, `RoleEquivariance` and `CFError` remain
  pre-registered endpoints with no measured value. Phase IB additionally
  pre-registers a **per-factor** filler-OOD readout for each `S_0` factor
  (`containment`, `contact`, `pose`) and an **unseen factor-combination**
  compositional generalization test; neither is measured in the first delivery.
- **Done since this was written:** the frozen config carries
  `leakage_mde_target` and the driver refuses to describe a run as powered
  unless the achieved minimum detectable effect meets it; each gate has a
  must-fire control built through the same code path; the protocol validator
  exists and its checks were verified to fail on a tampered artifact set.
- **Outcome of the first delivery:** the gate ran and returned `STOP`, on the
  Level-2 utility check alone. Level-1 passed decisively and M0 was clean. See
  `outputs/reports/relational_emergence/` for the measurement and its limits.
- **Closed since this was written:** the Level-2 parameterisation gap, by the
  amendment above, and the absent-versus-undetectable gap, by a pre-registered
  minimum detectable effect. `OracleConfig.delta_m_mde_target` declares the
  smallest `delta_M` a run must be able to resolve (0.05, a five-percent
  reduction in prediction error), the result reports the achieved MDE beside it,
  and a non-significant `delta_M` is now blocked as `delta_M_underpowered` when
  the run could not have seen an effect that size, distinct from
  `delta_M_not_significantly_positive` when it could. The first target tried,
  0.01, is unreachable at any feasible seed count — roughly 260 seeds — which is
  itself the argument for declaring a target rather than assuming one.
