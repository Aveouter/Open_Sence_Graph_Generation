# Phase I v2 development candidate

This is the versioned implementation and development gate for
[Issue #107](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/107).
It is an `implementation_audit`, not a completed Phase I study or a baseline
reproduction. See [ADR 0010](../../../reproduction/adr/0010-phase1-v2-development-boundary.md)
and the [development report](../../../outputs/reports/relational_emergence/v2/01_development_gate.md).
The sibling v1 modules remain historical and continue to use their original API.

```text
data.build_scenes             four-object shared-supporter counterfactual contexts
  data.build_batch           history through cutoff / separate future targets
    model.infer_mechanism    history-derived edge codes or one global scene code
    model.rollout           predicted state -> next predicted state
      transplant            fixed target inputs, matched correct/wrong source codes
run                         world + chronology + calibrated M0 + evaluator fixtures
  failed prerequisite       STOP_DATA; representation factorial and audit remain blocked
```

Run development validation with the existing environment:

```powershell
.venv/hsg/python.exe -m tools.relational_emergence.v2.run --out outputs/analysis/relational_emergence/v2/new-development-run
.venv/hsg/python.exe -m unittest tests.analysis.test_relational_emergence_v2_protocol tests.analysis.test_relational_emergence_v2 -v
```

The output directory must be empty. Artifacts record code, protocol, audit
configuration and dataset digests. Nonzero exit is expected when prerequisites
fail; the gate JSON explains why. No command opens the new audit allocation.
Using fewer groups, permutations, epochs or calibration replicates is allowed
for debugging, but cannot turn inadequate power into a PASS.

`P-R`, `P-G`, `Shuffle-R`, `InitialStatic-R` and `PostState-R` share a residual
autoregressive prediction objective. Model helpers are available for regression
tests and development; their raw validation losses are not Phase IB endpoints.
There is deliberately no confirmatory factorial command while the gates fail.
`require_phase_ib` rejects missing, old or stale gate inputs, missing real
transplant controls and a horizon exceeding either the inference or prediction
window. The global and relational temporal models match active parameter count
within 1%, using the same depth and optimizer. All checkpoint selection uses
prediction validation loss alone.

The model batch never contains mechanism labels, true future states, exposure,
family IDs or counterfactual pairing metadata. It contains the current state and
future **actions**, which are exogenous. Temporal codes receive positions and
past incoming actions; static codes receive no velocities or history. Future
targets are separate, and missing future steps raise rather than repeat the last
state. Each shuffled episode receives a fresh permutation in training and
evaluation, unknown to the model; all objects and incoming actions move together.

The transplant scorer uses the predictor's own history code. Within-filler and
cross-filler matches keep correct/wrong sources in the same context. Directed
query edges are aligned by physical roles before transplanting; global models
receive a complete scene code. The target cutoff state, structural attributes and
future actions remain fixed. Error is RMS normalized by a caller-supplied scale
frozen on training contexts. Zero sensitivity is reported and retained in
Delta-CF. No prediction-based off-manifold filter is applied. Exposure-conditioned
results are additional to the unconditional endpoint. Known-answer fixtures test
both positive and alias codes through the full scorer; these do not replace
calibrating a learned decoder on the actual world.

Remaining work follows #107's gates: settle the world horizon and chronology
control, reach M0 calibration power, run the multi-object Level-2 oracle and
real-data transplant controls, then evaluate all seeds and finally freeze the
fresh audit. No best-seed or representation-metric selection is implemented.
