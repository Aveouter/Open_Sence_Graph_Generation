# `tools/relational_emergence` — Phase I: Predictive Relational Emergence

**Status: research instrument, not a reproduction.** Nothing here loads an SGG
checkpoint or reports a benchmark number. Every artifact carries
`status: research_experiment` and `not_a_reproduction: true`.

The question this tree exists to answer is narrow and stated in advance:

> Under what conditions does predictive interaction learning induce abstract
> relational representations?

Phase I answers it in a synthetic world with a hand-written simulator, so that
the world's properties can be *measured* rather than assumed. The decisions
fixing the data-generating process, the development/audit separation, the
evidence criteria and the Phase IB endpoint specification are in
[`reproduction/adr/0005`](../../reproduction/adr/0005-phase1-data-generating-process.md),
[`0006`](../../reproduction/adr/0006-development-vs-audit-world-protocol.md),
[`0007`](../../reproduction/adr/0007-emergence-evidence-criteria.md),
[`0008`](../../reproduction/adr/0008-phase1b-endpoint-specification.md) and
[`0009`](../../reproduction/adr/0009-phase1b-protocol-freeze.md). Those files are
authoritative; this README is a map.

Phase IA is the world and its identifiability gate. Phase IB is the factorial of
representation arms and the three endpoints the phase is judged on. They are
separate runs against the same frozen episode file, and neither re-samples what
the other measured.

## Layout

```
simulator/     the canonical, pure-stdlib world
    geometry.py      gravity frame and the three pure labelling predicates
    factors.py       the factorized S0 (containment x contact x pose)
    compatibility.py C(S0_tuple, M) and its three admissibility constraints
    fillers.py       structural vs nuisance attributes, appearance families
    actions.py       impulse space and the passive/weak/rich policies
    mechanisms.py    the four dispositional laws
    state.py         bodies and world constants
    world.py         the integration step and the rollout
    labeling.py      S0 derived from a configuration, never attached to it
    sampling.py      X0 ~ P(X0 | S0)
    roles.py         role assignment and the bidirectional ordered-pair view
    counterfactuals.py  base contexts and twin construction
    splits.py        the four split kinds, over base contexts
    dataset.py       the frozen episode rows both phases read
audits/        every audit here is pure stdlib and CI-runnable
    ontology_audit.py     the compatibility table and its constraints
    leakage.py            the four M0 probes, the baseline, the must-fire control
    identifiability.py    the two floors and the J_ab curves
    protocol.py           horizon selection, separability, the gate verdict
    protocol_validator.py the invariants that make the statistics meaningful
eval/          the Phase IB endpoints, also pure stdlib and CI-runnable
    linalg.py        Jacobi eigensolver, SVD, orthogonal Procrustes
    endpoints.py     CCGP and role equivariance, each with its control
    transplant.py    the score scale, the error, the alignment, the bootstraps
    design.py        context / condition / validation partitions, transplant plan
models/        the ONLY place allowed to import torch
    oracles.py            Base / TrueM / ShuffledM, capacity-identical
    batch_stream.py       frozen rows -> tensors, and the multi-step futures
    representation.py     the seven arms, capacity-matched, multi-step objective
    transplant.py         the shared decoder the transplant endpoint reads through
experiments/
    run_phase1a.py   data generation, M0 audits, the Level-1 gate
    run_level2.py    the oracle utility check, folded in via --level2
    run_phase1b.py   the factorial, the three endpoints, and the oracle control
    summarise_phase1b.py  the report tables, straight from the artifacts
    diagnose_*.py    the evidence behind how the transplant endpoint is specified
```

`eval/` and `audits/` are pure stdlib on purpose. The endpoints are the numbers
the phase reports, and a measurement that only runs where torch is installed is a
measurement that rarely gets re-checked.

## The two conventions that matter

**One canonical simulator, pure stdlib.** The CI `validate` job runs
`unittest discover` on a bare Python 3.10 with no third-party packages, and a
test that imports torch skips itself there. Anything in `simulator/` or `audits/`
that reached for numpy would therefore stop being tested in CI — silently, and
exactly where it matters most. Torch is confined to `models/`, which consumes
frozen artifacts and never re-samples twins, rebuilds splits or redefines
compatibility.

**Every response is continuous in the state.** A discrete branch in the dynamics
— a hard constraint, an argmax, a threshold — makes a `1e-17` difference flip
and two runs diverge macroscopically, at which point `J_ab` measures noise rather
than laws. That is not hypothetical. It happened **four times** here, each
invisible until measured: hard constraints that zeroed a velocity at a threshold
(floor 14.8), arena bounds that relaxed by a fraction of the violation (26.9% of
rich body-steps outside a wall), inverted vertical normals in `containment` that
let bodies sink 0.15 through their own floor, and argmaxes over contact
rectangles and axes in `support`. Only after all four did the floor reach machine
precision — and the load-bearing cell went from 0.90 passive / 4.86 rich to
**0.020 passive / 4.07 rich**. Correctness and continuity were the same work.

**Re-measure `numerical_floor` after any change to the integration or the
constraints, and watch the mean rather than the max — a reintroduced switch shows
up there first. A mean above about `1e-9` means one came back.**

## Running it

```bash
.venv/hsg/python.exe -m tools.relational_emergence.audits.ontology_audit
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_phase1a
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_level2
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_phase1a \
    --level2 outputs/analysis/relational_emergence/phase1a/level2.json
.venv/hsg/python.exe -m tools.relational_emergence.audits.protocol_validator

# Phase IB, development world: the factorial and the three endpoints
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_phase1b
# the must-fire control: the same machinery fed a code that *is* the mechanism
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_phase1b \
    --code oracle --out outputs/analysis/relational_emergence/phase1b_oracle
# the sealed run, under the protocol frozen in ADR 0009
.venv/hsg/python.exe -m tools.relational_emergence.experiments.run_phase1b \
    --world audit_world --out outputs/analysis/relational_emergence/phase1b_audit
.venv/hsg/python.exe -m tools.relational_emergence.experiments.summarise_phase1b
```

**Order matters for the two-phase gate.** `run_phase1a` writes the episode file
and `run_level2` measures the oracles against it; the fold-in refuses a Level-2
artifact whose recorded dataset digest does not match the one this run wrote. So
any change that regenerates the episodes means re-running `run_level2` before the
verdict can be reached — the failure mode that guard closes is a gate reporting a
number measured on a dataset that no longer exists.

Artifacts land in `outputs/analysis/relational_emergence/` (gitignored). Reports
are hand-written and tracked, in
[`outputs/reports/relational_emergence/`](../../outputs/reports/relational_emergence/README.md).

On Windows, prefix with `PYTHONIOENCODING=utf-8` for scripts that predate the
encoding fix in `tools/ci_validate.py`.

## What Phase IA is not

Phase IA is the simulator and the identifiability gate, and nothing else. It
trains no representation model, and the three endpoints have no value in its
artifacts. The audit world is instantiated by Phase IB under the freeze in
ADR 0009, not here.

## What Phase IB is not

It does not load an SGG checkpoint, it does not touch a benchmark, and it does
not report a real-data number. Its world is the same hand-written simulator, so
a result here is a statement about a fully specified data-generating process and
nothing wider. The three endpoints are the only ones that may carry a claim;
relation probes, clustering and geometry diagnostics are recorded beside them and
are never primary evidence.

Two limits are worth knowing before reading any of its numbers:

**The transplant endpoint's power is bounded by the world, not by the code.**
Measured (`diagnose_horizon`), a decoder given the true mechanism as a one-hot
label beats one given a deranged label by a mixed-sign gap of about 10% across
horizons from 1 to 20. Where the label itself buys little, no learned code can be
read, and the endpoint's report says so rather than leaving a null to be
over-interpreted.

**Zeroing an input is not an ablation.** The `code_ablation` block records the
arm's loss with the bottleneck zeroed, which rises several-fold. That is not
evidence the predictor routes through the code: a zero vector is out of
distribution for a ReLU MLP, and most of the increase is distribution shift. The
reading it supports is the weak one — the code is not inert in the arm.
