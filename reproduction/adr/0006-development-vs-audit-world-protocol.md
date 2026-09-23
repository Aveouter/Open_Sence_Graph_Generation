# ADR 0006: Development and Audit World Protocol

## Status

Accepted

## Context

Phase I requires that final claims come from a sealed audit set, and forbids
selecting a checkpoint, architecture or hyperparameter by any relation-facing
metric. At the same time the first delivery of Phase I is the simulator
identifiability gate only: it trains no representation model and produces no
headline number. So it is not obvious whether an audit world has to exist yet,
and building one early risks freezing it against requirements that Phase IB has
not yet fixed.

Repository precedent is relevant here. `tools/ontology_probe/` keeps generated
artifacts out of version control and keeps hand-written reports in, and records
its judgement thresholds inside the artifacts so a later reader can re-check the
rule rather than trust the verdict.

## Decision

**Split the world axis now, instantiate the audit world later.**

- The simulator is parameterised by a world identifier from its first commit.
  `world` is a field of the frozen config and is stamped into every artifact.
- `development_world` permits debugging, visualisation, metric development,
  architecture exploration, `H` selection to the extent it is a gate parameter,
  and control-set calibration.
- `audit_world` uses disjoint filler-family pools, disjoint nuisance ranges, and
  disjoint rendering and action seed pools. It is instantiated only when a
  headline claim is to be produced, and each audit run is preceded by a protocol
  freeze recorded in a new ADR.
- **In the first delivery the audit world is defined but not instantiated.** The
  components that must exist now are the world parameter, the disjoint seed and
  family pools, and an assertion that the two worlds are disjoint. Split
  *construction* is likewise deferred.
- Nothing discovered on development data may be back-dated into the audit
  protocol. If the audit protocol changes after a run, the run is reported as
  performed and superseded, never quietly repeated.
- Any metric used to choose anything must be a development-world metric. The
  audit world may only produce numbers that are reported, not numbers that are
  selected against.

This is recorded as a **deferred** decision, not as work done. The Phase IA
protocol validator must assert world disjointness and must fail closed when the
audit world is referenced before its freeze.

## Evidence

- Plan/source: Phase I plan text supplied 2026-09-23 (not tracked in this
  repository).
- Local precedent for the develop-then-freeze boundary and for reporting
  discipline: `tools/ontology_probe/README.md` (single-seed and
  reporting-discipline caveats), `tools/reproduction/evidence_gates.md`.
- Local precedent for keeping generated artifacts untracked and reports tracked:
  `.gitignore`, `outputs/reports/ontology_probe/README.md`.

## Consequences

- **Can be claimed:** that the development and audit worlds are separated by
  construction, and that the separation is enforced by an assertion rather than
  by convention.
- **Cannot be claimed:** that any Phase IA gate result is a headline result. Gate
  outputs are instrument checks on the data-generating process, not findings
  about representation learning.
- **Done since this was written:** the `world` field and the disjoint-seed-pool
  assertion exist in `Phase1AConfig`, and
  `tools.relational_emergence.audits.protocol_validator` enforces the disjointness
  along with the artifact-stamp contract and the fail-closed rule for a premature
  `audit_world` reference. All four validator checks were verified to fire on a
  deliberately tampered artifact set, not merely to pass on a clean one.
- **Still deferred:** instantiating the audit world, and the dataset splits. Both
  wait on a headline claim to seal.

### The splits, and a correction

All four split types are implemented in `simulator/splits.py`: IID,
unseen-filler (hold out filler identities), unseen-filler-family (hold out
appearance families, which sit on the nuisance side by ADR 0005), and
pair-recombination (hold out identity *pairs* while keeping every identity in
training). The split unit is the **base context**, never the episode row — a
context's twins and its three regimes are near-duplicates, so a row-level split
would put a context on both sides and report generalization where it had measured
memorization. Disjointness on the context id and on the held-out axis is asserted
rather than warned, following `ood_split.audit_split` in `tools/ontology_probe`.

**This section previously recorded pair-recombination as blocked**, on the
grounds that one `Filler` supplies the structural attributes of both objects and
so there is no identity pair to hold out. That was wrong, and the correction is
worth keeping visible: `fillers._structural` draws the probe's attributes and the
supporter's attributes from *independent* uniform distributions, so the pair is
already a pair. What was missing was the split, not the data — and the cost of
the mistake would have been a data-generating change that invalidated every
frozen artifact to solve a problem that did not exist. Identity classes are
buckets over a structural parameter with edges fixed by the declared parameter
range rather than by the sample, so a class means the same thing across runs.

## Amendment: the audit world is instantiated

**Recorded 2026-09-23, Phase IB.** The deferral above was conditioned on a
headline claim being ready to seal, and that condition is now met: Phase IB has
been run on `development_world` and produces the endpoints the claim ladder reads.

- The world parameter is wired through `run_phase1a.build_groups`, which now
  draws its filler pool seed and its appearance families from
  `config.world_seed` and `config.world_families`. The development values are
  unchanged, so every Phase IA artifact regenerates identically.
- `audit_world` uses `AUDIT_FAMILIES` and generator seed `20261107`, disjoint from
  the development families and seed by construction.
- The generator is the identical code path. Only the sample changes, which is the
  whole content of the seal: a result that fails under audit is not explainable as
  a changed simulator.
- The protocol freeze is **ADR 0009**, written before the audit run as this ADR
  requires. It names the architecture, the objective, the hyperparameters, the
  checkpoint rule, the split and the endpoint definitions, and it forbids any of
  them changing between the freeze and the run.

The claim ladder's discipline carries over unchanged: development-world numbers
are reported as development-world measurements, not as the phase's result, and the
sealed run is reported alongside them even where the two disagree — a disagreement
is a finding about the stability of the measurement, and reporting only the
sealed number would turn the seal into a selection device.
