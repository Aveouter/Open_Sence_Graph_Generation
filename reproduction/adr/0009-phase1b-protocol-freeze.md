# ADR 0009: Phase IB Protocol Freeze for the Audit World

## Status

Accepted

## Context

ADR 0006 split the world axis and deferred the audit world until a headline claim
was to be produced, with the rule that "each audit run is preceded by a protocol
freeze recorded in a new ADR" and that nothing discovered on development data may
be back-dated into the audit protocol. Phase IB has now been run on
`development_world`, so that condition is met: either a headline is produced under
the seal, or the phase reports development-world numbers only.

This ADR records the freeze. Everything below is fixed as of the commit that adds
it and must not change between now and the audit run. If something does have to
change, the audit run is reported as performed and superseded, never quietly
repeated.

## Decision

### The world

- `world = audit_world`, generator seed `20261107` (`config.WORLD_GENERATOR_SEEDS`).
- Filler pool drawn from `fillers.AUDIT_FAMILIES`, disjoint from the four
  development families by construction and asserted so by
  `Phase1AConfig.__post_init__`'s seed-pool check and by `world_families`.
- The generator is the identical code path. Only the sample changes, which is the
  whole content of the seal: a result that fails under the audit is not
  explainable as a changed simulator.

### The data

- `Phase1AConfig` defaults otherwise unchanged: 48 base contexts per `S_0` tuple,
  horizon 60, filler pool 40.
- Split: `unseen_family`, `DEFAULT_SPLIT_SEED = 0` — the headline axis, held out
  by appearance family per ADR 0005.
- Context partition: `design.partition_contexts(..., seed=DEFAULT_SPLIT_SEED + 1)`.
- Condition partition: `design.family_condition_partition(..., seed=0)`, which for
  this split coincides with the split's own family holdout.

### The arms

All seven, run once, in one invocation:

| arm | temporal | pairwise | labels |
|---|---|---|---|
| `S-G` | no | no | yes |
| `S-R` | no | yes | yes |
| `StaticSSL-G` | no | no | no |
| `StaticSSL-R` | no | yes | no |
| `Shuffle-R` | destroyed | yes | no |
| `P-G` | yes | no | no |
| `P-R` | yes | yes | no |

### The training protocol

- `ModelConfig(latent=32, hidden=64, depth=2, history=4, epochs=40,
  batch_size=128, learning_rate=1e-3, seed=0, rollout_steps=4)`.
- Capacity matched across arms by `_matched_global_hidden`; the parameter counts
  are recorded in the artifact and must agree within 1% between `P-R` and `P-G`.
- Checkpoint: minimum validation loss on the rolled objective, over contexts
  disjoint from the fit set. No relation-facing metric selects anything — no
  probe, no CCGP, no role score, no transplant score. ADR 0007 forbids it and
  `train_arm` reads no relation label except the supervised arms' auxiliary head,
  which is their definition.
- No hyperparameter sweep has been performed on either world, and none will be
  performed on the audit world. The development run used these exact settings.

### The endpoints

- **Probe** — split-side conditions, reported as a diagnostic and never as primary
  evidence.
- **CCGP** — condition = (appearance family, regime), readout fit on the training
  families and scored on the held-out ones, compatibility-aware baseline, test set
  standardised by the training set's statistics (ADR 0008 §6). This is the
  headline.
- **Role equivariance** — one orthogonal `T_swap` fitted on development families,
  scored on unseen ones, with the deranged-pairing control and `||T^2 - I||`.
- **Transplant** — `CFError` over the arm-averaged wrong-law control, plus the
  paired alignment against its within-cell derangement, per regime. Reported with
  `--code oracle` alongside as the must-fire control (ADR 0008 §5).
- **Label efficiency** — CCGP at 1, 5, 10, 50 and all labels per class, with the
  test side held fixed across budgets.

### The claim

The audit run's numbers decide between the claim-ladder rows in ADR 0007. The
development-world numbers are **not** evidence for the headline and are reported
as development-world measurements; only the audit run's can be cited as the
phase's result.

## Amendment: one superseded run

**Recorded after the fact, as this ADR requires.** A first `audit_world` run was
started and then abandoned, because a correction landed to the transplant
endpoint's *control* while it was in flight (ADR 0008 §5: the alignment statistic's
within-cell derangement could not differ from its subject).

The corrected run is the one reported. The superseded one produced no artifact —
it was killed before writing — so nothing has to be withdrawn, but the sequence is
recorded here rather than omitted, because "the protocol changed mid-run" is the
case this ADR exists to make visible. The `CFError` contrast, which is the
endpoint's evidence, was unaffected by the correction; only the alignment
statistic's null was.

## Consequences

The audit world answers a strictly harder question than the development world in
one respect and an identical one in every other: the appearance families are new,
so an endpoint that survives it is one that did not transfer appearance statistics
rather than physics. Nothing in the architecture, the objective or the
hyperparameters has been chosen against either world's endpoint values.

If the audit run disagrees with the development run, both are reported. A
disagreement is a finding about the stability of the measurement, and suppressing
the development number in favour of the sealed one would turn the seal into a
selection device — which is the opposite of what ADR 0006 built it for.

## Evidence

- ADR 0006 (world split and the freeze rule this discharges), ADR 0007 (claim
  ladder and gate criteria), ADR 0008 (endpoint specification).
- `tools/relational_emergence/config.py`, `.../eval/design.py`,
  `.../models/representation.py`, `.../experiments/run_phase1b.py`.
- Development-world artifacts: `outputs/analysis/relational_emergence/phase1b/`.
