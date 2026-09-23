# Phase IA — ontology, M0 leakage, and the Level-1 identifiability gate

**Status:** `research_experiment` · **Scope:** `synthetic_relational_emergence_phase1a`
· **World:** `development_world` · **Not a reproduction.**

Run: `python -m tools.relational_emergence.experiments.run_phase1a` (288 base
contexts, horizon 60, 58 s wall clock). Artifacts and their SHA-256 digests are
listed in `outputs/analysis/relational_emergence/phase1a/manifest.json`; every
number below is read from those files rather than recomputed here. The
`config_sha256` stamped into every artifact covers the world constants as well as
the config dataclass, so a change to gravity, damping, the arena extents or the
contact softness invalidates the stamp rather than passing unnoticed.

---

## 1. Measured

### 1.1 Ontology and compatibility (M0, structural)

Six realizable joint `S_0` tuples, enumerated from the frozen factor domains
rather than listed. Compatible-mechanism counts per tuple are **4 / 4 / 4 / 3 /
3 / 3**: every tuple admits at least three mechanisms including `free`, every
active mechanism spans at least two tuples, and the tuple–mechanism bipartite
graph is connected. Zero constraint violations.

Because the count varies by tuple, the compatibility-aware chance level varies
too — a quarter for the four-mechanism tuples, a third for the three-mechanism
ones. No flat 25% or 33% figure is used anywhere.

**Roles come from the law, not from the argument order.** Each episode carries a
role assignment (`containment` names its objects `inner`/`container`, `support`
names them `supported`/`supporter`, `attachment` `endpoint_1`/`endpoint_2`) and
both directions of the relation, generated together from one episode id so they
cannot drift apart. `attachment` is marked symmetric, which is what gives Phase
IB's role-equivariance check a ground truth to score against rather than a claim
to make about an encoder. Both directions name the relation after the *law*:
containment seen from the container is still containment, and naming the reverse
direction `R^-1` would double the label space for no information.

### 1.2 M0 leakage probes

1056 records (288 contexts × compatible mechanisms), 528 held out for scoring.
Ridge-regularised linear probe, group-blocked split-half, 40-permutation test
that re-draws mechanism labels *within each tuple*.

Two estimators are run per feature set and a leak is flagged if *either* detects
it: a ridge readout on the raw features, and the same readout on a frozen random
`tanh` hidden layer of 64 units. The second is what ADR 0007 asks for — a
detector of the same *family* as the oracle — and the reason is that the failure
modes are not symmetric: a probe too weak to see a nonlinear leak reports "no
leak", and that false negative is indistinguishable in a table from a clean
result.

| probe | linear acc (p) | random-feature acc (p) | compatibility-aware baseline | MDE |
|---|---|---|---|---|
| `s0_only` | 0.250 (0.024) | 0.250 (0.024) | 0.273 | 0.048 |
| `x0_only` | 0.241 (0.951) | 0.142 (1.000) | 0.273 | 0.048 |
| `filler_only` | 0.244 (0.976) | 0.182 (1.000) | 0.273 | 0.048 |
| `x0_plus_filler` | 0.229 (1.000) | 0.144 (1.000) | 0.273 | 0.048 |

Every cell sits **below** the compatibility-aware baseline, under both
estimators, and none is flagged. Effects smaller than 4.8 accuracy points would
not have been detected at this sample size and are reported as **undetectable
rather than absent** — the sample size is set so this minimum detectable effect
meets the 0.05 declared in the frozen config before the run, and the driver
prints a warning if a run comes in underpowered.

The random-feature probe scores *lower* than the linear one on three of four
feature sets, which is worth stating rather than hiding: 64 random features and
2541 training rows is enough capacity to overfit, so at this sample size the more
expressive estimator is the noisier one. It is retained because the direction of
its error is the safe one — it can only make the audit stricter — and because the
must-fire control below is what establishes that the suite as a whole is
sensitive.

**Must-fire control.** The same probes on a deliberately leaky sampler — initial
state shifted by a mechanism-dependent offset — reach **0.591** (linear) and
**0.540** (random-feature) against the same 0.273 baseline, both at p = 0.024.
The control fires if *either* estimator catches the leak; requiring a specific one
would fail the gate when an expressive probe overfits a small sample, which is an
artefact of the probe rather than evidence about the sampler. It **fired**, so
the clean result on the canonical sampler is informative rather than vacuous.

One row needs care in reading. `s0_only` has a permutation p of 0.024 while
sitting *below* the analytic baseline. Both are correct: the permutation null is
built by shuffling labels within tuples and refitting, so its centre sits below
the analytic constant-predictor value. A probe can therefore be distinguishable
from the permutation null and still be worse than guessing a compatible
mechanism. The leak flag requires both conditions, which is why this row is not
flagged.


### 1.3 The two floors

| floor | mean | 95th pct | max | n |
|---|---|---|---|---|
| numerical (same seed, re-grouped arithmetic) | 5.89e-16 | 2.00e-15 | 7.17e-12 | 190 080 |
| null (same mechanism, re-drawn noise) | 0.1329 | 0.3653 | — | 30 cells |

The numerical floor is at machine precision in every statistic, so the `J_ab`
values below are distances between laws and nothing else. Getting there required
removing **four** separate discontinuities, and the list is worth recording
because each one was invisible until measured:

1. Hard constraints zeroed a whole velocity the instant a threshold was crossed —
   a step function of position, so a 1e-17 difference flipped a branch and two
   runs diverged macroscopically. The floor read **14.8**, larger than most of
   the signal it was meant to bound. Compliant responses fixed it.
2. The arena relaxed positions back by a *fraction* of the violation, so
   sustained impulses outran the correction: 26.9% of rich-regime body-steps sat
   outside a wall, and the floor's tail reached 21.6.
3. `containment` measured its own velocity response with inverted vertical
   normals, so it could not hold anything up at all — a body sank 0.15 through
   its own floor. Fixing the sign restored the constraint *and* reintroduced a
   switch, because the position pass cleared the entire velocity component the
   moment a limit was crossed by any amount.
4. `support` chose *the* contact rectangle by deepest overlap and *the* axis by
   larger gap. Both are argmaxes, and the second was full-strength near a corner.
   Applying every rectangle's response, with the normal taken as the gradient of
   the distance function rather than a choice between two axes, removed the last
   branch.

The mean floor is the number to watch: a hard constraint shows up there long
before it shows up in any `J_ab`.

**Precondition: passed.** The null floor's mean (0.1329) is far below the largest
between-mechanism separation, so the deterministic/shared-noise premise holds.

### 1.4 `J_ab` by mechanism pair and intervention regime

`J_ab^AUC` = horizon mean of the shared-noise matched-counterfactual trajectory
distance, normalised per state dimension by a frozen robust scale. 30 cells; the
full matrix is in `j_ab.json`. Representative rows:

| tuple | pair | passive | weak | rich |
|---|---|---|---|---|
| inside\|touching\|other | containment/support | **0.020** | 2.258 | **4.069** |
| inside\|touching\|other | attachment/support | **0.052** | 2.434 | 4.526 |
| inside\|touching\|other | attachment/containment | **0.035** | 0.462 | 1.505 |
| inside\|touching\|other | containment/free | 0.341 | 2.684 | 4.414 |
| inside\|touching\|other | attachment/free | 0.344 | 2.961 | 4.936 |
| inside\|touching\|other | free/support | 0.339 | 1.047 | 1.036 |
| inside\|non_touching\|other | containment/support | **0.008** | 2.202 | 4.079 |
| inside\|non_touching\|other | containment/free | 0.323 | 2.600 | 4.422 |
| outside\|touching\|on_top | attachment/support | 0.366 | 2.911 | 4.736 |
| outside\|touching\|on_top | attachment/free | 1.899 | 3.708 | 5.409 |
| outside\|non_touching\|other | free/support | **0.000** | 1.492 | 1.342 |
| outside\|touching\|other | free/support | 0.019 | 2.050 | 1.866 |

| regime | median `J_ab^AUC` |
|---|---|
| passive | 0.498 |
| weak | 2.494 |
| rich | 4.380 |

**28 of 30 cells** show a positive passive-to-rich increment, and the weak regime
sits between passive and rich in **25 of 30** — both exceptions being
`free/support`, which is the one pair whose weak-regime separation is transient
(see §2).

### 1.5 Horizon

Per-pair saturation horizons run **min 0, 90th percentile 7, max 11**. The
*mean* curve saturates at step 0, because most pairs contain a law that diverges
on the first step under gravity; the horizon is therefore taken from the
per-pair distribution instead. Selected **H = 10**, the pre-registered minimum,
which here sits just above the measured 90th percentile of 7. The
slowest-separating pairs are the ones the design expects to need excitation.

### 1.6 Gate verdict

```
verdict: GO_PHASE_IA
reasons: none
inseparable important pairs: none
```

All three conditions pass together. Level-1: the leakage probes are clean under
both estimators with a control that fires, both floors are at machine precision,
and no important mechanism pair fails to separate. Level-2: with the rollout
oracle, `delta_M = +0.1655`, CI [+0.1357, +0.1954], all five seeds positive,
MDE 0.0389 against the 0.05 pre-registered target, and the shuffled control does
not beat base — see report 02.

The gate did **not** pass on the first attempt, and the path matters. The
direct-head oracle failed **both** Level-2 conditions: `delta_M = +0.0014` with
an interval containing zero, and `shuffled_vs_base` positive in all five seeds —
a wrong-but-compatible label reliably beating no label, which is the control leak
the third condition exists to catch. That parameterisation could not learn the
dynamics well enough on unseen contexts for the mechanism to show, so what signal
it had came from the label revealing the group rather than the law. Report 02 §1
takes this apart.

The verdict changed when the oracle was fixed, not when the criterion was
relaxed: the criterion, the arms, the split and the checkpoint rule are unchanged
across both runs.

### 1.7 Dataset splits

All four of the plan's split kinds are implemented over base contexts and
emitted to `splits.json`; the unit is the **context**, never the episode row,
because a context's twins and its three regimes are near-duplicates and a
row-level split would put a context on both sides.

| kind | train | test | held out |
|---|---|---|---|
| `iid` | 201 | 87 | — |
| `unseen_filler` | 201 | 87 | 12 of the 40 filler identities |
| `unseen_family` | 144 | 144 | `dev_brushed_metal`, `dev_matte_polymer` |
| `pair_recombination` | 213 | 75 | identity pairs `i0/j2`, `i1/j1` |

Disjointness is asserted on both the context id and the held-out axis, never
warned. `unseen_family` is the headline axis: ADR 0005 puts appearance families
strictly on the nuisance side, so generalizing across one is invariance to
appearance rather than extrapolation of physics.

**A correction, kept visible.** This section previously recorded
pair-recombination as *not implemented and impossible without changing the
data*, on the grounds that a base context draws a single `Filler` supplying the
structural attributes of both objects, so there is no independently drawn
identity pair to hold out. **That was wrong.** `fillers._structural` draws the
probe's attributes and the supporter's attributes from *independent* uniform
distributions, so the pair already exists; what was missing was the split, not
the data. Acting on the wrong version would have meant a data-generating change
that invalidated every artifact in this directory, to solve a problem that did
not exist. ADR 0006 carries the same correction, and
`simulator/splits.py::_assert_axis_held_out` now asserts both halves of the
claim — that the held-out pair is novel *and* that neither of its identities
disappears from training — rather than trusting the construction.

### 1.8 Reproducibility

The whole Level-1 pipeline was re-run into a clean directory and every artifact
came out **byte-identical** — `config.json`, `floors.json`, `j_ab.json`,
`horizon.json` and `leakage.json` all compare equal after excluding the two
fields that are expected to differ (`git_sha` and the wall-clock time). The
stdlib layer is deterministic given the frozen config and the seeds, which is
what makes a report's numbers checkable by anyone who runs the command.

### 1.9 Protocol validator

`python -m tools.relational_emergence.audits.protocol_validator` checks the
conditions under which the statistics above mean anything, and fails closed:

- development and audit seed pools are disjoint, re-derived rather than trusted
  from the config's own assertion;
- every artifact carries `status`, `not_a_reproduction`, `scope`, `world`,
  `schema_version`, `config_sha256` and `git_sha` — a missing stamp is an error,
  not a warning;
- no artifact may claim the `audit_world` before a freeze record exists;
- `gate.json` and `level2.json` must agree on `config_sha256`, and `level2.json`
  must name the `oracle_dataset.jsonl` actually on disk.

The last two are the ones that catch a real failure mode rather than a typo: the
oracle is required to consume the frozen dataset rather than rebuild it, so if the
dataset is regenerated the measurement taken on the old one is silently orphaned —
every number still looks valid and describes data that no longer exists. All four
checks were verified to fire on a deliberately tampered copy, not merely to pass
on a clean one.

---

## 2. Interpretation

**Q1 — is the mechanism↔filler-nuisance shortcut eliminated?** Yes, to the
resolution this sample can support. No probe reads the mechanism from the joint
tuple, the continuous initial state, the nuisance attributes, or both together,
and the detector is demonstrably able to fire on a leak of this family. This
matters because a sampler that placed objects differently depending on the law
they were about to obey would make every downstream measurement circular.

**Q2 — does the mechanism still affect the future once `X_t`, `F` and `A` are
fixed?** Yes, and the *smallest* passive separations are at the `inside` tuples:
an object held in a cavity and one that falls straight through it differ by
0.008–0.35 until something moves.

**Q3 — does richer intervention raise identifiability?** Yes. The median cell
goes 0.498 → 2.494 → 4.380, and 28 of 30 cells increase. The *witness* exists in
falsifiable form: **thirteen of the thirty cells** are passively inseparable and
actively separable — passive `J_ab` below the null floor's 0.365 while rich
`J_ab` is above it. Under the pre-registered criterion in ADR 0007 the
intervention claim is therefore **resolved**, not unresolved.

The five with the largest ratios (the table is a selection, not the full
thirteen — all thirty rows are in `j_ab.json`):
|---|---|---|---|---|
| outside\|non_touching\|other | free/support | **0.000** | 1.342 | — |
| inside\|non_touching\|other | containment/support | **0.008** | 4.079 | 515× |
| inside\|touching\|other | containment/support | **0.020** | 4.069 | 203× |
| inside\|touching\|other | attachment/support | **0.052** | 4.526 | 86× |
| inside\|touching\|other | attachment/containment | **0.035** | 1.505 | 44× |

The first row is the cleanest case of all: under passive observation `support`
and `free` are *bit-identical* — a body resting beside the box with no excitation
never discovers whether the box would resist it — and rich excitation separates
them by 1.342, well above the floor.

The load-bearing case is `containment` versus `support` at the `inside` tuples:
**0.008 and 0.020 passively**, against 4.08 and 4.07 under rich. Two laws that
are indistinguishable from observation alone become sharply distinguishable once
the pair is probed, by a factor of several hundred. That is the phase's central
hypothesis stated in one number, and it is only visible because the mechanism is
defined as a *disposition* — something the configuration does not reveal and only
the response does. The mechanism is not a property of the configuration, so it
cannot be read off a configuration.

The strength of this result is a product of the continuity fixes in §1.3. Before
them the same cell measured 0.90 passively against 4.86 under rich — the right
direction, but a factor of five rather than two hundred, because the
discontinuous contacts were injecting noise that partly destroyed the equality
the design is built on. Correctness and continuity turned out to be the same
work, which is worth knowing for any future simulator in this repository.

One pair is a systematic exception, and it is worth stating precisely because it
is *not* simply "a pair that fails to separate". **Every ordering violation in
the matrix belongs to `free/support`, and the pair behaves in opposite ways in
different cells.**

| cell | passive | weak | rich | |
|---|---|---|---|---|
| outside\|non_touching\|other | **0.000** | 1.492 | 1.342 | a witness — passively identical |
| outside\|touching\|other | 0.019 | 2.050 | 1.866 | a witness |
| inside\|touching\|other | 0.339 | 1.047 | 1.036 | flat |
| outside\|touching\|on_top | 1.720 | 1.572 | 1.621 | **already separable passively, and richness reduces it** |
| outside\|non_touching\|on_top | 1.731 | 1.583 | 1.522 | the same, monotone downward |

The last two are the only cells in the matrix where rich separation falls below
passive, and they are the two that break the passive-to-rich ordering. The cause
is the same in each: `support` and `free` differ by whether a surface resists the
body, and once the body has come to rest that difference is behind them, so a
longer horizon averages the separation down rather than up. The difference is
real but transient, and a horizon-mean metric partly cancels it.

No ontology merge is indicated, and this pair is the reason the merge rule is
stated over *all* cells rather than per cell: ADR 0007 would require two laws
that fail to separate *everywhere*, and this pair separates strongly at four
cells and perfectly anchors two witnesses. A per-pair divergence-time metric
would report it more faithfully than the horizon mean, and is the obvious
refinement.

---

## 3. Not claimed

- **Any representation-learning result.** Phase IA measures whether the world is
  identifiable, not whether a model learns anything from it. No `P-R` or `P-G`
  model has been built, and `CCGP`, `RoleEquivariance` and `CFError` remain
  pre-registered endpoints with **no measured value**.
- **Any claim about natural video, object perception, or relations outside the
  four named laws.** Observation is simulator state plus a nuisance vector; there
  is no renderer in Phase I.
- **Any distributional mechanism.** Phase IA is restricted to deterministic,
  shared-noise mechanisms. A law differing from another only in the
  *distribution* of its response is out of scope and is covered by no number
  here.
- **Any claim about the shape of the passive-separability distribution.** The
  floors are reported as a mean and a quantile, not as a confidence interval over
  base contexts; no per-cell bootstrap is reported in this run.
- **That this is a reproduction of anything.** See `AGENTS.md`.
