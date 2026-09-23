# ADR 0005: Phase I Synthetic Data-Generating Process

## Status

Accepted

## Context

Phase I ("Predictive Relational Emergence") proposes a synthetic simulator to
test under which conditions predictive interaction learning induces abstract
relational representations. The governing plan supplies the research question,
the primary endpoints and the milestone order, but leaves the data-generating
process underspecified. Two of its own requirements cannot both be met by the
single-axis ontology it proposes:

1. Matched counterfactual twins must share an identical initial state `X_0`,
   identical fillers, identical actions and an identical noise seed, varying
   only the interaction mechanism.
2. A leakage probe using only the initial state must not predict the mechanism
   above a compatibility-aware baseline.

If the mechanism is a deterministic function of the configuration — for example
"containment" meaning "the object is geometrically inside the cavity" — the
second requirement fails by construction, because the mechanism is readable off
`X_0`. If instead every mechanism gets its own natural initial condition, the
first requirement fails.

Two repository constraints also apply. The CI `validate` job runs
`unittest discover` on a bare Python 3.10 with no third-party packages, so any
audit importing torch or numpy is skipped there rather than executed. And the
project standard forbids presenting synthetic or smoke output as baseline
evidence.

## Decision

The generating process is defined over two **crossed** axes.

- `S_0` — observable relational configuration, **factorized** rather than a
  single mutually-exclusive label:

  | factor | values |
  |---|---|
  | `containment` | `inside`, `outside` |
  | `contact` | `touching`, `non_touching` |
  | `pose` | `on_top`, `other` |

- `M` — latent interaction mechanism, defined as a **dispositional law**: it
  states how the pair is constrained *if* contact or excitation occurs, and it
  may be inactive at `X_0`.

The dispositional reading is what makes the crossing physical. `support` under
`(outside, non_touching, other)` means "would be held if it landed"; `free`
under the same tuple means "would pass through". Both are well-defined at the
same `X_0`, so twins exist for every legal cell.

`inside`, `on_top` and `separated` survive only as **derived readouts for
reporting**, not as `S_0` categories: `inside := containment=inside`,
`on_top := pose=on_top`, `separated := contact=non_touching`.

`pose=on_top` is a pure function of geometry and the gravity frame. It must not
read contact state, contact force, `M`, or mechanism-active state, and that
purity is enforced by a unit test.

Supporting decisions:

- **Compatibility constraints.** The set of realizable tuples `S0_realizable` is
  *enumerated from the frozen factor definitions*, never hard-coded. Compatibility
  is a predicate `C(S0_tuple, M)`, and:

  - every realizable tuple is compatible with at least three mechanisms,
    including `free`;
  - every active mechanism spans at least two distinct tuples;
  - the tuple–mechanism bipartite graph is connected;
  - the sampler is balanced strictly within each tuple's compatible mechanisms;
  - every legal cell admits a same-`X_0`/same-action twin.

  `compatible` and `mechanism_active` are **distinct predicates**. An inactive
  mechanism may still be compatible with the current `S_0` — that is the normal
  case for a dispositional law. Only `C` constrains the sampler; activation is
  dynamics, not eligibility.

  If these constraints cannot be met, the escalation is to **re-examine the
  definition of `M` or the `S_0` factorization**. Lowering the per-row mechanism
  count, or relaxing the leakage standard, is not an available response.
- **Observation.** Simulator state plus a nuisance vector. No raster renderer is
  implemented in Phase I; the packaging path must never read `M`. The plan's
  "renderer leakage" audit becomes a nuisance leakage audit.
- **Initial state.** `X_0 ~ P(X_0 | S_0)` with a zero-centred small velocity
  distribution. The intervention regime changes only the action policy, never
  the `X_0` distribution.
- **Actions.** Low-dimensional mechanism-independent impulses (target object,
  direction, magnitude, duration), with no legality gating on `M`. Actions that
  are only defined when a particular mechanism is present are excluded.
- **Roles.** Attachment is exactly symmetric in the generating process. The
  schema emits explicit bidirectional ordered-pair records sharing one episode
  id, with roles assigned by geometry and no loader-side derivation.

### Law semantics and integration decisions

Implementation forced four further decisions. They are recorded because each one
is load-bearing for the identifiability floor, and a reader re-deriving them from
the law names alone would get different answers.

- **Laws are surface laws, not axis laws.** ``support`` resists penetration
  along the **contact normal**. Every material rectangle contributes its own
  response, and in the clear the normal is the gradient of the distance function
  rather than a choice between the two frame axes. Choosing *the* contact
  rectangle by deepest overlap, or *the* axis by larger gap, is an argmax -- a
  discrete branch -- and the axis choice was full-strength near a corner. A law
  pushing along a fixed axis instead would eject objects sideways through walls
  and would make ``support`` and ``containment`` agree wherever they must differ.
  ``containment`` is compatible only with the ``inside`` tuples, where ``X_0``
  already lies within the region the law projects into, so satisfying it never
  requires dragging an object through the enclosure's own wall.
- **Contacts are compliant over ``contact_softness``, not switch-like.** This is
  not a taste preference, and the history is the evidence for it: with hard
  constraints the measured *numerical* floor was ``14.8`` in normalised units --
  larger than most of the between-mechanism signal it exists to bound -- because
  a step function of position lets a ``1e-17`` difference flip a branch. Removing
  that, the arena's fractional relaxation, an inverted pair of normals in
  ``containment``, and an unramped velocity clear in its position pass -- four
  separate discontinuities, each invisible until measured -- brought the mean
  floor to ``5.9e-16``. The physics is unchanged in the limit. **Any future change
  to the integration or the constraints must re-measure ``numerical_floor``: a
  mean above ``1e-9`` means a switch has come back.**
- **The world is bounded, and the bound is exact.** Ground, a ceiling, and two
  lateral arena walls, each a full projection rather than a relaxation: a partial
  correction left 26.9% of rich-regime body-steps outside a wall, because
  sustained impulses outrun a fractional gain. The projection is continuous in
  the state, so exactness costs nothing in the continuity the ramps provide --
  only the *velocity* response needs ramping, since that is the term that would
  otherwise be a switch. The bounding is needed because an unbounded sideways
  world plus damping means one impulse of magnitude ``v`` drifts a body roughly
  ``v / (1 - damping)`` -- about 150 units -- so every distance would measure
  which way a body happened to wander; gravity bounds an upward impulse only at
  ``v^2 / 2g``, far above the geometry the laws act on, which is why the ceiling
  is needed as well as the walls.
- **Initial velocity is zero-centred and tiny relative to the geometry**
  (``2e-3``, drifting about ``0.07`` over a horizon). At ``2e-2`` the drift is
  about ``0.7`` -- further than the cavity's lateral slack -- which separates
  ``containment`` from ``support`` under *passive* observation merely because one
  has walls and the other does not, destroying the one cell where the three main
  laws are meant to be passively indistinguishable. The perturbation is reported
  as an audited quantity so this stays checkable rather than asserted.

## Evidence

- Plan/source: Phase I plan text supplied 2026-09-23. It is **not tracked in
  this repository**; this ADR and its siblings are the tracked record.
- Local constraint files: `.github/workflows/ci.yml` (dependency-free validate
  job), `tests/_optional.py` (`require_modules` skip guard).
- Local precedent for artifact stamping and for writing decision thresholds into
  the artifact so they can be re-checked: `tools/ontology_probe/README.md`,
  `tools/ontology_probe/common.py`.

## Consequences

- **Can be claimed:** the simulator implements a stated two-axis generating
  process with pre-registered compatibility constraints, and the leakage gate is
  meaningful because it is able to fail.
- **Cannot be claimed:** anything about natural video, object perception, or
  relation types outside the named laws. Results apply only to deterministic,
  shared-noise mechanisms; distributional mechanism differences are out of scope
  for Phase IA and must be stated as such wherever these results are cited.
- **Must happen next:** freeze `configs/relational_emergence/` and the schema
  validator before generating data, and commit the protocol document so the plan
  text becomes tracked rather than conversational.
