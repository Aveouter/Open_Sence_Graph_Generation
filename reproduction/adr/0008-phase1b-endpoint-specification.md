# ADR 0008: Phase IB Endpoint Specification

## Status

Accepted

## Context

ADR 0007 fixed the claim ladder and the Phase IA gate: what may be said, and what
must be measured before it may be said. It left the *third* primary endpoint — the
counterfactual latent transplant — specified only as the plan specifies it, in
terms of a decoder:

```
X_hat^{A->B} = Decoder(X_A, z_B, A)
CFError_{A->B} = D(X_hat^{A->B}, X^B_future)
```

That is a definition of the measurement, not of the instrument. Building the
instrument produced five decisions that a reader of the resulting number could not
have reconstructed from the definition, and one of them changed whether the
endpoint produced a number at all. Several were found by measuring rather than by
reasoning, in the same way as the Level-2 amendment in ADR 0007, and for the same
reason: a null result is only interpretable if the device that produced it is
known to be able to produce a positive.

## Decision

### 1. Both the arms and the transplant decoder train on a rolled sequence

`ModelConfig.rollout_steps` and `DecoderConfig.rollout_steps`, both defaulting to
4. The plan asks for this of the arms (§18, "禁止只训练 t -> t+1"). It was applied
to the transplant decoder as well after a measurement showed it had to be.

**The measurement.** With a one-step objective, the decoder's output was
**exactly independent of the code** in 193 of 200 sampled pairs: swapping the
learned code produced a bit-identical prediction. The cause is not subtle once
seen — the next state is largely determined by the current one, so the decoder
minimised its loss without reading the code at all. The one-step residual is
fitted by the state and the structural block; the mechanism is a correction to it.

**Why this is recorded rather than fixed silently.** The measurement as first
taken said "the representation does not carry the mechanism". That claim would
have been false: it was the *objective* that gave the decoder no reason to look.
A one-step-trained instrument reports its own indifference as its subject's
irrelevance, and the two are indistinguishable in the output. This is the same
failure class as the direct-head oracle in ADR 0007, and it is recorded here so
the same mistake is not made again in the same phase.

### 2. The controlling arm varies only the mechanism, and averages over all of them

`design.TransplantTrial.wrong_mechanisms` is *every* other law the configuration
admits, not one arbitrary alternative. The first version picked the alphabetically
first, which made the control a constant: every trial sharing a source produced
the same wrong-arm error, and the contrast then measured how much the *correct*
latents varied rather than whether they were better. Averaging over the
alternatives gives the control the same power as the treatment and removes the
arbitrary choice.

The source context is held fixed between the correct and wrong arms, and required
to share the target's `S_0` tuple (compatibility is a function of the tuple) and to
have a *different* filler (which is what makes the reading cross-filler at all).

### 3. The error scale is floored against the widest dimension, not each dimension's own

`eval.transplant.trajectory_scale`. Phase IA's `ident.robust_scale` floors each
dimension at a fraction of that dimension's own spread, which is right when every
dimension moves. In a decoded rollout one does not: the supporter's velocity along
the gravity axis is held near zero by a mass an order of magnitude larger than the
probe's, so its MAD and its range are both dust and a self-relative floor is dust
too. Dividing by it weighted that single coordinate about a hundred thousand times
more heavily than the rest, every rollout scored as maximally wrong on it, and the
arm contrast was decided by floating point.

**Measured.** Before the fix the three arms sat at 2.81, 2.83 and 2.81 — and the
controlling observation was that the *derangement control* was numerically
identical to the real measurement, which cannot happen when the metric is carrying
signal. After the fix the scale's dead dimensions take the same floor as the live
ones and the arms separate.

### 4. Deviations are capped in scale units

`trajectory_error(..., cap=DEVIATION_CAP)` with `DEVIATION_CAP = 5.0`. An
autoregressive rollout of a contact law diverges: a decoder without the constraint
in it is off by a hair at step one and by hundreds of scale units at step ten, and
an unclipped root-mean-square then reports the *size of the divergence* rather
than whether the prediction followed the right law. Clipping makes every diverged
rollout score as maximally wrong, which is what it is, and leaves the contrast to
be decided by the steps before divergence. The unclipped error is still available
(`cap=None`) and is the right choice where the trajectories are known not to have
diverged.

### 5. The endpoint is reported with a must-fire control and a paired sensitivity measure

Two additions to the plan's `CFError`, both of which exist because the endpoint
returned a null and a null is not reportable without them. The first bounds the
null from the instrument's side; the second was meant to bound it from the
statistic's side and turned out not to, which is itself recorded.

- **The oracle code.** `run_phase1b --code oracle` feeds the decoder a one-hot
  mechanism label in place of the learned code, through identical machinery. If
  the decoder cannot turn a code that *is* the mechanism into a prediction that
  moves the way the law moves the world, the harness is broken and no null about
  the representation is reportable. This is the role `leaky_sampler_control` plays
  in Phase IA.
- **The alignment measure, and why it is reported as confounded.**
  `eval.transplant.sensitivity` compares the direction the code moves the
  prediction with the direction the law moves the world (`cosine` of the two
  differences). A decoder is an imperfect instrument and its absolute error is
  large for reasons unrelated to the code; both arms of a transplant share that
  error, so a difference of *errors* can be swamped by it while a difference of
  *predictions* cannot.

  **The control was wrong twice, and both corrections are instructive.** The first
  version deranged the pairing *within* a `(regime, law pair)` cell. Every element
  of such a cell is the same kind of pair, so permuting them leaves the mean
  exactly where it was and the contrast is identically zero — a control that
  cannot differ from its subject. The second drew the control from a *different*
  law pair in the same regime. Measured on the oracle code, that control's mean
  came out at **0.5901 against the real pairing's 0.5918**: both differences are
  dominated by the same large direction — gravity, and the action schedule — so
  *any* two of them are positively aligned, and no control drawn from this
  population can separate a matched code from an arbitrary one.

  So the statistic is now tested against **zero**, with a cluster bootstrap over
  contexts, and the control is kept as a **confound check**:
  `SensitivityResult.confounded` is true when the control's mean falls inside the
  pairing's interval. When it does, nothing may be read from the alignment and the
  endpoint's evidence rests on `CFError` alone. This is reported rather than
  suppressed because a statistic whose control cannot reject is a statistic that
  must not be quoted as though it had.

### 6. The CCGP test set is standardised by the training set's statistics

`audits.leakage.apply_standard`. The first implementation called `standardize` on
the test rows, giving them their own means and variances. That is transductive
adaptation: the readout sees test-time moments it could not have known at fit
time, and on a condition-generalization endpoint — where the whole question is
whether the code transfers to conditions the readout never saw — it biases the
number upward along exactly the axis being measured.

### 7. What a null on this endpoint means

The claim ladder in ADR 0007 already says it: cross-condition generalization plus
role structure with a failed transplant licenses "no causal claim on the relation
variable", not "abstract relational representation emerged". Two further
qualifications are required of any report citing a null here, both established by
measurement:

- **Zeroing an input is not an ablation.** `run_phase1b` records the arm's own
  prediction loss with the code zeroed, which rises four- to seven-fold. That
  number cannot be read as "the predictor routes through the code": a zero vector
  is out of distribution for a ReLU MLP, and most of the increase is distribution
  shift rather than information loss. The reading it *does* support is the weak
  one — the code is not inert in the arm — and it must be reported as such.
- **The mechanism explains only a small part of the rollout loss.** Measured by
  `diagnose_horizon`: the same decoder trained under a true one-hot label and a
  deranged one differs by a mixed-sign gap of about 10% across horizons from 1 to
  20. Where the label itself buys little, no code can be read, and the endpoint's
  power is bounded by that rather than by the representation.

## Consequences

A Phase IB report must state, for the transplant endpoint: the code mode
(`learned` or `oracle`), the horizon, the regime split, the alignment against its
derangement, and the must-fire control's result. Omitting any of them makes the
number uninterpretable in a way that is not visible from the number itself.

The diagnostics that produced these decisions are kept rather than discarded —
`experiments/diagnose_codes.py`, `diagnose_horizon.py`, `diagnose_transplant.py`.
They are not part of the pipeline; they are the evidence behind how the endpoint
is specified, and a reader who doubts a clause above can re-run the clause.

## Evidence

- Plan/source: Phase I plan text supplied 2026-09-23 (not tracked in this
  repository), sections 16, 18 and 27.
- `tools/relational_emergence/eval/transplant.py`,
  `tools/relational_emergence/eval/design.py`,
  `tools/relational_emergence/models/transplant.py`,
  `tools/relational_emergence/models/representation.py`,
  `tools/relational_emergence/experiments/diagnose_*.py`.
- Tests: `tests/analysis/test_relational_emergence_endpoints.py`.
- Related: ADR 0005 (data-generating process), ADR 0007 (evidence criteria and the
  Level-2 amendment, whose failure mode this repeats and names).
