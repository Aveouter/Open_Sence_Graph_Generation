# Phase IB — the factorial arms and the three primary endpoints

**Status:** `research_experiment` · **Scope:** `synthetic_relational_emergence_phase1b`
· **World:** `development_world` · **Not a reproduction.**

Run: `python -m tools.relational_emergence.experiments.run_phase1b` (288 base
contexts, 3168 episodes, all seven arms in one invocation, 515 s wall clock).
Every number below is read from `outputs/analysis/relational_emergence/phase1b/` —
the tables are emitted by `experiments/summarise_phase1b.py` straight from the
artifacts rather than retyped. The `config_sha256` stamped into every artifact
covers the world constants and the generator seed as well as the config dataclass.

Phase IA asked whether the mechanism is identifiable from the observable state
given the actions. This phase asks what the plan is actually about: *what*
experience and *what* inductive bias make a predictive model form a relational
code that survives a change of filler. The arms are trained on the episodes Phase
IA froze and audited; neither phase re-samples what the other measured.

The endpoint definitions and the implementation decisions they forced are in
[ADR 0008](../../../reproduction/adr/0008-phase1b-endpoint-specification.md).
The protocol freeze for the sealed run is
[ADR 0009](../../../reproduction/adr/0009-phase1b-protocol-freeze.md); the sealed
result is in [04](04_phase1b_audit_world.md).

---

## 1. What was run

Seven arms, one invocation, one frozen dataset. Every non-compared thing is held
fixed: same split, same contexts, same optimiser, same epoch budget, and the same
checkpoint rule. The five unsupervised arms are capacity-matched to within 3.9%
by construction (`representation._matched_global_hidden`); the two supervised ones
are not, and §2 says why. What varies between the compared arms is the pairwise
bottleneck and the temporal order.

| arm | temporal | pairwise bottleneck | relation labels |
|---|---|---|---|
| `S-G` | no | no | yes |
| `S-R` | no | yes | yes |
| `StaticSSL-G` | no | no | no |
| `StaticSSL-R` | no | yes | no |
| `Shuffle-R` | reversed | yes | no |
| `P-G` | yes | no | no |
| `P-R` | yes | yes | no |

The supervised arms are a reference, not a target: the phase does not ask that
`P-R` beat `S-R`.

Three partitions, kept apart on purpose and asserted rather than assumed
(`eval/design.py`). The **context** partition decides what the encoder may learn
from, with a slice of the training contexts held back for checkpoint selection.
The **condition** partition decides what the readout is fit on and what it is
scored on. The **evaluation** side is untouched by every choice the run makes.

### The objective is multi-step, and that is load-bearing

Both the arms and the transplant decoder roll four steps (§18 of the plan,
[ADR 0008 §1](../../../reproduction/adr/0008-phase1b-endpoint-specification.md)).
This is not a hyperparameter. With a one-step objective the next state is largely
determined by the current one, so a decoder can fit without reading the code at
all — measured, 193 of 200 swapped-code pairs produced a **bit-identical**
prediction. An instrument trained that way reports its own indifference as its
subject's irrelevance, and the two are indistinguishable in the output. The same
failure class as the direct-head oracle in ADR 0007.

### The chance level is the mean of reciprocals

A condition holds `S_0` tuples with different admissible mechanism counts, so its
compatibility-aware chance level is the mean of `1/|compatible|`, not the
reciprocal of a mean count. The latter rounds any mix averaging between 3.5 and
4.5 onto 4 and reports **0.250** where the true level is **0.2727** — a
systematic understatement of chance, and so a systematic overstatement of every
result measured against it. Phase IA's independent value of 0.273 confirms the
corrected figure.

---

## 2. Training

| arm | parameters | val loss | best epoch |
|---|---|---|---|
| `P-G` | 23880 | 0.092215 | 16 |
| `P-R` | 24008 | 0.091507 | 16 |
| `S-G` | 31692 | 0.751508 | 36 |
| `S-R` | 31820 | 0.743991 | 39 |
| `Shuffle-R` | 24008 | 0.091220 | 17 |
| `StaticSSL-G` | 23112 | 0.090575 | 18 |
| `StaticSSL-R` | 23240 | 0.089982 | 30 |

The five unsupervised arms are capacity-matched to within 3.9% (23 112 to 24 008),
which is the comparison the phase rests on: a difference between a bottleneck and
a differently-sized model would measure capacity rather than factorization.
`P-R` and `Shuffle-R` are **identical** at 24 008, as they must be — they differ
only in whether the history is reversed.

**The supervised arms are not matched to the rest** (31 692 and 31 820, about 33%
larger): they carry an auxiliary classification head, which is their definition.
Their validation losses are also on a different scale, because the auxiliary
cross-entropy is added to the prediction loss. They are a reference rather than a
comparison, and no reading below rests on their capacity or loss being
comparable.

---

## 3. Primary endpoint A — CCGP on unseen appearance families

| arm | accuracy | baseline | above baseline |
|---|---|---|---|
| `P-G` | +0.5518 | +0.2727 | +0.2791 |
| `P-R` | +0.5603 | +0.2727 | +0.2876 |
| `S-G` | +0.6222 | +0.2727 | +0.3494 |
| `S-R` | +0.6176 | +0.2727 | +0.3448 |
| `Shuffle-R` | +0.5603 | +0.2727 | +0.2876 |
| `StaticSSL-G` | +0.5596 | +0.2727 | +0.2869 |
| `StaticSSL-R` | +0.5677 | +0.2727 | +0.2949 |

This is the headline. The readout is fit on one set of appearance families and
scored on families it never saw, against the compatibility-aware baseline, with
the test set standardised by the **training** set's statistics — scaling it by its
own would let the readout use test-time moments it could not have had at fit time,
which is transductive adaptation on precisely the axis this endpoint measures.

### The four split kinds

`P-R` at the headline sample size, one line per split (288 base contexts, 40
epochs, capacity and seed identical):

| split | held out | CCGP above baseline | role gain |
|---|---|---|---|
| `unseen_family` (headline) | `dev_brushed_metal`, `dev_matte_polymer` | +0.2876 | +0.1280 |
| `iid` | — (random contexts) | +0.2949 | +0.0954 |
| `unseen_filler` | 12 of the 40 filler identities | +0.2817 | +0.0657 |
| `pair_recombination` | identity pairs `i0/j2`, `i1/j1` | +0.3010 | +0.1231 |

**These four numbers are not comparable to each other**, and the table is here
because the plan asks for all four rather than because the differences mean
anything. CCGP's *condition* partition holds out appearance families in every
row — that never changes. What changes is whether the encoder saw those families
during training: for `iid`, `unseen_filler` and `pair_recombination` it did, and
for `unseen_family` it did not. So only the headline row measures a code that
generalizes to a family the encoder never encountered; the other three measure a
readout's transfer to conditions the *encoder* had seen, which is a weaker claim
and a different question.

### By intervention regime

| arm | passive | weak | rich |
|---|---|---|---|
| `P-G` | +0.3140 | +0.2934 | +0.2633 |
| `P-R` | +0.3587 | +0.3246 | +0.2742 |
| `S-G` | +0.3801 | +0.3773 | +0.3439 |
| `S-R` | +0.3833 | +0.3735 | +0.3203 |
| `Shuffle-R` | +0.3644 | +0.3140 | +0.2634 |
| `StaticSSL-G` | +0.3419 | +0.3241 | +0.2511 |
| `StaticSSL-R` | +0.3517 | +0.3360 | +0.2602 |

The plan's M5. If relation identifiability is what makes abstraction possible, the
endpoint should improve with intervention richness; Phase IA measured `J_ab^AUC`
at **0.020 passive / 4.069 rich**, so the world provides the contrast the arms are
being asked to exploit. Every arm runs the other way — §10 takes this up.

---

## 4. Primary endpoint B — role equivariance on unseen families

| arm | E_role | E_shuffled | gain | E_inv |
|---|---|---|---|---|
| `P-G` | +0.0650 | +0.2659 | +0.2010 | +0.2371 |
| `P-R` | +0.1473 | +0.2752 | +0.1280 | +0.2258 |
| `S-G` | +0.2223 | +1.1690 | +0.9467 | +0.2400 |
| `S-R` | +0.8481 | +1.3879 | +0.5397 | +0.1992 |
| `Shuffle-R` | +0.2010 | +0.2937 | +0.0927 | +0.2210 |
| `StaticSSL-G` | +0.0464 | +0.1823 | +0.1359 | +0.2424 |
| `StaticSSL-R` | +0.2266 | +0.3014 | +0.0749 | +0.2102 |

One orthogonal `T_swap` is fitted on the development families and scored on the
unseen ones — never refitted per family, which would let each family have its own
transform and make the number say nothing about a shared role structure.
`E_shuffled` is the control: the same fit scored against a deranged pairing of
forwards and reverses. `role_gain = E_shuffled − E_role` is the quantity to read;
a transform that swaps nothing achieving a comparable residual would mean
`E_role` was measuring how alike the latents are in general.

Orthogonal rather than general linear because a general map has `d^2` free
parameters against `d(d−1)/2` and can absorb an arbitrary re-encoding — after
which neither number says anything about roles. `E_inv = ||T^2 − I||` is the
plan's §15 involution expectation, and it is reported because it *fails*: see §10.

---

## 5. Supporting endpoint — label efficiency (H3)

| arm | 1 | 5 | 10 | 50 | None |
|---|---|---|---|---|---|
| `P-G` | -0.0044 | +0.1373 | +0.2394 | +0.2503 | +0.2791 |
| `P-R` | +0.0503 | +0.0784 | +0.1812 | +0.2595 | +0.2876 |
| `S-G` | +0.0571 | +0.2458 | +0.2988 | +0.3333 | +0.3494 |
| `S-R` | +0.0694 | +0.0574 | +0.2919 | +0.3292 | +0.3448 |
| `Shuffle-R` | +0.0558 | +0.0915 | +0.2413 | +0.2726 | +0.2876 |
| `StaticSSL-G` | -0.0053 | +0.0748 | +0.2523 | +0.2813 | +0.2869 |
| `StaticSSL-R` | +0.0219 | +0.0731 | +0.2004 | +0.2753 | +0.2949 |

The encoder is frozen and the readout is refit on 1, 5, 10, 50 and all labels per
class, on the H3 question: how few labels reach the mechanism in the code. The
test side is held fixed across budgets on purpose — shrinking it with the label
budget would make the curve measure two things at once.

`S-G` and `S-R` lead at every budget, including one label per class. Among the
unsupervised arms the ordering is not the hypothesized one: `Shuffle-R` is above
`P-R` at 1, 5 and 10 labels, and `StaticSSL-R` is above both at the full budget.

---

## 6. Primary endpoint C — counterfactual latent transplant

| arm | CFError self | correct | wrong | gain | CI | positive |
|---|---|---|---|---|---|---|
| `P-G` | +0.9586 | +0.9745 | +0.9745 | -0.0000 | [-0.0000, +0.0000] | False |
| `P-R` | +1.0182 | +1.1433 | +1.1433 | +0.0000 | [-0.0000, +0.0000] | False |
| `S-G` | +1.0904 | +1.1402 | +1.1402 | -0.0000 | [-0.0000, +0.0000] | False |
| `S-R` | +1.0093 | +1.0840 | +1.0840 | -0.0000 | [-0.0000, +0.0000] | False |
| `Shuffle-R` | +1.1535 | +1.2632 | +1.2632 | +0.0000 | [-0.0000, +0.0000] | False |
| `StaticSSL-G` | +0.9807 | +1.0044 | +1.0044 | +0.0000 | [-0.0000, +0.0000] | False |
| `StaticSSL-R` | +1.0113 | +1.0700 | +1.0700 | +0.0000 | [-0.0000, +0.0000] | False |

A decoder is trained on the frozen encoder's outputs, then held fixed while only
the code it is fed varies. Three arms per evaluated context, and the source
context is held fixed between the last two so that only the mechanism changes:

- **`self`** — the code from this context's own episode under the target law. Not
  a causal claim; a ceiling, so a gain can be read against a scale.
- **`correct`** — the code from a **different** context (different filler, same
  `S_0` tuple) under the target law.
- **`wrong`** — that same source under every other law it admits, **averaged**. All
  of them rather than one arbitrary alternative: a single fixed alternative makes
  the control a constant, and the contrast then measures how much the correct
  latents *vary* rather than whether they are better.

`gain = mean(wrong) − mean(correct)`, with a cluster bootstrap over contexts —
contexts, not rows, because the rows inside one share their `X_0`, their action
schedule and their filler.

**Every arm returns exactly zero**, and `correct` equals `wrong` to four decimals.

### Why the number is unreadable without its controls

#### The alignment statistic, and why it is reported as confounded

| arm | passive | weak | rich | overall | control mean | confounded |
|---|---|---|---|---|---|---|
| `P-G` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `P-R` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `S-G` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `S-R` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `Shuffle-R` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `StaticSSL-G` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |
| `StaticSSL-R` | n/a | n/a | n/a | no scorable pairs (code moved nothing: 4048; law inactive: 272) | n/a | n/a |

A decoder is an imperfect instrument: its absolute error is large for reasons
unrelated to the code, and both arms of a transplant share that error, so a
difference of errors can be swamped by it while a difference of *predictions*
cannot. `eval.transplant.sensitivity` compares the direction the code moves the
prediction with the direction the law moves the world.

It is reported **confounded**, and the reason is measured rather than assumed. The
first control deranged the pairing *within* a `(regime, law pair)` cell — where
every element is the same kind of pair, so the permutation leaves the mean exactly
where it was and the contrast is identically zero. The second drew the control
from a *different* law pair. On the oracle code that control's mean came out at
**0.5901 against the real pairing's 0.5918**: both differences are dominated by
the same large direction — gravity and the action schedule — so any two of them
are positively aligned, and no control drawn from this population can separate a
matched code from an arbitrary one.

The statistic is therefore tested against **zero**, and `confounded` flags the
case where the control falls inside the pairing's interval. It does here, so
nothing is read from the alignment and **the endpoint's evidence is the `CFError`
contrast above**.

#### Pairs excluded rather than scored

**4048 of 4320 pairs were excluded because the code did not move the decoded
prediction at all**, and 272 because the law made no difference to that context —
leaving zero scorable pairs in every arm. Both are counted rather than scored as
zero, because a zero would be a claim ("the code moved the prediction
perpendicular to the law") where the truth is that the pair says nothing. The two
are different facts — one about the instrument, one about the world — and they are
counted apart for that reason.

#### The must-fire control, and it fires

`--code oracle` feeds the decoder a one-hot mechanism label in place of the
learned code, through identical machinery. It was run once, with
`--arms P-R --groups-per-tuple 48 --epochs 40`: the oracle code does not depend on
the arm at all — it replaces what the encoder emits — and the decoder is trained
from the same seed on the same rows, so the number holds for every arm. That is
an argument rather than a per-arm measurement, and it is stated as one.

It produces `CFError` gain **+0.1856** (`self` and `correct` both 1.1613, `wrong`
1.3524), an alignment of **+0.5948**, and **no unscorable pairs at all** — against
the learned code's 0.0000 and 4048 of 4320. The harness responds to a code that
*is* the mechanism by a large margin, so
the null on the learned code is not a dead device.

---

## 7. Diagnostics, and what they do not support

| arm | with code | without code | relative increase |
|---|---|---|---|
| `P-G` | +0.0239 | +0.1388 | +4.8017 |
| `P-R` | +0.0248 | +0.1551 | +5.2444 |
| `S-G` | +0.0250 | +0.2318 | +8.2553 |
| `S-R` | +0.0235 | +0.1899 | +7.0883 |
| `Shuffle-R` | +0.0234 | +0.1424 | +5.0789 |
| `StaticSSL-G` | +0.0236 | +0.0854 | +2.6192 |
| `StaticSSL-R` | +0.0240 | +0.2017 | +7.4151 |

`code_ablation` records the arm's own one-step loss with the bottleneck zeroed.
The rise is large, and it **cannot** be read as "the predictor routes through the
code": a zero vector is out of distribution for a ReLU MLP, and most of the
increase is distribution shift rather than information loss. The reading it
supports is the weak one — the code is not inert in the arm.

`diagnose_horizon` measures the endpoint's ceiling independently of any encoder:
the same decoder trained under a true one-hot mechanism label versus a deranged
one differs by a **mixed-sign gap of about 10%** across horizons from 1 to 20
(steps=1 −6.5%, 2 −9.3%, 4 −14.0%, 6 +16.3%, 8 −6.7%, 12 +5.8%, 20 −1.9%). Where
the label itself buys little, no learned code can be read, and the endpoint's
power is bounded by that rather than by the representation.

---

## 8. Across-seed spread

| arm | CCGP mean | min | max | role gain mean |
|---|---|---|---|---|
| `P-G` | +0.2531 | +0.2386 | +0.2677 | +0.2082 |
| `P-R` | +0.2787 | +0.2740 | +0.2866 | +0.1721 |
| `Shuffle-R` | +0.2946 | +0.2816 | +0.3109 | +0.1450 |
| `StaticSSL-R` | +0.2756 | +0.2737 | +0.2765 | +0.1496 |

The frozen protocol runs one seed, so this is reported as a limitation rather than
as the result. It exists because a between-arm difference is only readable against
an across-seed one, and comparing them after the seed runs are gone is not
possible. These runs use 12 base contexts per tuple and 25 epochs rather than 48
and 40, so they bound the seed variation at a smaller sample and over a shorter
budget, and are **not** the headline's error bars.

---

## 9. Deviations from the plan

Recorded because the plan asks each milestone to state them, and because a
deviation that is not written down reads as an omission.

| Plan item | What was done | Why it does not change the interpretation |
|---|---|---|
| §14 parallelism score `Γ` | not implemented | the plan marks it descriptive-only and forbids it as primary evidence; omitting it removes a diagnostic, not an endpoint |
| §22 discrete latent `K ∈ {2,4,8,16}` | continuous `z` only | the sweep is conditional in the plan ("如果测试 discrete z"); the checkpoint rule it constrains is applied as written |
| §21 temporal shuffle | history **reversed**, not permuted | with a four-frame history reversal destroys the ordering as a permutation does and is deterministic; a random permutation would be a stronger control and is a candidate for a later run |
| §29 `eval/static_transfer.py`, `experiments/run_audit.py` | folded into `label_efficiency` in `run_phase1b` and a `--world` flag | the interfaces are the same; two entry points rather than four |
| §31 M4–M8 as separate milestones | one driver, one invocation | the arms share one dataset and one split, so separate invocations would re-derive the partitions and could disagree |
| §1.5 ground-truth object state | used, as permitted | object perception is out of scope for this phase by construction |
| renderer `O_t = R(X_t, F, B, C)` | none; the state is the observation | ADR 0005 records the decision and its consequence: appearance enters only through the nuisance attributes, so the leakage and family-holdout audits test the intended channels rather than a rendering pipeline |

---

## 10. Interpretation

**Endpoint A is met. Endpoint B is met in the weak sense only. Endpoint C fails.
The phase's two architectural hypotheses are not supported.**

### Relation coding generalizes across conditions

Every arm decodes the mechanism on appearance families it never saw, well above
the compatibility-aware chance level: the five unsupervised arms land between
**+0.279 and +0.295** above a 0.2727 baseline, i.e. accuracy **0.552–0.568**
against 0.273 for guessing a compatible law. This is the ADR 0007 ladder's second
rung — *relation coding generalizes across conditions* — and it is met.

### But the hypotheses that predict *why* are not

The phase is built on two contrasts. Neither survives.

**H1, temporal teaching: contradicted.** `Shuffle-R` destroys the history ordering
and scores **+0.2876** on CCGP — identical to `P-R` to four decimals, at the same
accuracy 0.5603. `StaticSSL-R`, which has no temporal input at all, scores
**+0.2949**, above both. At three seeds on a smaller sample the same pattern holds
(`Shuffle-R` +0.2946 against `P-R` +0.2787). There is no measured sense in which
temporally ordered predictive experience produces a better cross-family code here
than matched shuffled or static experience.

**H2, relational bottleneck: not supported.** `P-R` (+0.2876) against `P-G`
(+0.2791) is a gap of 0.0085, and the across-seed spread of `P-R` alone on a
smaller sample spans 0.0126. The direction is at least consistent with the
hypothesis, but the headline scale does not resolve it. On role equivariance the
ordering is *reversed*: `P-G` gains **+0.2010** against `P-R`'s +0.1280, and
`StaticSSL-G` (+0.1359) beats `StaticSSL-R` (+0.0749). The arm with the pairwise
bottleneck has the *weaker* role structure, which is the opposite of what the
bottleneck hypothesis predicts.

What this is, in the plan's decision matrix, is **Case C made stronger**. Case C
says "`P-R` and `P-G` both strong → an explicit relational bottleneck is not
necessary under sufficiently informative predictive experience". Here it is not
only the bottleneck that is unnecessary: the temporal ordering and the presence
of time at all are equally unnecessary. Five arms that differ in what they can
see reach the same number.

**A structural caveat, stated because it weakens the above.** All three endpoints
read the *same* snapshot encoder, and CCGP reads it through a linear ridge
readout. A linear readout of a 32-dimensional code under 1272 training conditions
is the most forgiving possible probe of "is `M` in there". A null on the
*architecture* contrast could mean the architecture does not matter, or it could
mean this probe cannot see the difference.

The evaluator does discriminate on synthetic latents whose answer is known: over
eight seeds, a mechanism-carrying latent scores **+0.7448** above baseline
(range +0.7083 to +0.7500) and a condition-only latent scores **+0.0156**
(−0.0417 to +0.0833). So the readout is not blind. What that does **not**
establish is that a *nonlinear* readout would also fail to separate the arms —
and it is the specific worry here, because a bottleneck could put `M` into a
nonlinear function of `z` that a ridge readout cannot reach.

### Intervention richness runs the wrong way

Phase IA measured the mechanism as far more identifiable under the rich regime:
`J_ab^AUC` **0.020 passive / 4.069 rich**. Every arm here generalizes *worse*
under rich and *better* under passive — `P-R` at **+0.3587 passive / +0.2742
rich**, and the same ordering in all seven arms.

This is a dissociation worth stating precisely: more identifiable interaction
experience in the trajectory made the code generalize *less* well across
appearance families. The plan's M5 predicted the opposite.

**Hypothesis, not measurement.** Under the rich regime the trajectory is dominated
by the exogenous impulses, so an encoder minimizing prediction error can explain
most of the variance from the action stream. The action stream varies across
conditions, so the code acquires condition-specific structure the readout then has
to unlearn — which costs more than the mechanism information it gains. This is a
plausible account and it is **not tested here**; distinguishing it from "the rich
regime simply has harder geometry" needs a run this phase did not do.

### Role structure exists, but not as an involution

Every arm has `role_gain > 0`: one orthogonal `T_swap` fitted on the development
families explains the unseen families' forwards/reverses better than a deranged
pairing does. Role structure is therefore present and shared across families.

But `E_inv` sits between **0.199 and 0.242** in every arm — `T^2` is nowhere near
the identity, against residuals (`E_role`) of 0.046–0.848 on the same scale. The
plan's §15 asks that `T_swap^2 ≈ I`. It does not hold, in any arm, including the
supervised ones. A shared transform exists; a shared *involution* does not.

### Endpoint C fails, and the failure is bounded on both sides

`CFError` gain is **±0.0000** in all seven arms — `correct` and `wrong` agree to
four decimals — and no sensitivity pair was scorable. The ladder forbids a causal
claim on the relation variable, and none is made.

Three facts bound what that can mean, and all three are measured:

- **The instrument fires on a code that is the mechanism**: the oracle control
  returns **+0.1856** where the learned code returns **0.0000**, with no unscorable
  pairs against 4048 of 4320.
- **The decoder assigns the learned code no weight at all.** Swapping the code
  across mechanisms changes its output by less than float32 resolution, while the
  oracle's code moves the same decoder substantially.
- **The world bounds the endpoint independently of any encoder.** §7's horizon
  sweep: a decoder given the true mechanism as a label beats a deranged label by a
  mixed-sign gap of about 10%.

Read together these give a sharper statement than "the endpoint is underpowered",
and a narrower one than "the codes are not portable". The decoder can express a
mechanism code and does act on the oracle's, so the learned codes are not being
*misread* — they are being *dropped*. What that does **not** establish is where
the failure lies: a decoder that finds the code useless for its own objective will
drop it whether the code is uninformative or merely hard to exploit. The learned
code is linearly decodable for `M` at 0.5603 against a 0.2727 chance level (§3),
so it is not uninformative — which leaves the exploitation as the untested step.

The phase therefore reports **no evidence of counterfactually portable predictive
use**, and the ladder forbids the causal claim. It does not report that the code
carries nothing, because §3 says otherwise, and it does not report that the code
fails to transfer across fillers, because the measurement that would show that was
never reached.

### What the supervised arms say

`S-R` and `S-G` reach **+0.3448** and **+0.3494**, roughly 0.055 above the best
unsupervised arm, with role gains of +0.5397 and +0.9467. Labels help, and help
most on the endpoint the unsupervised arms are weakest at. That is a ceiling
reading, not a target: the phase does not ask that `P-R` beat `S-R`, and the
supervised arms are a reference rather than a comparison — they carry an auxiliary
classification head and are correspondingly larger, so their numbers are not
capacity-matched to the rest.

### The claim, in the ladder's terms

Permitted: **relation coding generalizes across conditions**, and role structure
is present and shared across unseen families.

Not permitted: **abstract relational representation emerged** — endpoint C fails,
and the ADR 0007 ladder requires all three.

And a finding the ladder has no rung for, which is the phase's actual result:
**under this world and this objective, the cross-family code does not depend on
the inductive biases the phase was built to test.** Removing the pairwise
bottleneck, reversing the history, or deleting time entirely leaves the headline
number where it was.

---

## 11. What is not claimed

- **Not a reproduction.** No SGG checkpoint, no benchmark, no real data. The world
  is a hand-written simulator, so a result here is a statement about one fully
  specified data-generating process and nothing wider.
- **No emergence claim** beyond what the claim ladder in ADR 0007 licenses, and
  the ladder is applied in §10 explicitly rather than by implication.
- **Relation probes, clustering and geometry diagnostics are not primary
  evidence** and are not cited as such anywhere above.
- **The transplant endpoint's null is bounded on both sides.** §6 and §7 state
  both bounds; neither is dropped when the other is quoted.
- **The alignment statistic is confounded** and nothing is read from it. It is
  reported because a statistic whose control cannot reject must be visible rather
  than quietly dropped.
- **The across-seed runs are not the headline.** They use a smaller sample and a
  single split kind.
- **This is the development world.** The sealed result is
  [04](04_phase1b_audit_world.md), and where the two disagree both are reported.
