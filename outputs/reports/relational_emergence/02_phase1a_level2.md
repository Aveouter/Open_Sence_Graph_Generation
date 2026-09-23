# Phase IA — Level-2 oracle utility

**Status:** `research_experiment` · **Scope:** `synthetic_relational_emergence_phase1a`
· **World:** `development_world` · **Not a reproduction.**

> This report covers **two measurements of the same criterion**, and the second
> supersedes the first. The oracle's parameterisation turned out to be part of the
> criterion rather than an implementation detail of it, and ADR 0007 was amended
> to say so. §1 reports the **direct** head, which failed to measure anything;
> §2 reports the **rollout** head, which is the primary and which produced the
> verdict. The two loss scales are not comparable to each other — what is
> comparable is `delta_M`, a paired difference within a run.

The **exact** commands, because the flags are part of the measurement and a
reader who takes the defaults does not reproduce it — the `OracleConfig` defaults
are 200 epochs at batch 64, while the rollout below was measured at 60 epochs and
batch 256, and the two give different `delta_M` and different across-seed spread:

```bash
# the rollout head, which produced the verdict
python -m tools.relational_emergence.experiments.run_level2 \
    --oracle rollout --epochs 60 --hidden 128 --depth 3 --batch-size 256 \
    --out outputs/analysis/relational_emergence/phase1a/level2.json

# the direct head, which is the defaults plus the mode
python -m tools.relational_emergence.experiments.run_level2 \
    --oracle direct --epochs 200 \
    --out outputs/analysis/relational_emergence/phase1a/level2_direct.json

# folded into the gate
python -m tools.relational_emergence.experiments.run_phase1a \
    --level2 outputs/analysis/relational_emergence/phase1a/level2.json
```

An earlier version of this line said only "defaults to the rollout head", which
is true of the *mode* and false of the hyperparameters. Every number is read from
`outputs/analysis/relational_emergence/phase1a/`; each artifact pins the Phase IA
config digest, the oracle's own hyperparameter digest
(`oracle_config_sha256` — the field that would have caught the mistake), and the
SHA-256 of the frozen dataset it read.

---

## 1. The direct head — measured, and not measurable

Predicting `(X_0, F, A) -> X_1..X_H`, 480 outputs from one snapshot, 200 epochs,
3×128 MLP.

| arm | mean validation loss |
|---|---|
| `TrueM` | 0.858767 |
| `ShuffledM` | 0.860141 |
| `Base` | 0.863291 |

| quantity | value | 95% CI | seeds agreeing in sign |
|---|---|---|---|
| `delta_M` = `L_ShuffledM` − `L_TrueM` | +0.001374 | [−0.000393, +0.003141] | 4 of 5 |
| `shuffled_vs_base` = `L_Base` − `L_ShuffledM` | **+0.003151** | [+0.001504, +0.004797] | **5 of 5** |

**Both Level-2 conditions fail, and the second failure is the informative one.**

The ordering is right — `TrueM` < `ShuffledM` < `Base` — but the margins are
~0.2% of a base of ~0.86 and `delta_M` has an interval containing zero, with one
seed of five against. The cause is visible in the training trace: train loss
0.49–0.66 against validation 0.80–0.97, best epoch **2–11 of 200**, against a
mean-predictor reference of 1.05–1.16 on the same held-out groups. The model fits
the training contexts and does not transfer to unseen ones, so all three arms sit
near the trivial predictor and the `M` slot has no room to show an effect.

`shuffled_vs_base` is the part worth reading. It is **positive and excludes zero
in all five seeds**: a *wrong but compatible* label is reliably better than no
label at all. The gate's third condition exists for exactly this, and it fires —
a shuffled label still tells the model which mechanisms the group's `S_0` tuple
admits, so it leaks the condition while carrying nothing about the law. `TrueM`
then adds nothing on top of that leak, which is why it cannot beat `ShuffledM`.

So the honest reading is not "`M` carries no utility". It is "this oracle does
not measure it, and what signal it does have is the label leaking the group
rather than the law" — and those are indistinguishable in a results table, which
is exactly the gap M0 closes with a pre-registered minimum detectable effect.

---

## 2. The rollout head — the primary measurement

Trained on `(X_t, F, A_t) -> delta X_{t+1}` and scored by rolling the transition
out autoregressively for **10 steps** — the horizon Level-1 selected from the
`J_ab` saturation curve, so the oracle is judged over the interval the gate treats
as covering the interaction consequence. 5 seeds, 60 epochs, 3×128 (38 280
parameters), batch 256, ~152 000 training transitions against ~38 000 validation.

| arm | mean validation loss |
|---|---|
| `TrueM` | **0.252908** |
| `ShuffledM` | 0.418445 |
| `Base` | 0.423402 |

| quantity | value | 95% CI |
|---|---|---|
| `delta_M` = `L_ShuffledM` − `L_TrueM` | **+0.165537** | [+0.135660, +0.195414] |
| `shuffled_vs_base` = `L_Base` − `L_ShuffledM` | +0.004957 | [−0.017898, +0.027811] |

Per seed, `delta_M` = [+0.19734, +0.14985, +0.14162, +0.18455, +0.15433] —
**all five positive**, spread 0.056.

Minimum detectable effect, derived from the artifact's own per-seed deltas:
**0.0389**, against the 0.05 declared in `OracleConfig.delta_m_mde_target` before
the run. The comparison is adequately powered, so a null would have meant
*absent*; this is not a null.

### Verdict

```
passed: True
  TrueM beats ShuffledM, ci excluding zero by a wide margin
  ShuffledM does not beat Base (interval spans zero)
blocked_by: []
```

Folded into the gate, `run_phase1a --level2` reports **`GO_PHASE_IA`**. Level-1
passed independently; see report 01.

Note that the control's *point estimate* is positive here (+0.004957): the
criterion is that the interval spans zero, and it does, with the shuffled arm
only 0.005 above base against a `delta_M` of 0.166. Reporting the point estimate
as "negative" — as an earlier version of this block did — was true of the
previous dataset and is not true of this one, and the sign of a quantity whose
interval spans zero is not a finding either way.

### The four implementation decisions that made it measurable

Each was found by measuring, and each had produced a silent failure:

1. **Residual targets.** Predicting `delta X` rather than `X` makes the identity
   map mean "nothing moved", which is a sane baseline for a one-step physical
   transition, and keeps an autoregressive rollout on the data manifold.
2. **A bounded evaluation horizon.** Rolling all 60 steps out compounded the
   model's own error until the loss measured divergence rather than information —
   observed at O(10³) with a trivial predictor scoring O(1).
3. **Every input block standardised, including the filler block.** It holds the
   masses, which run to about 18 where every other block is order 1. Left raw it
   dominated the first layer and the outputs diverged; the symptom was a training
   loss in the hundreds.
4. **State statistics pooled across the horizon.** `dataset.y` is the whole
   trajectory flattened, so a state dimension lives at `step * w_y + dim`.
   Taking the first `w_y` columns scales by the release state alone, which is far
   quieter than the rest of the trajectory.

---

## 3. Interpretation

**The mechanism is usable, and substantially so.** Knowing `M` correctly instead
of incorrectly reduces rolled-out prediction error by 0.166, in units where the
trivial predictor scores 1.0 — a **40% reduction relative to the shuffled
control**. The effect is larger than the entire gap between the shuffled and base
arms by a factor of thirty, and it holds in all five seeds.

**The control is clean.** `ShuffledM` does not beat `Base`: the interval spans
zero. A wrong-but-legal label is therefore not worth more than no label — which
is precisely what the direct head's control *did* show, and the contrast between
the two parameterisations on this one quantity is as sharp as the contrast in
`delta_M` itself (§1: +0.003151, five seeds of five).

**The two modes disagree, and that is the finding.** The same criterion, the same
three arms, the same checkpoint rule, the same data — and one parameterisation
measures nothing while the other measures a large and consistent effect. This is
not measurement noise; it is a statement about how the criterion is realised. A
gate that had fixed only the comparison and left the parameterisation implicit
would have returned `STOP` on the strength of an unlearnable task.

**What this does not yet establish.** The rollout's `delta_M` is not "information
per step": compounding amplifies whatever one-step advantage a correct label
confers, so the number is larger than the per-step effect and the two modes'
losses are not comparable. And validation loss was still improving at the end of
the budget for two of five seeds (best epoch 59 of 60), so 60 epochs is not
obviously enough — the effect is a lower bound in that respect.

---

## 4. Not claimed

- **That any of this transfers to a real model.** The oracle is a small MLP on
  simulator state. It shows the mechanism is *usable by a generic predictor with
  the right parameterisation*; it says nothing about what a predictor will
  discover when it has to find the representation itself.
- **Any headline result.** `CCGP`, `RoleEquivariance` and `CFError` remain
  pre-registered endpoints with **no measured value**.
- **That the two parameterisations agree.** They do not, and the disagreement is
  reported rather than averaged away. Neither loss scale is comparable across the
  change.
- **That `delta_M = 0.17` is a per-step quantity.** It is a ten-step rolled-out
  figure and compounds.
- **That this is a reproduction of anything.** See `AGENTS.md`.
