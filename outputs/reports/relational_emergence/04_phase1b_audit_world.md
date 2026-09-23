# Phase IB — the sealed audit world

**Status:** `research_experiment` · **Scope:** `synthetic_relational_emergence_phase1b`
· **World:** `audit_world` · **Not a reproduction.**

Run: `python -m tools.relational_emergence.experiments.run_phase1b --world
audit_world` (288 base contexts, 3168 episodes, seven arms, 512 s wall clock),
under the protocol frozen in
[ADR 0009](../../../reproduction/adr/0009-phase1b-protocol-freeze.md). Artifacts
under `outputs/analysis/relational_emergence/phase1b_audit/`; the tables are
emitted by `experiments/summarise_phase1b.py` straight from them. The
development-world counterpart is
[03](03_phase1b_factorial.md), and §3 compares the two.

---

## 1. What the seal is, and what it is not

`audit_world` runs the **identical generator** on a **disjoint sample**: filler
pool seed `20261107` instead of `20260923`, and `AUDIT_FAMILIES` instead of
`DEVELOPMENT_FAMILIES` (held out here: `audit_polished_steel`, `audit_rubber`).
Nothing else changes — not the physics, not the arms, not the objective, not the
hyperparameters, not the checkpoint rule.

That is deliberate, and it bounds what a disagreement here could mean. A result
that fails under audit cannot be explained as a changed simulator, because the
simulator is the same code path; it can only be a result that did not generalize
to a new draw of fillers and appearance families.

The freeze matters in the other direction too. Every setting was fixed **before**
this run, and none was chosen against any endpoint value on either world. Where
the two worlds disagree both numbers are reported — §3 — because suppressing the
development number in favour of the sealed one would turn the seal into a
selection device, the opposite of what ADR 0006 built it for.

**One supersession is recorded.** The first audit run was started before a
correction to the transplant endpoint's control (ADR 0008 §5: a within-cell
derangement cannot differ from its subject), and was abandoned rather than
reported. The run below is the one made on the final code. The `CFError` contrast
was unaffected by that correction — only the alignment statistic's null was — but
a run made against a protocol that then changed is reported as superseded, never
quietly repeated.

---

## 2. The factorial on `audit_world`

### Training

| arm | parameters | val loss | best epoch |
|---|---|---|---|
| `P-G` | 23880 | 0.086925 | 17 |
| `P-R` | 24008 | 0.087716 | 17 |
| `S-G` | 31692 | 0.743421 | 38 |
| `S-R` | 31820 | 0.758304 | 32 |
| `Shuffle-R` | 24008 | 0.085884 | 36 |
| `StaticSSL-G` | 23112 | 0.085875 | 19 |
| `StaticSSL-R` | 23240 | 0.087123 | 27 |

Parameter counts are identical to the development world's, as they must be: the
world changes the sample, not the architecture.

### CCGP on unseen appearance families

| arm | accuracy | baseline | above baseline |
|---|---|---|---|
| `P-G` | +0.5820 | +0.2727 | +0.3093 |
| `P-R` | +0.5631 | +0.2727 | +0.2904 |
| `S-G` | +0.6472 | +0.2727 | +0.3745 |
| `S-R` | +0.6096 | +0.2727 | +0.3369 |
| `Shuffle-R` | +0.5482 | +0.2727 | +0.2755 |
| `StaticSSL-G` | +0.5468 | +0.2727 | +0.2741 |
| `StaticSSL-R` | +0.5334 | +0.2727 | +0.2607 |

### CCGP by intervention regime

| arm | passive | weak | rich |
|---|---|---|---|
| `P-G` | +0.3233 | +0.3275 | +0.2992 |
| `P-R` | +0.3498 | +0.3267 | +0.2712 |
| `S-G` | +0.3881 | +0.3903 | +0.3487 |
| `S-R` | +0.3744 | +0.3598 | +0.3038 |
| `Shuffle-R` | +0.3415 | +0.3119 | +0.2667 |
| `StaticSSL-G` | +0.3206 | +0.3030 | +0.2402 |
| `StaticSSL-R` | +0.3386 | +0.2820 | +0.2504 |

### Role equivariance on unseen families

| arm | E_role | E_shuffled | gain | E_inv |
|---|---|---|---|---|
| `P-G` | +0.0454 | +0.2461 | +0.2007 | +0.2163 |
| `P-R` | +0.2085 | +0.3192 | +0.1106 | +0.2178 |
| `S-G` | +0.2179 | +1.1103 | +0.8924 | +0.2509 |
| `S-R` | +0.4263 | +1.2307 | +0.8044 | +0.2229 |
| `Shuffle-R` | +0.2377 | +0.3258 | +0.0881 | +0.2246 |
| `StaticSSL-G` | +0.0499 | +0.1712 | +0.1213 | +0.2364 |
| `StaticSSL-R` | +0.2548 | +0.3322 | +0.0774 | +0.2237 |

### Label efficiency (CCGP above baseline, labels per class)

| arm | 1 | 5 | 10 | 50 | None |
|---|---|---|---|---|---|
| `P-G` | -0.0081 | +0.0599 | +0.1706 | +0.2667 | +0.3093 |
| `P-R` | -0.0124 | +0.0948 | +0.1740 | +0.2674 | +0.2904 |
| `S-G` | -0.0470 | +0.1509 | +0.2723 | +0.3122 | +0.3745 |
| `S-R` | -0.0481 | +0.1258 | +0.2212 | +0.3059 | +0.3369 |
| `Shuffle-R` | -0.0304 | +0.1096 | +0.2215 | +0.2597 | +0.2755 |
| `StaticSSL-G` | -0.0058 | +0.0816 | +0.1920 | +0.2467 | +0.2741 |
| `StaticSSL-R` | +0.0075 | +0.0664 | +0.1391 | +0.2352 | +0.2607 |

### Code ablation (one-step loss with and without the bottleneck)

| arm | with code | without code | relative increase |
|---|---|---|---|
| `P-G` | +0.0228 | +0.1400 | +5.1396 |
| `P-R` | +0.0229 | +0.1356 | +4.9123 |
| `S-G` | +0.0266 | +0.1789 | +5.7164 |
| `S-R` | +0.0260 | +0.1767 | +5.8001 |
| `Shuffle-R` | +0.0219 | +0.1265 | +4.7829 |
| `StaticSSL-G` | +0.0229 | +0.0767 | +2.3443 |
| `StaticSSL-R` | +0.0245 | +0.1941 | +6.9139 |

### Transplant

| arm | CFError self | correct | wrong | gain | CI | positive |
|---|---|---|---|---|---|---|
| `P-G` | +0.8980 | +0.9373 | +0.9373 | -0.0000 | [-0.0000, -0.0000] | False |
| `P-R` | +0.9959 | +1.0548 | +1.0548 | +0.0000 | [-0.0000, +0.0000] | False |
| `S-G` | +0.9289 | +0.9343 | +0.9343 | -0.0000 | [-0.0000, +0.0000] | False |
| `S-R` | +1.0146 | +1.0570 | +1.0570 | -0.0000 | [-0.0000, +0.0000] | False |
| `Shuffle-R` | +0.9341 | +1.0392 | +1.0392 | -0.0000 | [-0.0000, +0.0000] | False |
| `StaticSSL-G` | +0.9642 | +1.0182 | +1.0182 | +0.0000 | [-0.0000, +0.0000] | False |
| `StaticSSL-R` | +0.9545 | +0.9990 | +0.9990 | -0.0000 | [-0.0000, +0.0000] | False |

No scorable alignment pairs in any arm: **4048 of 4320 pairs excluded because the
code did not move the decoded prediction, 272 because the law was inactive** —
the same counts as the development world, which is expected, since both are
properties of the design rather than of the sample. Nothing is read from the
alignment statistic, for the reason given in
[03 §6](03_phase1b_factorial.md) (its control cannot differ from its subject).
**The oracle must-fire control was not re-run on this world**; it is a property of
the harness, which is identical here, and the development-world value
(`CFError` gain +0.1856 with no unscorable pairs) is the one that bounds this
null. That is a gap, and it is recorded as one in §4.

---

## 3. Does the development result survive the seal?

| arm | CCGP dev | CCGP audit | signed gap | role gain dev | role gain audit |
|---|---|---|---|---|---|
| `P-G` | +0.2791 | +0.3093 | +0.0302 | +0.2010 | +0.2007 |
| `P-R` | +0.2876 | +0.2904 | +0.0028 | +0.1280 | +0.1106 |
| `S-G` | +0.3494 | +0.3745 | +0.0251 | +0.9467 | +0.8924 |
| `S-R` | +0.3448 | +0.3369 | -0.0080 | +0.5397 | +0.8044 |
| `Shuffle-R` | +0.2876 | +0.2755 | -0.0121 | +0.0927 | +0.0881 |
| `StaticSSL-G` | +0.2869 | +0.2741 | -0.0128 | +0.1359 | +0.1213 |
| `StaticSSL-R` | +0.2949 | +0.2607 | -0.0343 | +0.0749 | +0.0774 |

### What reproduces

- **Endpoint A holds.** Every arm clears the 0.2727 baseline on families it never
  saw, from +0.2607 to +0.3745. *Relation coding generalizes across conditions*
  is the claim the phase may make, and it survives the seal.
- **The intervention-richness inversion reproduces, in every arm.** `passive`
  is the best regime and `rich` the worst, exactly as on the development world
  and exactly opposite to Phase IA's `J_ab` of 0.020 passive / 4.069 rich.
  Seven arms on two independent draws is not a fluke of the sample.
- **The role-equivariance reversal reproduces.** `P-G` (+0.2007) is again well
  above `P-R` (+0.1106), and `StaticSSL-G` (+0.1213) above `StaticSSL-R`
  (+0.0774). The pairwise bottleneck costs role structure, twice.
- **`E_inv` stays far from zero**, at 0.216–0.251 — the plan's §15 involution
  expectation fails on both worlds.
- **Endpoint C is null on both worlds**, with identical exclusion counts.

### What does not reproduce

**The bottleneck's CCGP advantage flips sign.** On the development world `P-R`
was 0.0085 above `P-G`; here it is **0.0189 below** — `P-G` is the better arm.
The direction reverses across worlds, and the swing (0.0274) is larger than the
across-seed spread of `P-R` measured at a smaller sample (0.0126,
[03 §8](03_phase1b_factorial.md)).

That comparison is a yardstick, not a test: the seed spread was measured on the
development world at 12 contexts per tuple, so it bounds one kind of variation and
this is another. What it does establish is that the swing is not obviously within
run-to-run noise.

This matters more than its size. The development world's `P-R > P-G` was the
*only* evidence pointing the bottleneck hypothesis's way, and it was already too
small to resolve the contrast. Across two worlds it has no consistent sign, and
the honest reading is that **the phase has no evidence for H2 in either
direction** on this endpoint.

**`S-R`'s role gain jumps.** +0.5397 to +0.8044 — a 49% increase from a change of
appearance families alone, on an arm whose training data does contain the relation
label. No unsupervised arm moves by more than 0.0174 on the same measurement
(`P-R`, 0.1280 to 0.1106), and the other supervised arm moves by 0.0543. This is a
stability property of the supervised reference rather than of the phase's
question, but it is visible and is reported rather than dropped.

### What the seal changes about the claim

Nothing in the claim ladder. The ADR 0007 ladder is indexed by *which endpoints
pass*, and the audit reproduces the pattern that determined it:

- Endpoint A passes on both worlds → *relation coding generalizes across
  conditions* is permitted, and is the phase's claim.
- Endpoint B passes in the weak sense on both worlds (a shared `T_swap` exists)
  and fails its involution expectation on both.
- Endpoint C fails on both → no causal claim on the relation variable, and
  **"abstract relational representation emerged" is not permitted**.

The audit adds one thing the development world could not: the phase's *negative*
result is not a property of one draw of appearance families. Five arms that differ
in what they can see reaching the same cross-family number holds on both.

---

## 4. What is not claimed

- **Not a reproduction**, and not a second independent experiment: the same
  generator, the same design, the same code. The seal changes the sample, and a
  result that holds across it is a result about the physics rather than about the
  appearance families of one draw.
- **No emergence claim** beyond the ADR 0007 ladder, applied in §3.
- **A disagreement between the two worlds is a finding about stability**, not a
  reason to prefer one number. Both are reported, including the one that flipped
  against the hypothesis.
- **The oracle must-fire control was run on the development world only.** It is a
  property of the harness, which is identical here, so the control's value carries
  over — but this report cites a number measured elsewhere, and says so rather
  than leaving the reader to assume it was measured here.
- **The alignment statistic is confounded on this world too**, and nothing is read
  from it.
- **One seed per world**, as the frozen protocol specifies. The across-world
  comparison in §3 is therefore two points per arm, not a distribution.
