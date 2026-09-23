# Phase I v2 implementation and development gate

Status: `implementation_audit`, `STOP_DATA`. This is not a completed Phase I v2
result or a reproduction. [Issue #107](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/107)
remains open. The fresh audit allocation is **SEALED**.

## Changes checked against #107

| Milestone | Implementation / evidence | Remaining exit condition |
|---|---|---|
| 0: preserve v1 | Existing code, reports and ADRs untouched; v1 report hashes recorded; fresh v2 seed/families reserved | Final v2 freeze is not yet justified |
| 1: predictive task | Real residual autoregression; early-action, intermediate-state, hand-transition and first-step tests | Measured horizon exceeds current windows |
| 2: state vs mechanism | Separate snapshot and history encoders; label-free inputs; cutoff/future leakage tests; active/exposed metadata | Confirm informative history under revised horizon |
| 3: transplant | Own-predictor history codes; within/cross-filler matched source twins; role alignment; disjoint context-dyad bootstrap; zero-sensitivity retained | Calibrate a learned decoder with true-M/alias controls on this world |
| 4: temporal controls | Named initial/post static position inputs; fresh per-episode permutations in train/eval | Chronology remains partly recoverable from frame content |
| 5: bottleneck | Four-object star world; no raw other-object bypass in P-R; global scene latent; active parameters matched within 1% | No general many-body or factorial conclusion |
| 6: M0 | Six feature sets, linear + trained MLP, grouped splits/bootstrap/permutations, maximum-statistic FWER correction and graded nonlinear controls | MDE-scale detection power fails |
| 7: IA v2 | Compatibility, numerical and self-divergence floors, curves/AUC/divergence/exposure and horizon measured anew | Horizon coverage and M0 fail; Level-2 not run |
| 8: evaluators | Known-positive/negative CCGP, role/involution/symmetry and transplant fixtures pass; full transplant path also regression-tested | Actual-world learned-decoder calibration pending |
| 9: factorial | Model and prediction-only checkpoint helpers available | Not run; prerequisites fail |
| 10: audit | Fresh allocation cannot be generated through development API | Not opened; no frozen protocol yet |

The two-body v1 world and its scientific interpretations are not silently
rewritten. The new world shares one externally driven supporter among three
probes; probes do not collide or exert backreaction. The term bilateral coupling
describes the displacement constraint, not reciprocal forces. This restriction
must accompany any future result from this candidate.

## Executed development validation

```powershell
.venv/hsg/python.exe -m tools.relational_emergence.v2.run --out outputs/analysis/relational_emergence/v2/development-validated
```

This ran 1,584 episodes from 144 independent base contexts: four objects, six
initial-state tuples, three intervention regimes, all compatible query laws,
24 groups per tuple. The candidate uses cutoff 10, prediction window 10 and total
horizon 20. No representation arm was fitted for scientific scoring. The only
representation optimization in verification is the two-epoch unit-test fixture.

| Check | Measured result |
|---|---:|
| Compatibility and matched initial state/actions | pass |
| Numerical floor, maximum | 4.5934e-13 |
| Numerical floor, mean | 1.2735e-16 |
| Self-divergence floor, p95 | 0.013892 |
| Rich-regime compatible cells above floor | all |
| Selected slowest 90%-peak horizon | 19 |
| Available history/prediction windows | 10 / 10: **fail** |
| Held-out chronology accuracy | 0.16134 |
| Chronology chance / group SE | 0.09091 / 0.00768 |
| Chronology control | `FAIL_RECOVERABLE_CHRONOLOGY` |
| Maximum observed M0 delta-CE | -2.5932e-9 nats |
| M0 calibration: weak / MDE / strong | 1/20, 1/20, 2/20 detections |
| M0 randomized-label negative control | 0/20 detections |
| Required MDE control power | 0.8; observed 0.05: **fail** |
| M0 verdict | `FAIL_AUDIT_POWER` |
| Synthetic evaluator fixtures | pass |
| Combined verdict | `STOP_DATA` |

M0 used 120 training epochs per estimator, hidden width 32 for the trained MLP,
199 compatible group-level permutations across the complete 12-probe family,
1,000 base-group bootstrap resamples and 20 calibration draws per level. The
nonlinear injected channels carry 0.025/0.05/0.20 nats of Bayes information. The
clean-data near-zero effect therefore **does not establish absence of leakage**:
the detector fails the declared resolution. Calibration rates are empirical
estimates; 20 draws do not provide a tight power confidence bound.

The chronology diagnostic is deliberately stronger than checking that random
permutation columns are uniform: it tries to recover original frame age from
held-out physical content. Its positive result blocks a claim that this control
has removed usable chronology. Uniform random permutation itself passes its
regression test. No permutation distribution or threshold was adjusted to turn
this failure into a success.

## Engineering evidence

Changing the first action in the v1 predictor changed the next three predictions
by exactly `[0.0, 0.0, 0.0]`. The v2 tests require all later steps to change, test
an injected intermediate-state perturbation, and compare every step with a
hand-constructed transition. P-R has 30,120 active parameters and P-G has 29,976
(0.48% difference), with identical depth, optimizer and prediction objective.

A regression also demonstrated why leakage arithmetic needs care: on balanced
twins with exactly identical initial observables, float32 accumulation reported
delta-CE `1.03e-8` with family-wise p=0.03. Such a population cannot beat its
compatibility entropy. Double-precision scoring and conservative numerical ties
remove the false positive. The test was observed failing before that repair and
passing afterward. The practical MDE remains 0.05 nats.

The earlier local development directories `development-issue107` and
`development-issue107-final` are superseded diagnostics: the first preceded
paired filler allocation, and the second exposed the float32 false positive.
They were not audit runs and were not overwritten. Only `development-validated`
supplies the numbers above; it was rerun after these fixes, and its source digest
matches the submitted Python tree.

Reviewable machine-readable evidence, all world curves, M0 results and v1 report
hashes are in [development_evidence.json](development_evidence.json). Raw runner
files stay under the gitignored output directory named in the command. The
manifest's `git_sha` is the checkout base during development; `code_sha256` binds
the actual source snapshot, including the then-uncommitted v2 modules:

```text
code:    0741ed0662542728c2fd98eb467fb0226c46cd141ced703326756756337ef3c0
dataset: 81347306171257f182dedd2bb839166eec823863ff5d62b5a77555f9d014ba37
```

## Next admissible work

Revise the development history/prediction horizon to cover the measured signal,
develop and power-calibrate the chronology control, and strengthen/sample-scale
M0 until it detects the declared nonlinear MDE without inflating false positives.
Revalidate those definitions, then run the multi-object Base/TrueM/ShuffledM
Level-2 and actual-world transplant controls. Only a successful gate can authorize
the complete seed factorial and a later fresh audit freeze. Do not close #107
or claim abstract relational emergence from this PR.
