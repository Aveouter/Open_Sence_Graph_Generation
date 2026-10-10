# RelateAnything Paper-Code Critical Audit

Tracker: [issue #157](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/157).
Audit date 2026-10-10. Author: audit agent, on behalf of Xyu Liu.

This document records an independent, evidence-driven audit of RelateAnything
and the falsification tests it motivated ([#155](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/155),
[#156](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/156)).
It supersedes the `PAPER_FULLTEXT_NOT_VERIFIED` status recorded in #157: the
paper was retrieved and read in full for this audit.

It is **not** a reproduction report. No checkpoint was run, no benchmark was
scored, and no metric appears below. Nothing here changes the
`DEFERRED_NOT_REPRODUCED; OFFICIAL_PACK_UNAVAILABLE` status in
[status.md](status.md).

## 1. Sources and evidence classes

Every claim below is tagged with how it was established. The classes are kept
separate because they carry different weight, and only the first three were
available to this audit.

| Tag | Meaning | Available |
|---|---|---|
| `PAPER` | Stated in the published paper, with a section/equation/table pointer | Yes |
| `SOURCE` | Confirmed by reading the pinned official source | Yes |
| `EXEC` | Produced by executing the pinned source on synthetic fixtures | Yes |
| `CKPT` | Observed on a released frozen checkpoint | **No** — no checkpoint installed |
| `MODEL-FAIL` | Independently validated degradation of predictive quality | **No** — needs adjudicated truth and a full-split run |

### Retrieved artifacts

| Artifact | Identity | Retrieved |
|---|---|---|
| Paper | arXiv:2609.12552v1, "RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs", Maëlic Neau, submitted 2026-09-11, 73 pp. | PDF, sha256 `eb84290c483aeef87f34be061dceb57ce2b45e6d3dda9977cef5016126e59ee4` |
| Official source | `Maelic/RelateAnything` @ `06766fdf56752ca535fc9b971fca99ce563676d0` | Resolved as `refs/heads/main` HEAD on 2026-10-10 |
| Local port | `src/modules/relateanything/` at repo `15dee3e` | Present |

The paper's HTML rendering truncates before Appendices E-J; the PDF was
downloaded and converted to text so that §10, §11 and Appendices A-H could be
read. Section, table and figure numbers below refer to the v1 PDF.

### Port provenance, verified rather than assumed

The port was checked file-by-file against the pinned upstream tree:

| Result | Count |
|---|---|
| Upstream `relsgg/**/*.py` modules | 34 |
| Ported | 25 |
| Byte-identical to upstream | 24 |
| Differing | 1 — `eval/evaluator.py`, 5 changed lines, all import-path rewrites (`from relsgg.scoring import ScoreContract` -> `from ..scoring import ScoreContract`) plus a one-line comment |
| Upstream modules **not** ported | 9 — `data/multipack.py`, `data/multiscale.py`, `eval/haystack.py`, `eval/spatialsense.py`, `paths.py`, `training/embeddings.py`, `training/engine.py`, `training/monitor.py`, `training/setup.py` |
| Port-only | 1 — `provenance.py` |

Consequence: **metric semantics are byte-identical to upstream**, so the A1
evaluator can be audited locally. The training loop (`training/engine.py`) and
two evaluation axes (`eval/haystack.py` = A5, `eval/spatialsense.py` = A6) are
*not* ported, so loss-weight assembly and those axes could not be executed
here. Hashes of the audited files:

```
model/sampler.py      1499b6edaed0ff46e898694978b4a3377b632e47820b3a31ca7e04e83bc0060b
model/relsgg.py       6058674791628bc78e58d65cddd52bb49c42ceddd2d61e2aa060d267711a89b1
training/losses.py    52a1907531a66f1f42171015122bd156b633d618fb5418eec4186cf1c2d7d85b
model/transformer.py  560db7506854aa9f8decdb691379b5b05ac9e8e07b4b9908a4906bbef9bb1de4
model/backbone.py     6d4d2b7283522838a85f40847db288978efe1aae6d6b9fd4bd3e6e1daa17c6e6
eval/evaluator.py     fa8a081b5522627dee0d3878d1a350191be264b89009dcbb045e190e26b3e144
```

## 2. Component audit

Each row: the paper's contract, the code that implements it, what the contract
assumes, what was actually checked, and a verdict. `Verdict` uses the
vocabulary #157 asks for; a verdict of *Supported* means "the code does what
the paper says", not "the model is good".

### 2.1 Backbone feature extraction and multi-layer fusion

- **Claim** `PAPER` §A.1, Tab. 14: DINOv3 ViT-S/16+ (LVD-1689M), fully
  fine-tuned, hidden states tapped at relative depths `-6, -3, -1`,
  layer-normalised, combined by a softmax over three learned scalars.
- **Implementation** `SOURCE` [backbone.py:16](../../../src/modules/relateanything/model/backbone.py#L16)
  (`LAYER_OFFSETS = [-6, -3, -1]`), [backbone.py:64-77](../../../src/modules/relateanything/model/backbone.py#L64-L77)
  (`extract`: `F.layer_norm` per tap, `softmax(layer_weights)`, weighted sum).
- **Assumptions** Hidden states are indexed relative to the *last* entry of
  `hidden_states`, and patch tokens are the trailing `h*w`.
- **Verification** `SOURCE`: offsets resolve as `total + off`; taps slice
  `[:, -n_patch:, :]`, which is correct only because CLS/register tokens lead
  the sequence. Both match the paper's description.
- **Limitation** The paper attributes "layer-normalised feature taps" as
  "free, and equivalent to a multi-level read costing 3x as much" `PAPER`
  §H.2. That is an ablation claim; not re-executed here.
- **Verdict** Supported.

### 2.2 SoftSpatialPool, object / union / contact features

- **Claim** `PAPER` §3.2, Fig. 2: one feature per subject box, object box,
  union box and contact zone; the contact zone is the intersection when boxes
  overlap and the gap between facing edges when they do not.
- **Implementation** `SOURCE` [relsgg.py:315-331](../../../src/modules/relateanything/model/relsgg.py#L315-L331):
  `contact_boxes_k` built from `max`/`min` of the two xyxy boxes; union and
  contact pooled in one call into `spatial_pool`.
- **Assumptions** Optional `cov` rasters refine pooling; without them a box is
  its rectangle.
- **Verification** `SOURCE`: the overlap/non-overlap branch is expressed purely
  through `max`/`min`, so both cases are the same expression; the contact half
  receives a flat coverage raster (comment: "log(1) = 0, no bias").
- **Limitation** On a box-trained checkpoint the mask coverage weight is near
  zero, so mask input buys little. Already recorded in
  [ADR 0012](../../adr/0012-relateanything-region-input-contract.md); not a
  new finding.
- **Verdict** Supported.

### 2.3 Pair candidate generation and sampling

- **Claim** `PAPER` §3.2 verbatim: "A two-stage sampler retains 400 of the
  N(N-1) ordered pairs by geometric plausibility and then 128 by a learned
  relatedness score. It retains 99.79% of annotated positives; exhaustive
  scoring costs 1.02x and yields no measurable gain." Tab. 14 confirms
  "pair budget 400 by geometry | 128 learned".
- **Implementation** `SOURCE` [sampler.py:27-48](../../../src/modules/relateanything/model/sampler.py#L27-L48),
  [sampler.py:95-186](../../../src/modules/relateanything/model/sampler.py#L95-L186);
  config defaults `geo_budget=400`, `final_budget=128`
  ([config.py:27-31](../../../src/modules/relateanything/config.py#L27-L31))
  match Tab. 14 exactly.
- **Assumptions** The 99.79% figure is an author-reported property of the
  trained sampler on the authors' corpus.
- **Verification** `SOURCE`: the two-stage top-K, the unconditional force-include
  of annotated and swapped pairs during training, and the inference-time
  absence of that force-include are all present and consistent with the text.
  The 99.79% figure was **not** reproduced — that needs the corpus.
- **Limitation** `PAPER` §D.5 concedes the deployed pair budget "covers 43%"
  of candidate pairs, and that scoring all of them adds 2.1 Base+Novel and 3.7
  Novel R@50. The 99.79% claim is about *annotated positives surviving
  training-time sampling*, not about deployed coverage. The two are different
  estimands and must not be conflated.
- **Verdict** Supported (code); the coverage number is author-reported and
  unverified here.

### 2.4 RelationTransformer and pair-context interaction

- **Claim** `PAPER` §3.2, §7.4: pairs are refined through pair-to-pair
  context; "Pair context accounts for nearly all the remainder, 87-93% of the
  variance" of the semantic logit.
- **Implementation** `SOURCE` [transformer.py:51-119](../../../src/modules/relateanything/model/transformer.py#L51-L119)
  (`n_self` encoder layers over pairs, `n_cross` decoder layers, then a
  hand-rolled `CrossAttentionLayer`), [backbone.py:80-128](../../../src/modules/relateanything/model/backbone.py#L80-L128)
  (`RelationInteractionBlock`: `n_dep` pair self-attention layers then `n_gnd`
  joint query+scene layers).
- **Assumptions** Padding slots must be inert, or a prediction would depend on
  the batch's padding width.
- **Verification** `EXEC` — see §3.2. Pair-set context was measured, its
  invariants were tested, and the routes were isolated by mutation.
- **Limitation** Context dependence is intentional and large; the audit found
  **three** independent routes (see §3.2), so any attempt to ablate or
  stabilise context must address all three.
- **Verdict** Supported; context dependence is expected behaviour, not a defect.

### 2.5 Deformable relation readout

- **Claim** `PAPER` §A.2, Tab. 14: "4 points, 8 heads, 2 null slots";
  offsets in units of the anchor half-extent, clamped to the image; null slots
  let a pair attenuate the read.
- **Implementation** `SOURCE` [deformable.py:29-116](../../../src/modules/relateanything/model/deformable.py#L29-L116);
  config defaults `deformable_points=4`, `deformable_heads=8`,
  `deformable_nulls=2` match Tab. 14.
- **Assumptions** Each pair reads independently — no cross-pair path.
- **Verification** `EXEC`: adding slots leaves existing slots' reads unchanged
  to `1e-5` (test `test_deformable_read_is_per_slot`). Geometric position
  clamping to `[0, 1]` present.
- **Limitation** None found.
- **Verdict** Supported.

### 2.6 Semantic / spatial query construction

- **Claim** `PAPER` §3.2, Eq. 2: two query experts mixed per predicate by a
  learned gate `alpha_p` read off the predicate's text embedding.
- **Implementation** `SOURCE` [relsgg.py:354-360](../../../src/modules/relateanything/model/relsgg.py#L354-L360)
  (semantic query = gated sum of pair context, subject-text and object-text
  projections; spatial query from pair context + geometry),
  [vocab_head.py:118-129](../../../src/modules/relateanything/model/vocab_head.py#L118-L129)
  (`(1-alpha)*cos_sem + alpha*cos_spa`).
- **Assumptions** The gate must route *unseen* predicates too, so it is
  computed from the text embedding rather than learned per class.
- **Verification** `SOURCE`: `current_alpha()` computes live from `W` while
  training and from a baked buffer after `reparameterize()`; no text encoder
  runs per frame, matching the paper's claim that inference carries only the
  embedding bank.
- **Limitation** §7.4 reports the optional direct subject/object path is gated
  by a scalar driven to 0.014-0.044, contributing 0.1% / 0.0% of logit
  variance. `SOURCE` confirms the gate (`compose_gate`, initialised 0.1) is a
  free parameter, so the paper's "almost nothing" is consistent with the code.
- **Verdict** Supported.

### 2.7 Vocabulary embedding and predicate scoring

- **Claim** `PAPER` §3.3, §B.2: the vocabulary is a bank of text embeddings,
  not a learned classifier; the teacher is distilled into a 512-d student with
  an antonym-repulsion hinge; mean inverse cosine 0.92 -> 0.09, AUC 0.85 ->
  0.99.
- **Implementation** `SOURCE` [vocab_head.py:66-75](../../../src/modules/relateanything/model/vocab_head.py#L66-L75)
  (`set_vocabulary_matrix` L2-normalises and installs `W`),
  [vocab_head.py:77-85](../../../src/modules/relateanything/model/vocab_head.py#L77-L85)
  (`reparameterize` seals it).
- **Assumptions** `W` rows are L2-normalised; scoring is a scaled cosine.
- **Verification** `SOURCE`: `score_query_dual` clamps `logit_scale.exp()` at
  100 and normalises both queries, so the head is a pure cosine bank.
- **Limitation** `W` and `alpha` are co-determined (the gate reads `W`), so
  swapping the vocabulary moves the routing too. That is by design, but it
  means "swap the vocabulary" is not a matrix swap with fixed routing.
- **Verdict** Supported, with the routing caveat recorded.

### 2.8 Training losses, positive supervision and negative handling

- **Claim** `PAPER` §3.3, Eq. 3, §B.1, Tab. 14: primary batch-local InfoNCE
  with synonym-group positives; four auxiliary terms (per-cell sigmoid, swap
  hinge, background suppression, object-text grounding); PU negative discount
  `log(1 - P(q|p))`; annotated predicates never down-weighted and directional
  inverses always full weight.
- **Implementation** `SOURCE` [losses.py:185-287](../../../src/modules/relateanything/training/losses.py#L185-L287)
  (`BatchLocalInfoNCE`), [losses.py:143-178](../../../src/modules/relateanything/training/losses.py#L143-L178)
  (`swap_direction_hinge`), [losses.py:123-140](../../../src/modules/relateanything/training/losses.py#L123-L140)
  (`build_slot_targets`).
- **Assumptions** Multi-positive slots combine per-predicate discounts as if
  independent (`PAPER` §3.3 verbatim: "a pair carrying several annotations
  combines their discounts as if independent").
- **Verification** `SOURCE` for the discount accumulation
  ([losses.py:272](../../../src/modules/relateanything/training/losses.py#L272),
  `(hot @ self.neg_lw[:, S])`, then `masked_fill(pos_any | inv_any, 0.0)` which
  implements both exemptions exactly as stated). `EXEC` for the multi-hot
  target ([test_relateanything_cfa_order.py](https://github.com/Aveouter/Open_Sence_Graph_Generation/blob/main/tests/reproduction/test_relateanything_cfa_order.py)
  `test_slot_targets_keep_every_positive_under_permutation`): every positive is
  preserved and the result is invariant to relation-row order.
- **Limitation** The five loss *weights* of Tab. 14 (InfoNCE 0.5, sigmoid 0.25,
  swap 0.5, background 0.05, grounding 0.1) are assembled in
  `training/engine.py`, which is **not ported**. The config does carry
  `lambda_swap=0.5`, `lambda_sigmoid=0.25`, `lambda_bg=0.05`, `lambda_obj=0.1`
  ([config.py:53-62](../../../src/modules/relateanything/config.py#L53-L62)),
  consistent with Tab. 14 for those four, but the InfoNCE weight of 0.5 and the
  final combination could not be verified locally.
- **Verdict** Supported for the objective; **loss-weight assembly unverified**
  (port gap, recorded as a precise blocker).

### 2.9 CFA augmentation and multi-positive relation handling

- **Claim** `PAPER` §H.2, quoted verbatim as the authors state it. This is an
  author-reported attribution, **not** an independently verified result — no
  training run was performed in this audit: "Same-predicate feature mixing:
  +10.1% zero-shot tail mean recall, reproduced with two seeds, whereas the
  identical mechanism with random partners yields -2.8%. **The
  class-conditional matching rather than the mixing is the mechanism**; both
  mixing arms have a lower training loss than the control, which rules out an
  augmentation effect and points to **prototype smoothing**."
- **Implementation** `SOURCE` [relsgg.py:174-216](../../../src/modules/relateanything/model/relsgg.py#L174-L216)
  (`_mix_partners` groups slots by a single predicate id; `_mix_entities`
  blends subject/object features through a `Beta(alpha, alpha)` coefficient),
  driven by `pred_labels` from the sampler, and applied at
  [relsgg.py:333-335](../../../src/modules/relateanything/model/relsgg.py#L333-L335).
- **Assumptions** The mix is "Label-preserving, so no loss change is needed"
  (the docstring's words), and the paper's justification is prototype
  smoothing under class-conditional matching.
- **Verification** `EXEC` — full falsification in §3.1.
- **Limitation** Which predicate a multi-positive slot is matched on is set by
  `scatter_` (last row wins), so the partner graph is a function of annotation
  *row order*, not of the label set. Confirmed and quantified in §3.1.
- **Verdict** **CONFIRMED_TRAINING_PROTOCOL_SENSITIVITY.** The paper's stated
  mechanism (class-conditional matching for prototype smoothing) is *not*
  violated — every selected partner genuinely shares the selected predicate —
  so this is a protocol sensitivity, not a paper contradiction. The dependence
  reaches the forward pass and the loss and gradients of the optimisation step
  (§3.1); **no model-performance consequence was demonstrated or tested**. See
  §3.1 for the one narrow internal-contract issue.

### 2.10 Graph decoding and calibration

- **Claim** `PAPER` §D.1, §D.4, docs/pitfalls: one predicate per ordered pair
  under the graph constraint; the score is a calibrated
  `sigmoid(pred + w * pair)`; calibration is monotone and therefore cannot move
  a ranking.
- **Implementation** `SOURCE` [scoring.py](../../../src/modules/relateanything/scoring.py)
  (`ScoreContract.scores`, `graph_constrained`), byte-identical to upstream.
- **Assumptions** Eval and deployment must use the *same* score function, not
  a product of two sigmoids.
- **Verification** `SOURCE` + existing repo test
  `test_pair_logits_contribute_before_the_shared_calibrated_sigmoid`
  ([test_relateanything_port.py](https://github.com/Aveouter/Open_Sence_Graph_Generation/blob/main/tests/reproduction/test_relateanything_port.py)).
- **Limitation** Calibration is monotone, so it cannot change ranking — but
  replacing the shared sigmoid with a product of sigmoids does. Already
  recorded upstream (`docs/pitfalls.md`) and in
  [implementation_audit.md](implementation_audit.md) §"Protocol details".
- **Verdict** Supported.

### 2.11 Training / inference differences

- **Claim** `PAPER` §3.1: object labels are never an input; the vocabulary is
  supplied at inference.
- **Implementation** `SOURCE` [relsgg.py:222-243](../../../src/modules/relateanything/model/relsgg.py#L222-L243)
  (no label parameter in `forward`); labels enter only through
  `_object_text_loss` ([relsgg.py:148-172](../../../src/modules/relateanything/model/relsgg.py#L148-L172))
  and the sampler's PU weighting
  ([sampler.py:62-72](../../../src/modules/relateanything/model/sampler.py#L62-L72)).
- **Assumptions** "No label input" must not be read as "no label-dependent
  training signal".
- **Verification** `SOURCE` for the training-only paths; `EXEC` for the
  training/inference split of the sampler's force-include
  ([sampler.py:123-125](../../../src/modules/relateanything/model/sampler.py#L123-L125),
  gated on `training`, absent at inference).
- **Limitation** Training *does* consume `entity_labels` for auxiliary
  supervision; the weaker claim "no label-dependent training signal" is false.
  Already recorded in #157's original cross-walk.
- **Verdict** Supported for the inference contract; the distinction is
  preserved and must not be overstated in either direction.

### 2.12 Official evaluation implementations and existing tests

- **Claim** `PAPER` §5, §D: A1-A6 axes; A1 R@K averages image recalls, mR@K
  averages per-image class recalls within each observed predicate.
- **Implementation** `SOURCE` `src/modules/relateanything/eval/evaluator.py`
  (differs from upstream by 5 import-path lines only).
- **Assumptions** The class name `SGClsEvaluator` does not make this standard
  OpenSGG SGCls or PredCls.
- **Verification** `SOURCE` + existing repo tests
  `test_a1_evaluator_uses_image_recall_and_observed_class_mean`.
  A5 (`eval/haystack.py`) and A6 (`eval/spatialsense.py`) are **not ported**
  and were not audited.
- **Limitation** Two of six axes are outside the port. The full metric
  semantics audit is therefore partial by construction.
- **Verdict** Supported for A1-A4; **A5/A6 not auditable locally**.

## 3. Executed falsification tests

Both suites live in `tests/reproduction/`, run CPU-only on synthetic fixtures
with fixed seeds, cost ~0.1 s each, and are skippable when torch is absent.
They are characterisation tests of *source behaviour*: they measure no model
quality and produce no baseline evidence.

Environment: Python 3.10.8, torch 2.5.1+cu121, numpy 1.23.5, CPU only,
`CUDA_VISIBLE_DEVICES` empty. Commands:

```
python -m unittest tests.reproduction.test_relateanything_cfa_order
python -m unittest tests.reproduction.test_relateanything_pair_context
```

### 3.1 #155 — CFA partner selection is annotation-order dependent

**Fixture.** One multi-positive ordered pair `(0,1)` carrying `{r1, r2}`, a
pair `(2,3)` carrying `{r1}`, and a pair `(0,2)` carrying `{r2}`. Two
annotation tables that are permutations of one another, so the labelled set is
identical and only row order differs.

**Result.** The hypothesis is **CONFIRMED at the code, forward-pass and
optimisation-step level**:

| Observation | Order A | Order B |
|---|---|---|
| `_gt_grid` predicate for `(0,1)` | `r1` | `r2` |
| CFA partner-eligible groups | `{r1: [(0,1), (2,3)]}` | `{r2: [(0,1), (0,2)]}` |
| Partner for `(0,1)` | `(2,3)` | `(0,2)` |

Both orders mix exactly one slot, so the augmentation *budget* is unchanged;
only the *content* changes. The difference reaches the model: with
`cfa_prob=1.0`, `max|delta logits| = 1.38e-01` and
`max|delta pair_features| = 4.43e-01`.

It also reaches the **training signal**, which is the strongest statement this
audit supports (plan.md §4 items 5 and 6):

| `cfa_prob` | `loss` order A / B | `abs(delta loss)` | `max(abs(delta grad))` | Parameters with a changed grad norm |
|---|---|---|---|---|
| `0.0` (control) | 2.23927760 / 2.23927760 | `0.0` | `0.0` | 0 of 134 |
| `1.0` | 2.01421618 / 2.22334290 | `2.09e-01` | `1.95e+00` (87% of the largest gradient) | 103 of 134 |

The moved parameters are the ones `_mix_entities` feeds — `pair_proj`,
`sub_text_proj` / `obj_text_proj`, `compose_norm`, the pair encoder — which is
where the mixing enters the graph.

**Controls, all passing.**

1. *CFA disabled* (`cfa_prob=0.0`): the two orders agree **bit-exactly**
   (`max|delta| = 0.0`) on the logits, the pair features, the **loss and every
   parameter gradient**. This is the decisive control — it proves no other part
   of the training step (sampler losses, slot-target lookup, loss reduction,
   backward order) is order sensitive, so the entire effect is attributable to
   CFA.
2. *Single-positive fixture* (`cfa_prob=1.0`): unchanged to `0.0`.
3. *`build_slot_targets`*: multi-hot targets are invariant to row order and
   keep **both** positives, exactly as `PAPER` §3.3 and the code claim.
4. *Released hyperparameters*: the deterministic cases pin `cfa_alpha` high so
   the mixing coefficient is fixed and the partner swap is the only variable.
   The shipped recipe is `cfa_prob=0.5, cfa_alpha=1.0`
   ([config.py:63-64](../../../src/modules/relateanything/config.py#L63-L64)),
   where the coefficient is drawn per step. Partner *assignment* does not
   depend on the coefficient, so the dependence must persist, and it does —
   checked over three seeds
   (`test_order_dependence_survives_the_released_cfa_hyperparameters`).

**Falsifiability.** The suites were mutation-tested. Replacing `_gt_grid` with
a first-wins (set-valued) variant fails the two characterisation cases;
disabling `_mix_entities` fails the forward-dependence and gradient cases. The
tests can fail, so their passing is evidence.

**Verdict — scoped deliberately narrowly.**

In scope, and confirmed: the CFA partner graph, the forward pass, and the loss
and gradients of one optimisation step are not functions of the labelled set
alone. `CONFIRMED_TRAINING_PROTOCOL_SENSITIVITY`.

Explicitly **out** of scope, and therefore *not* established either way:

- Whether either of the two gradients is *better*. Nothing here measures
  quality; a different gradient is not a worse gradient.
- Any effect at trained-checkpoint level. No training run, no checkpoint, no
  benchmark, no metric. This is **not** a model failure and must not be cited
  as one.
- Whether the effect survives to convergence, or is averaged out over an epoch
  (the two orders still mix the same number of slots per step, so a mean over
  batches could plausibly wash out — untested).

Related findings:

- *Not* a contradiction of the paper. `PAPER` §H.2 says the mechanism is
  "the class-conditional matching ... points to prototype smoothing", and
  every selected partner does genuinely share the selected predicate. The
  paper is silent on annotation order (no occurrence of annotation ordering
  anywhere in the text).
- One narrow internal-contract issue, stated precisely: the `_mix_entities`
  docstring claims "Label-preserving, so no loss change is needed". For a
  multi-positive slot that holds only with respect to the *last-scattered*
  predicate, not the full positive set. Mixing `(0,1)` toward `(0,2)` in
  order B is label-preserving for `r2` while the slot's InfoNCE target still
  contains `r1`. This is a documentation/contract defect with a measurable
  feature consequence; it is **not** shown to harm accuracy.
- Model-performance verdict: `INCONCLUSIVE`. Demonstrating degradation would
  need a matched-budget training comparison on the real corpus, which is
  outside this audit's authorisation and blocked by data availability.

**Smallest possible correction (GREEN, not applied).** Nothing is applied —
`AGENTS.md` and the plan require separate authorisation for model or training
changes. If a fix is ever wanted, the minimal one is to make the partner key
set-valued: match slots that share *any* positive predicate rather than one
scattered id, leaving the mixing arithmetic untouched. That is a change to the
augmentation contract, so it would need a matched-budget ablation against the
current behaviour before adoption, and #155's own GREEN gate forbids applying
it without approval. Given that the paper attributes the gain to
class-conditional matching and that the current code already satisfies that, a
fix is **not** indicated by the evidence.

### 3.2 #156 — pair-set context dependence

**Result.** Every invariant that must hold, holds; the designed dependence is
real and large.

| Case | Measured | Verdict |
|---|---|---|
| Permuting the pair set permutes the output (transformer) | `7.2e-07` | Equivariant |
| Permuting the pair set permutes the output (interaction block) | `3.0e-07` | Equivariant |
| Extra padded slots do not move real slots (transformer) | `4.8e-07` | Inert |
| Extra padded slots do not move real slots (interaction block) | `3.6e-07` | Inert |
| Batch items are independent | `0.0` exactly | Independent |
| Deformable read is per-slot | `< 1e-5` | No cross-pair path |
| Adding one irrelevant pair moves a retained pair (transformer) | `4.9e-01`, 13.6% relative | **Designed** |
| Adding one irrelevant pair moves a retained pair (interaction) | `2.7e-01`, 7.6% relative | **Designed** |

At model level, with `final_budget` raised so every ordered pair survives the
sampler, adding a fifth object leaves the target pair `(0,1)` **retained in the
graph** while still moving its logits. That separates "the candidate set
changed underneath it" from "its context changed", and answers the issue's
question directly: an already-selected pair's prediction does depend on
unrelated candidates.

**Route decomposition, by mutation.** This was the audit's most useful
unexpected result. Pair-set context reaches a query through **three
independent routes**, established by disabling them one at a time:

| Mutation | Context still present? |
|---|---|
| `n_self = 0` (drop the pair self-attention layers) | Yes |
| `n_self = 0` **and** box-corner tokens removed from the memory | Yes — the decoder layers' own target self-attention over the pair set is sufficient |
| Each slot processed alone (all cross-pair routes removed) | No — the dependency is detected as inert |

The third route is the shared cross-attention memory, which contains the four
box-corner tokens of *every* sampled pair. Consequence: any future attempt to
ablate or stabilise context must remove all three, or it will silently keep the
dependence.

**Verdict — scoped to the invariants that were actually tested.**

- `EXPECTED_BEHAVIOR` **for the code-level mechanism**. `PAPER` §7.4 attributes
  87-93% of semantic-logit variance to pair context; a large, masked,
  permutation-equivariant, batch-separated dependence is precisely the designed
  mechanism. On the tested fixtures the implementation is behaving as specified.
- The candidate **code defect** in those invariants is `NOT_REPRODUCED`: no
  invariance violation was found, and the masks that prevent illegitimate
  leakage (padding width, batch neighbours) all hold.
- **`NOT_TESTED` / `INCONCLUSIVE` for harmful context interference on the
  released model.** This is the claim the issue actually poses, and this audit
  does *not* refute it — it never tested it. A stub backbone, random small
  weights, synthetic feature maps and a widened candidate budget characterise
  structure; they say nothing about whether the trained model's context
  dependence is harmful. Adjudicating that needs relation truth unchanged
  under the intervention plus an independent cohort — the issue's own bar,
  which requires the benchmark and a checkpoint.

So the honest summary is narrower than "refuted": **the mechanism is designed
and the tested invariants hold; whether the designed dependence causes harm in
the released model is still open.**

## 4. What this audit did not establish

Stated plainly, because the guardrails in `AGENTS.md` exist for exactly this.
This audit's scope is **the pinned source under static inspection, plus
CPU-only synthetic fixtures**. It is not a complete audit of the paper, not a
complete audit of the upstream repository, and not an evaluation.

### 4.1 Coverage checklist — audited vs not

"Completed" is used nowhere in this document; the audited scope is stated
explicitly instead. Uncovered items are listed so they cannot be assumed:

| # | Area | Status |
|---|---|---|
| 1 | Backbone feature extraction and multi-layer fusion | Audited (source) |
| 2 | SoftSpatialPool, object / union / contact features | Audited (source) |
| 3 | Pair candidate generation and sampling | Audited (source); 99.79% figure unverified |
| 4 | RelationTransformer and pair-context interaction | Audited (source + exec) |
| 5 | Deformable relation readout | Audited (source + exec) |
| 6 | Semantic / spatial query construction | Audited (source) |
| 7 | Vocabulary embedding and predicate scoring | Audited (source) |
| 8 | Training losses, positive supervision, negative handling | Audited (source) — **but the five Tab. 14 loss weights are not** |
| 9 | CFA augmentation and multi-positive handling | Audited (source + exec) |
| 10 | Graph decoding and calibration | Audited (source) |
| 11 | Training / inference differences | Audited (source + exec) |
| 12a | Official evaluation — A1-A4 | Audited (source + existing tests) |
| 12b | Official evaluation — **A5** (`eval/haystack.py`) | **Not audited — module not ported** |
| 12c | Official evaluation — **A6** (`eval/spatialsense.py`) | **Not audited — module not ported** |
| 13 | Training loop and final loss assembly (`training/engine.py`) | **Not audited — module not ported** |
| 14 | Corpus construction, RA-4M verification gate, generation prompts (§C, §J) | **Not audited** |
| 15 | A2, A3, A4 numeric outcomes and the OvR-SGG leaderboard (§6) | **Not audited — author-reported only** |
| 16 | Appendix H ablations in full (Tab. 30 recipe ladder, Tab. 32) | **Read, not reproduced** |
| 17 | Appendices E, F, G, I (extra results, probes, backbone scaling, cost) | **Not read** |
| 18 | Any behaviour of the released checkpoint | **Not audited — no checkpoint installed** |
| 19 | Any metric on any split | **Not audited — no pack obtainable (#134)** |

### 4.2 Explicitly not established

- **No reproduction.** No checkpoint was loaded, no split was scored, no
  metric was computed. The baseline remains `DEFERRED_NOT_REPRODUCED;
  OFFICIAL_PACK_UNAVAILABLE` per #134 and
  [ADR 0014](../../adr/0014-relateanything-open-data-deferral.md).
- **No model failure.** Neither #155 nor #156 produced an independently
  validated degradation of predictive quality. Every quantitative result above
  is a synthetic-fixture source measurement.
- **No A5/A6 audit.** `eval/haystack.py` and `eval/spatialsense.py` are not
  ported.
- **No loss-weight verification.** Tab. 14's loss weights are assembled in the
  unported `training/engine.py`.
- **No author-reported number verified.** The 99.79% sampler coverage, the
  +10.1% CFA attribution, the 1.3% noise floor and the A1-A6 outcomes remain
  author-reported; re-measuring them needs the corpus and a training run.
- **Two axes of the pipeline are author-attested only.** §H.2's attributions
  rest on a 50k proxy and single-variable arms read against a 1.3% noise
  floor.

## 5. Verdict summary

| Candidate issue | Verdict | Model-performance verdict |
|---|---|---|
| #155 CFA annotation-order sensitivity | `CONFIRMED_TRAINING_PROTOCOL_SENSITIVITY` — partner graph, forward pass, **loss and gradients** all order-dependent (+ one narrow docstring/contract defect) | `INCONCLUSIVE` — no quality effect measured |
| #156 Pair-set context dependence | `EXPECTED_BEHAVIOR` for the mechanism; candidate code defect in the tested invariants `NOT_REPRODUCED`; harmful interference on the released model `NOT_TESTED` | `INCONCLUSIVE` — not refuted, not tested |
| #157 Paper-code audit | Scope stated, not "completed": paper retrieved and read (§1-§11, App. A-D, H); port parity verified; 12 components audited; 7 areas listed as not audited in §4.1 | — |

Neither candidate issue is a confirmed model defect, and neither is dismissed.
#155 records a genuine, reproducible protocol sensitivity that propagates to
the optimisation step, with a clean mechanism; #156 records that the context
path is stronger and more entangled than a single-layer reading suggests, while
satisfying every invariance a correct implementation must satisfy — and leaves
the harm question open rather than refuting it.

Both verdicts are about **source behaviour under synthetic fixtures**. Neither
is evidence about the released model's accuracy.

## 6. Reproducing this audit

```bash
# RED suites (CPU, skip cleanly without torch; 14 + 9 cases, ~0.3 s each)
python -m unittest tests.reproduction.test_relateanything_cfa_order -v
python -m unittest tests.reproduction.test_relateanything_pair_context -v

# Source-parity and provenance
python -c "from src.modules.relateanything.provenance import verify_source; print(verify_source())"

# Repository guardrails (see the note below on the two local-only failures)
python tools/reproduction/run_reproduction_guardrails.py
```

On this machine the aggregate guardrail script exits non-zero on **two
pre-existing, unrelated local failures**: the Application Control policy blocks
SciPy's `_interpnd` DLL (an *error* in `test_penet_adapter`), and the gitignored
local extraction has advanced one commit past its frozen manifest record
(`c312c54` vs `95397eb`, a *failure* in `test_check_repository_boundary`).
Both are documented in [implementation_audit.md](implementation_audit.md).
`check_reproduction_docs.py`, `check_reproduction_claims.py` and
`check_repository_boundary.py` each pass individually, and CI's `validate` job
runs the full suite green on ubuntu-latest, which confirms the failures are
environmental rather than defects. Full local run: 561 tests, those two failing.

Test-file hashes at the time of writing: `test_relateanything_cfa_order.py`
`5cca413e8689bfbd` (273 lines, 14 cases), `test_relateanything_pair_context.py`
`9167ca63119ac985` (408 lines, 9 cases), `_relateanything_audit_fixtures.py`
`1436fac2e0c15650` (204 lines).

### Falsifiability of the suites

A passing test is only evidence if it can fail. Four in-memory mutations were
applied to the pinned classes (nothing on disk was modified) and the suites
re-run:

| Mutation | Required outcome | Observed |
|---|---|---|
| `_gt_grid` made first-wins (set-valued) | #155 cases fail | Failed as required |
| `_mix_entities` made a no-op | #155 forward **and gradient** cases fail | Failed as required |
| Pair padding mask dropped in `RelationTransformer` | #156 padding case fails | Failed as required (`0.695` vs `1e-5` tolerance) |
| Every cross-pair route removed | #156 distractor case fails | Failed as required (context reads as inert) |

A fifth probe, removing only the pair self-attention layers, correctly did
**not** fail the distractor case — which is what exposed the additional context
routes reported in §3.2.
