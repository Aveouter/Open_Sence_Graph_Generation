# RelateAnything Implementation Audit

## Scope and status

Review of [issue #124](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/124),
dated 2026-10-09. Official source revision:
`06766fdf56752ca535fc9b971fca99ce563676d0`.
This section preserves the initial pre-port audit. Current integration evidence
is maintained in [status](status.md) and [checkpoint manifest](checkpoint_manifest.md).
At the initial audit, no weights were downloaded, no inference or model evaluation was run, and no
checkpoint compatibility verdict was established. This is a hand-authored
reproduction contract audit, not an experiment result.

## Assessment of the plan

The issue correctly separates official reference evaluation from stable
integration, forbids random-init substitutes, and requires provenance, task and
metric alignment. It should proceed through audit gates before integration.
Its reference command is a starting point rather than an executable acceptance
contract: the released-artifact layout and the pinned benchmark runner disagree
in concrete ways described below.

## Blocking gaps

1. **A1 needs explicit text-student resolution and strict-load evidence.**
   The release tool rewrites the checkpoint student path to `text_student.pt`.
   `eval_zeroshot.py` passes this string directly to the text encoder; it does
   not resolve it relative to the checkpoint directory as the public API does.
   Add `--text_student <snapshot>/text_student.pt` to the proposed A1 command.
   The runner calls `build_model_from_ckpt` with its default `strict=False`;
   therefore a completed benchmark is insufficient evidence of a clean load.
   Before metrics, verify the official strict API load and record all key
   exceptions, including the loader's deliberate obsolete-key removals and
   optional `W_obj` handling. This path analysis has not been tested against
   downloaded checkpoint contents. Sources: [release stripping](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/release/strip_checkpoint.py),
   [benchmark runner](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/eval_zeroshot.py),
   [checkpoint loader](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/checkpoint.py).

2. **A3 has a released-layout incompatibility, not simply missing weights.**
   The release tool scrubs `args.pred_embeds` and emits
   `predicate_embeddings.npz` with keys `names` and `W`. The runner's A3 path
   instead loads `ck_args['pred_embeds']` and expects `predicates` and
   `embeddings`. Its calibration identity check also depends on that path.
   An explicit student override alone cannot fix this. Require an official
   layout-compatible runner change or a documented, minimal reference patch
   preserving vocabulary order, embedding values and matcher semantics;
   retain the patch and both source identities externally. Do not silently
   rewrite checkpoint args or substitute another embedding space. Sources:
   [release stripping](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/release/strip_checkpoint.py),
   [A3 runner and calibration check](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/eval_zeroshot.py).

3. **Code, release and paper identities are not yet linked.**
   The pinned deployment manifest records source git SHA
   `e9ea42aed60f766f12ad19d51709129c50110a3b`, final checkpoint epoch 12,
   and student SHA256
   `e0317830b68ea51e6711fc90d4a35954d0528e5bd78a8d5afd966601ce4ed119`.
   These are provenance leads, not hashes verified against local files.
   The inspected HF model repository snapshot identity is
   `2db90096be5217bdc7a9003c042950f45723d105`; pin and hash its required
   artifacts before running. Establish compatibility across these revisions
   and identify the exact version/table of the paper; the paper table mapping
   remains pending. The upstream results page provides a candidate VG150
   released ViT-S/16+ A1 reference: R@50 0.533, mR@50 0.282, F1@50 0.369.
   Those are upstream reported values, not OpenSGG results. The four-axis model
   ladder OVS and five-axis headline OVS must remain distinct. Sources:
   [manifest](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/deploy/dist/relsgg-vits16plus/relateanything.json),
   [model release](https://huggingface.co/maelic/relsgg-vits16plus/tree/2db90096be5217bdc7a9003c042950f45723d105),
   [reported results](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/docs/results.md),
   [paper](https://arxiv.org/abs/2609.12552).

4. **Full split and denominator evidence is missing.**
   The official SPEC lists 26,404 VG150 test images. Access to the named
   evaluation-pack endpoint was not confirmed: the attempted unauthenticated
   request returned an authentication error, which does not establish that
   the dataset is absent or private. Pin the pack revision and file hashes,
   verify raw-image availability and image-root resolution, and reconcile
   expected, loaded and scored image/relation counts. The loader caps objects
   and drops relations involving trimmed objects; the exact cap is part of
   the official protocol and must be quantified, not changed to improve results.
   The evaluator skips images with no valid predicted pairs or no relations;
   fail or explain this explicitly rather than presenting an unexplained
   reduced denominator as complete evaluation. Sources:
   [SPEC](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/SPEC.md),
   [loader](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/data/dataset.py),
   [evaluator](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/eval/evaluator.py),
   [named data release](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

## Protocol details the acceptance contract must specify

- Runner defaults are `split=val`, graph constraint off, EMA weights,
  sigmoid scoring, 448-pixel images, batch size 64, 8 workers,
  evaluation budget 500, object cap 100, and `limit=0`. Explicitly record
  the effective values; A1 requires `--split test --graph_constraint`.
  Box mode has no rasters. The deployment manifest's 32-box/128-pair limits
  and 35-predicate example vocabulary are not the A1/A3 benchmark defaults.
  [Runner](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/eval_zeroshot.py).
- A1 uses image plus GT boxes without object-label model inputs, closed
  benchmark vocabulary reparameterization, and exact ordered box-index and
  predicate matching. The source class name `SGClsEvaluator` does not justify
  calling it standard OpenSGG SGCls or PredCls. Public API pixel-xyxy inputs
  must be distinguished from the packed loader's normalized cxcywh inputs,
  square bilinear resize, and float images in [0,1].
  [SPEC input contracts](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/SPEC.md),
  [loader](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/data/dataset.py).
- Code semantics override informal metric labels: A1 R@K averages image
  recalls; A1 mR@K averages per-image class recalls within each observed
  predicate, then across observed predicates. A3 SoftmR instead pools
  per-GT-group TP/GT before averaging. A1 R's upstream description as
  "micro" must not be interpreted as pooled relation TP/GT. F1 is the
  harmonic mean of the corresponding R and mR from the same pass.
  Graph constraint selects one argmax predicate per sampled ordered pair,
  then ranks those pairs. The A3 matcher uses exact strings or student-space
  cosine matching, and its inverse-mask flag is off by default; the current
  runner does not populate `ov_inverse_mask`. Verify the claimed inverse
  exclusion at the calibrated threshold rather than trusting stale help text.
  [Evaluator](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/eval/evaluator.py),
  [runner](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/eval_zeroshot.py).
- Preserve the shared score contract and pair-logit contribution. A
  monotone calibration does not change ranking, but replacing the shared
  sigmoid of combined logits with a product of two sigmoids does.
  Preserve training templates and student identity. A3 threshold 0.72 is
  a released-text-space property, not a universal threshold.
  [Pitfalls](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/docs/pitfalls.md).

## Integration and environment gaps

No local supported RelateAnything implementation has been established. The
existing `src/core/metrics.py` namespace/parser accepts standard
`predcls|sgcls|sgdet` R/mR keys, not this distinct input contract or F1/Soft
metrics. The existing RA-SGG PredCls path requires boxes and labels. Add an
explicit relation-from-supplied-regions contract and metric namespace after
the official reference gates pass. `guides/adding_new_model.md` alone does
not establish an inference-only adapter contract or reproduction evidence.

Upstream `pyproject.toml` requires Python >=3.12 and pins compatible dependency
series including torch 2.13, torchvision 0.28, transformers 5.14 and numpy
2.5; its comments identify Python 3.13 as the verified environment. OpenSGG's
documented older environment needs an isolated upstream reference environment
and a separately decided integration boundary. Do not loosen pins silently.
[Upstream dependency contract](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/pyproject.toml).

Before importing or redistributing upstream implementation, document the
AGPL-3.0-only code boundary, NOTICE attribution requirements, and separate
weight/data/DINOv3 terms against this repository's MIT distribution. An
external official-process boundary is a candidate to assess, not an already
resolved licensing conclusion. [NOTICE](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/NOTICE),
[third-party terms](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/THIRD_PARTY_NOTICES.md).

## Suggested acceptance gates

1. Fill baseline status, checkpoint manifest and evaluation protocol; record
   environment/license boundary, task mapping and evaluator ADRs.
2. Resolve paper table/model identity, dataset access and complete split
   inventory; pin release/code/data identities and hash required artifacts.
3. Validate strict offline loading and A1 invocation; resolve the A3
   released-layout mismatch before accepting any A3 run.
4. Run official checkpoint-backed A1 on the full official pack externally,
   with recorded configuration, denominators, logs and prediction artifacts.
5. Specify the stable inference contract and compare preprocessing, pair
   selection, logits, score composition, graph decoding and evaluator outputs
   against the official implementation, with tolerances agreed before acceptance.
6. Evaluate the same checkpoint/full split through the adapter, explain every
   deviation, and run repository documentation/claim/boundary guardrails.

All gates above remain proposed work. The current audit does not establish
checkpoint availability, paper parity, protocol parity or reproduction success.

## Review validation limits

The aggregate guardrail attempt during this review encountered an existing
extraction-manifest tip mismatch (`c312...` versus `95397...`) and an Application
Control block while importing SciPy's `_spline` DLL in the PENet adapter check.
These failures do not validate or invalidate RelateAnything parity; they leave
the aggregate repository check unpassed. They were not repaired in this
plan-review scope. Documentation and claims checks are reported separately by
the reviewing agent.
