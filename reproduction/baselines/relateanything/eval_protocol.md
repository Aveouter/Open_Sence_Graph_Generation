# Evaluation Protocol: RelateAnything

## Official Protocol

- Dataset/split: pinned `maelic/OV-SGG-Bench` VG150 full test pack; dataset
  revision and local image inventory still unverified. Official SPEC reports
  26,404 test images; do not substitute OpenSGG's other VG split definition.
- Task: image + supplied GT object regions, without object labels as model inputs.
- A1: encode/reparameterize to target predicate vocabulary with official student
  and templates. EMA, sigmoid, 448 square resize, budget 500, max objects 100,
  graph constraint, test, limit 0 are the audited benchmark settings.
- Public prediction API: original-image pixel xyxy boxes; PIL RGB images or
  explicitly BGR uint8 arrays. Mask coverage is optional and changes region input.
- Packed benchmark: normalized cxcywh boxes, square-resized float image in [0,1].
- Evaluator: official `relsgg/eval/evaluator.py`, ordered object indices and exact
  predicate identity; no box-IoU/object-class matching substitute in A1.
- A1 metrics: R@K averages image recalls; mR@K averages within observed classes
  using official per-image class accumulators; F1@K is harmonic R/mR.
- A3: full checkpoint vocabulary and original embedding order; exact/synonym
  matcher bound to the shipped student space. SoftmR aggregates differently from
  A1. Resolve released-layout and tau-record incompatibility before evaluation.
- Graph rule: one argmax predicate per sampled ordered pair, then global pair
  ranking. Preserve pair-logit contribution and official score contract.
- Denominators: reconcile expected/loaded/valid/scored image counts and relations
  dropped by official object caps. Empty predictions must never silently reduce
  reported coverage. Preserve official metric semantics and report exclusions.

## Local Protocol

The port exposes a separate supplied-region task with strict checkpoint loading,
the complete official model and an official packed-batch evaluation loop.
The model receives no target labels or GT relation pairs. A1 and A3 metrics use
their own `regions_*` namespace. Released-checkpoint inference and A1 evaluator
aggregation have passed integration comparisons; full-split parity is pending.
The loop rejects empty predictions/GT that the official evaluator would silently
skip, so no reduced-denominator result is accepted as a baseline run.

## Alignment Verdict

`partial_alignment` / `not_reproduction_ready`.

| Area | Official | Local | Impact |
|---|---|---|---|
| Model | Complete released architecture | Apache parent source port with hash verification | Source parity checked |
| Weights | Author release snapshot | Strict loaded; API predictions compared | Full-split verdict pending |
| Regions | GT boxes; labels excluded from model | Explicit region contract and raw forward | Separate supported task |
| Evaluator | Official index-based A1/A3 semantics | Official code; A1 loop comparison and A3 synonym test | Full-split parity pending |
| A3 assets | Runner expects legacy path/NPZ schema | Port accepts released names/W with bound consumer evidence | Matcher tau still required |
| Full split | Official pack | Access/inventory pending | No valid baseline denominator |

## Data Availability (2026-10-10)

The official VG150 evaluation pack is not publicly obtainable (see
[ADR 0014](../../adr/0014-relateanything-open-data-deferral.md) and the
[status](status.md) blocker section): `maelic/OV-SGG-Bench` returns 401
anonymously, is absent from the author's public datasets, and has no mirror.
Until access is restored the Official Protocol above cannot be executed. No
local substitute is accepted: the three accessible VG150 sources disagree on
the test denominator -- 26,404 (released pack), 26,446 (in-repo copy), 31,876
(public `maelic/VG150-coco-format` parquet rows).

## Sources

- [Pinned SPEC](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/SPEC.md)
- [Pinned evaluator](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/eval/evaluator.py)
- [Pinned runner](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/benchmark/eval_zeroshot.py)
- [Input contract ADR](../../adr/0012-relateanything-region-input-contract.md)
