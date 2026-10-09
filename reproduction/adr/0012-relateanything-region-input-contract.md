# ADR 0012: Give RelateAnything a supplied-region relation contract

## Status

Accepted for task semantics; implementation and parity remain pending.

## Context

Issue #124 targets a relation model that receives image and supplied regions,
without object labels. OpenSGG's existing PredCls/SGCls/SGDet names and evaluator
adapters are not evidence of this task's semantics. The upstream class name
`SGClsEvaluator` does not imply standard OpenSGG SGCls.

## Decision

Expose supplied-region relation inference explicitly. The public boundary uses
original-image pixel xyxy boxes; object labels may be retained by the caller for
display or evaluation but are never passed as model features. Relation pair
selection remains the model's responsibility, not a GT pair oracle.

Use separate A1/A3 relation metric namespaces and preserve the official evaluator
aggregation, graph constraint, pair-score contribution and vocabulary order.
F1 means harmonic R/mR, not precision/recall F1. A3 needs its own matcher and
text-space calibration. Do not add a background channel or use generic softmax
decoding to make the outputs fit an existing adapter.

## Evidence

- [Issue #124](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/124)
- [Pinned API](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/api.py)
- [Pinned evaluator](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/eval/evaluator.py)
- Local `src/core/metrics.py` only names standard SGG task R/mR metrics.

## Consequences

Integration work must test inputs, vocabulary changes, scores, decoding and
aggregation through public interfaces against the official reference. Task
identity alone does not prove reproduction; a full checkpoint-backed reference
and local parity run remain required. The integration/license/environment
boundary will be recorded separately before upstream code is imported or copied.
