# ADR 0004: PENet SGDet PR84 Adapter Scope

## Status

Accepted for checkpoint-backed evaluation; not sufficient alone for a reproduction claim.

## Context

PR84 aligns PE-NET PredCls/SGCls behavior with the official PENET implementation.
SGDet additionally requires detector proposals from the official detector path.
The previous SGDet branch mixed relation and detector ROI box extractor weights,
which produced near-zero smoke metrics even with the official SGDet checkpoint.

## Decision

Build the SGDet path on top of PR84 and keep detector proposal generation
separate from relation prediction:

- `detector_box_extractor` loads `roi_heads.box.feature_extractor.*`.
- `relation_box_extractor` loads `roi_heads.relation.box_feature_extractor.*`.
- The detector RPN and detector box predictor load from the official SGDet checkpoint.
- Relation ROI features are re-extracted with the relation extractor after detector proposals.
- PE-NET SGDet outputs use a model-family-specific evaluator path.

## Consequences

- The change is scoped to PE-NET SGDet and should not affect other model families.
- The smoke metric is no longer near zero after fixing extractor separation and relation-score background handling.
- Full Visual Genome metrics are close to, but still below, the official table:
  R@50 `28.40` vs `30.41` and mR@50 `11.50` vs `12.25`.
- This remains a checkpoint-backed evaluation until detector proposal,
  data-interface, and evaluator parity gaps are resolved or explicitly deferred.
