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
- The official X-101 SGDet test setting `TEST.RELATION.REQUIRE_OVERLAP=False`
  is used, including a CLI boolean parser guard for `False`.
- OpenSGG eval images are converted back to official BGR255 detector space and
  padded to the official size-divisible-by-32 image list shape.
- The union feature extractor includes the official reduce-channel ReLU after
  concatenating all FPN levels.

## Consequences

- The change is scoped to PE-NET SGDet and should not affect other model families.
- The smoke metric is no longer near zero after fixing extractor separation and relation-score background handling.
- Full Visual Genome metrics are close to, but still below, the official table:
  R@50 `28.82` vs `30.41` and mR@50 `11.94` vs `12.25`.
- Post-full 64-image probes for absolute-`xyxy` propagation and ROIAlign
  `aligned=False` did not clearly reduce the gap, so they are documented as
  diagnostics and excluded from the accepted adapter scope.
- This remains a checkpoint-backed evaluation until detector proposal,
  ROIAlign/FPN numeric, data-interface, and evaluator parity gaps are resolved
  or explicitly deferred.
