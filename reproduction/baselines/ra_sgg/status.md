# Baseline Status: RA-SGG PredCls

## Outcome

`implementation_audit`

Protocol alignment verdict: `protocol_mismatch`.

## Claim Boundary

The historical v14 run is not reproduction evidence. It used incompatible
evaluation and checkpoint/preprocessing paths, and its dirty source diff was not
archived. The current fixes and completed full checkpoint-backed evaluation
remain an implementation audit despite the candidate's close metric match,
until the source, provenance, evaluator-fixture, and runtime gaps are resolved.

## Evidence Summary

| Requirement | Evidence | Verdict |
|---|---|---|
| Official paper identified | AAAI 2025 RA-SGG paper | present |
| Official repository identified | `KanghoonYoon/torch-rasgg`, pinned commit recorded | present |
| Checkpoint provenance verified | SHA256 known, original URL/name/checksum absent | partial |
| Config aligned | Core PredCls settings corrected to bundled config/test script | partial |
| Official H5 targets aligned | 26,446 images / 325,570 boxes / 183,642 raw relations; dual target behavior checked | implementation-audited |
| Inference flow aligned | all-pairs, no GT predicate input, no inference retrieval/frequency | implementation-audited |
| Evaluator semantics aligned | global Top-K, fixed 50-class denominator, and PE-Net relation-NMS equivalence tests pass | core semantics aligned |
| Metrics comparable to bundled result | completed six-metric vector; maximum absolute delta `0.0001554974` | candidate match |

## Diagnostic Result Shape

| Metric | v14 | Bundled reference | Delta |
|---|---:|---:|---:|
| R@20 | 0.5199 | 0.5546 | -0.0347 |
| R@50 | 0.5889 | 0.6217 | -0.0328 |
| R@100 | 0.6082 | 0.6410 | -0.0328 |
| mR@20 | 0.3705 | 0.2873 | +0.0832 |
| mR@50 | 0.3789 | 0.3615 | +0.0174 |
| mR@100 | 0.3801 | 0.3910 | -0.0109 |

The almost flat v14 mR curve is consistent with predicate-specific Top-K and
GT-pair candidate filtering: small K is inflated while larger K adds little.

## Completed Full-Run Candidate

- Directory:
  `outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715`
- Dataset target audit: 26,446 official-H5 test images, 325,570 evaluator boxes,
  and 183,642 raw relations. Relative to the legacy COCO conversion, 31,416
  duplicate relation tuples are restored.
- Dual target audit: seven degenerate boxes in three images are removed only
  from the model view (325,563 boxes); all evaluator boxes and the two incident
  relations remain in GT.
- Checkpoint audit: SHA256
  `2488946f213686cad43a06f718735067f02a9fcbf966e2108cccfb1cd0edd8db`;
  637/648 tensors load with exact shape (63 predictor + 574 visual), with no
  shape adaptation.
- Runtime: Python 3.10.8/Lightning from `hsg`, temporary PyTorch
  2.5.1+cu121/torchvision 0.20.1+cu121 compatibility overlay, RTX 2080 Ti.
- State: completed.

| Metric | Candidate | Bundled official | Signed delta |
|---|---:|---:|---:|
| R@20 | 0.5547371506690979 | 0.5546 | +0.0001371507 |
| R@50 | 0.6218554973602295 | 0.6217 | +0.0001554974 |
| R@100 | 0.6410786509513855 | 0.6410 | +0.0000786510 |
| mR@20 | 0.2872789204120636 | 0.2873 | -0.0000210796 |
| mR@50 | 0.36152592301368713 | 0.3615 | +0.0000259230 |
| mR@100 | 0.3910185992717743 | 0.3910 | +0.0000185993 |

The maximum absolute delta is `0.0001554974`. Exact metrics are stored in
`eval/summary.json`, `eval/predcls/metrics.json`, and `lightning/metrics.csv`;
`metadata.json` and `checkpoint_load_report.json` preserve runtime and load
evidence. A tracked, self-contained metric summary plus hashes for those local
run artifacts is preserved in
`reproduction/evidence/ra_sgg/predcls_candidate_2026-07-15.json`. This
checkpoint-backed candidate matches the bundled official result extremely
closely, but it is not labeled `reproduced`.

## Current Blocker

- Exact checkpoint/source provenance is incomplete.
- The relation-NMS thresholds are upstream-inferred from PE-Net because the
  RA-SGG repository does not include its evaluator file directly.
- No official/local golden prediction evaluator fixture is available.
- The completed full run uses a dirty source tree and a temporary hybrid runtime;
  an exact clean-environment rerun has not been established.

## Next Action

- Obtain a direct RA-SGG evaluator archive/prediction dump to strengthen the
  current PE-Net upstream evidence.
- Add an official/local golden evaluator fixture.
- Preserve the completed candidate's summary, metric CSV, metadata, and load
  report, and do not overwrite or reinterpret v14.
- Pin a clean runtime and archive the exact source diff before any claim upgrade.
