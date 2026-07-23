# ADR 0004: Treat RA-SGG PredCls v14 as a Protocol Mismatch

## Status

Accepted

## Context

The v14 run reports mR@20/50/100 `0.3705/0.3789/0.3801`, while its bundled
reference reports `0.2873/0.3615/0.3910`. The local evaluator gave each
predicted predicate class its own Top-K list, and the checked-in pair adapter
constructed predictions only for annotated GT relation pairs. Both reveal
information that the official global all-pair ranking does not provide.

The run metadata records a dirty commit without its diff. Other historical
differences include test resize, preprocessing, union pooling, and external
visual-weight loading. RA-SGG's repository omits the evaluator file, but its
README explicitly refers to PE-Net; the PE-Net evaluator hard-codes relation-NMS
for PredCls/SGCls with IoU `0.6` and default L21 `0.7`.

A subsequent target audit found another protocol mismatch: the legacy COCO
conversion has 152,226 test relations, while the official Stanford H5 selection
has 26,446 images, 325,570 boxes, and 183,642 raw relations. The 31,416 omitted
tuples are duplicate relations retained by the official evaluator. Official VG
loading also removes seven degenerate boxes (and two incident relations) from
the model view while retaining them in evaluator GT.

## Decision

- Label v14 `protocol_mismatch`; retain it only as diagnostic evidence.
- Correct all-pair/global-Top-K evaluation and the verified checkpoint/config
  mismatches in code.
- Read evaluation GT directly from the official H5 source, preserving order,
  coordinates, and duplicate tuples. Keep separate model and evaluator target
  views so the seven degenerate boxes and two incident relations follow the
  official behavior rather than being silently removed from metric GT.
- Implement PE-Net's relation-NMS thresholds as upstream-inferred protocol
  evidence and retain that provenance qualifier until a direct RA evaluator,
  log, or prediction fixture is recovered.
- Permit an optimized relation-NMS only while tests establish identical
  inclusive-IoU, tie ordering, predicate reassignment, background selection,
  and randomized selection results against a brute-force upstream transcription.
- Require a new output directory and clean source metadata for any
  claim-eligible full rerun. A dirty-source full run may remain implementation
  audit evidence only when its limitation is recorded explicitly.

## Evidence

- Paper/source: AAAI RA-SGG paper and official repository at commit
  `e8be01b9fde5c694243606e73931a4c8a8b1bf41`.
- Official evaluator source: RU-Net `SGMeanRecall.collect_mean_recall_items`.
- Relation-NMS source: VL-Group PE-Net `sgg_eval.py`, the upstream evaluator
  referenced by RA-SGG.
- Local run: `outputs/runs/ra_sgg/2026-07-12_rasgg_v14`.
- Local reference: `checkpoints/RASGG/result.txt` and `config.yml`.
- Regression test: `tests/reproduction/test_rasgg_predcls_protocol.py`.
- Dataset audit: official H5 test counts `26,446 / 325,570 / 183,642`, model
  view `325,563` boxes and `183,640` relations, legacy COCO test relations
  `152,226`.
- Completed checkpoint-backed candidate:
  `outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715`.
- Candidate R@20/50/100:
  `0.5547371506690979 / 0.6218554973602295 / 0.6410786509513855`; bundled
  official: `0.5546 / 0.6217 / 0.6410`.
- Candidate mR@20/50/100:
  `0.2872789204120636 / 0.36152592301368713 / 0.3910185992717743`; bundled
  official: `0.2873 / 0.3615 / 0.3910`.
- Maximum absolute metric delta: `0.0001554974`. Evidence files are
  `eval/summary.json`, `eval/predcls/metrics.json`, `lightning/metrics.csv`,
  `metadata.json`, and `checkpoint_load_report.json` under the candidate run.

## Consequences

- We can claim an implementation audit, identified protocol bugs, and a
  completed checkpoint-backed candidate whose six reported metrics match the
  bundled official result extremely closely (maximum absolute delta
  `0.0001554974`).
- We cannot claim RA-SGG reproduction or compare post-hoc corrected v14 metrics.
- The current hybrid-runtime, dirty-source run may be reported only with those
  qualifiers; completion alone will not close predictor/checkpoint provenance
  or direct-evaluator evidence gaps.
