# Baseline Reproduction Issue: RA-SGG PredCls

## Baseline

- Method: RA-SGG / ReTAG
- Paper: *Learning Long-Tail High-Order Scene Graph Generation via Retrieval-Augmented Learning*
- Official repository: `https://github.com/KanghoonYoon/torch-rasgg`
- Official commit audited: `e8be01b9fde5c694243606e73931a4c8a8b1bf41`
- Target dataset/split: Visual Genome VG150, full test split (26,446 images)
- Target task: PredCls
- Target metrics: graph-constrained R@20/50/100 and mR@20/50/100

## Official Evidence

- Paper result: PredCls R@50/100 `62.2/64.1`, mR@50/100 `36.2/39.1`.
- Bundled checkpoint result: `checkpoints/RASGG/result.txt` reports
  R@20/50/100 `0.5546/0.6217/0.6410` and mR@20/50/100
  `0.2873/0.3615/0.3910`.
- Official config: `checkpoints/RASGG/config.yml` records GT boxes, GT object
  labels, 51 relation logits including background, `PREDICT_USE_BIAS: false`,
  all-level union pooling, and test resize `600/1000`.
- Official inference command: upstream `scripts/test.sh`; it sets model,
  retrieval, and frequency logit coefficients to `1/0/0`.
- Official evaluator reference: RU-Net `sgg_eval.py` mean-recall implementation.
- Reported checkpoint: local candidate `checkpoints/RASGG/model_final.pth`.
- Official Stanford H5 test selection: 26,446 images, 325,570 evaluator-view
  boxes, and 183,642 raw relation tuples. The legacy COCO conversion contains
  152,226 test relations because it removes 31,416 duplicate tuples.
- Official VG loading has two target views. The model view removes seven
  degenerate boxes across three images (and two relations incident to those
  boxes), while the evaluator view retains all H5 boxes and raw relations.

## Checkpoint Evidence

- Checkpoint path: `checkpoints/RASGG/model_final.pth`
- Source owner: presumed RA-SGG release, but original download record is absent.
- Download date: unknown.
- SHA256: `2488946f213686cad43a06f718735067f02a9fcbf966e2108cccfb1cd0edd8db`
- Expected architecture: ResNeXt-101-FPN plus RA-PENet relation predictor.
- Expected config: `checkpoints/RASGG/config.yml`.
- Compatibility verdict: partial; see the checkpoint manifest.

## OpenSGG Candidate Implementation

- Local files: `src/models/ra_sgg.py`, `src/methods/ra_sgg_method.py`,
  `src/core/metrics.py`, and `src/exp.py`.
- Claimed method name: `RA_SGG`.
- v14 differences: GT-pair-only prediction construction in the checked-in base,
  per-predicate Top-K, 800/1333 resize, incompatible RGB preprocessing, omitted
  all-level union reduction, and no verified visual-extractor checkpoint load.
- Fixes in this audit: all-pair ranking, global Top-K mean recall, fixed 50-class
  denominator, official image conversion/resize, all-level union pooling,
  visual checkpoint routing, legacy ROIAlign coordinates, official raw-H5 GT
  loading with separate model/evaluator targets, and removal of the non-official
  inference frequency prior.

## Decision Records

- Existing ADR: `reproduction/adr/0004-ra-sgg-predcls-v14-protocol-mismatch.md`.
- Relation-NMS is implemented from the explicitly referenced PE-Net upstream
  evaluator and recorded as upstream-inferred evidence in that ADR.

## Protocol Alignment Checklist

- [x] dataset split and raw target counts match the official H5 selection:
      26,446 images / 325,570 boxes / 183,642 relations
- [x] all 31,416 relation tuples omitted by the deduplicated COCO conversion are
      restored for evaluator GT
- [x] model/evaluator target views reproduce official handling of seven
      degenerate boxes; evaluator GT retains the two incident relations
- [x] object labels/classes match the VG dictionary after the class-order fix
- [x] predicate labels/classes match the VG dictionary after the class-order fix
- [ ] preprocessing has a checkpoint-backed golden-output parity test
- [x] PredCls ranks all directed object pairs and does not expose GT predicates
- [x] PE-Net upstream relation-NMS setting and thresholds are implemented/tested
- [x] graph-constrained global Top-K and 50-class mean-recall semantics are tested
- [ ] checkpoint loads without unexplained provenance or source-version gaps

## Success Conditions

- Close checkpoint provenance and exact predictor-source gaps.
- Obtain a direct RA-SGG evaluator archive or prediction dump to upgrade the
  upstream-inferred relation-NMS evidence to direct evidence.
- Pass a fixed official/local prediction evaluator parity fixture.
- [completed for the audit candidate] Run the full test split and persist the
  load report. A clean-source rerun is still required for a claim upgrade.
- Compare all six target metrics with `result.txt` without changing labels,
  ground truth, failed samples, or evaluator semantics.

## Failure / Defer Conditions

- Missing checkpoint provenance, unexplained active weight gaps, unresolved
  relation-NMS semantics, or a dirty/unarchived evaluator revision require a
  deferred or mismatch outcome.

## Current Outcome

- Claim status: `implementation_audit`
- Protocol verdict: `protocol_mismatch`

## Completed Full-Split Candidate

Run directory:
`outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715`.

| Metric | Candidate | Bundled official | Signed delta |
|---|---:|---:|---:|
| R@20 | 0.5547371506690979 | 0.5546 | +0.0001371507 |
| R@50 | 0.6218554973602295 | 0.6217 | +0.0001554974 |
| R@100 | 0.6410786509513855 | 0.6410 | +0.0000786510 |
| mR@20 | 0.2872789204120636 | 0.2873 | -0.0000210796 |
| mR@50 | 0.36152592301368713 | 0.3615 | +0.0000259230 |
| mR@100 | 0.3910185992717743 | 0.3910 | +0.0000185993 |

The maximum absolute delta is `0.0001554974`. This is strong numerical evidence
that the checkpoint-backed candidate now follows the intended metric protocol,
but it does not resolve the non-public predictor revision or checkpoint
provenance gaps.

Persisted evidence:

- `eval/summary.json`
- `eval/predcls/metrics.json`
- `lightning/metrics.csv`
- `metadata.json`
- `checkpoint_load_report.json`

## Notes

- `outputs/runs/ra_sgg/2026-07-12_rasgg_v14` is retained as diagnostic evidence
  only. It is not a baseline result and must not be relabeled after code changes.
- The completed full-split candidate is checkpoint-backed and matches the
  bundled official six-metric vector with maximum absolute delta
  `0.0001554974`; it remains implementation-audit evidence, not a successful
  reproduction claim.
- The completed run uses Python 3.10.8 and Lightning from the `hsg`
  environment with a temporary compatibility overlay providing PyTorch
  2.5.1+cu121/torchvision 0.20.1+cu121 and vendored `pkg_resources`. Its metadata
  records an RTX 2080 Ti and a dirty commit, so runtime/source reproducibility
  remains part of the audit boundary.
