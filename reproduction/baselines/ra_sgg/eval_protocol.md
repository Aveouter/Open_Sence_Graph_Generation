# Evaluation Protocol: RA-SGG PredCls

## Official Protocol

- Dataset: Visual Genome VG150
- Split: full test split, 26,446 images
- Ground truth source: Stanford `VG-SGG-with-attri.h5`, selected with official
  split/valid-box/valid-relation rules; 325,570 boxes and 183,642 raw relations
- Task: PredCls
- Preprocessing: shorter side 600, maximum side 1000; BGR255 with pixel mean
  `[102.9801, 115.9465, 122.7717]`; zero padding divisible by 32
- Inference inputs: GT boxes and GT object labels, but no GT relation-pair oracle
- Target views: model input drops clipped zero-width/zero-height boxes, while
  evaluator GT retains them and their raw relations (`evaluation=True` behavior)
- Candidate pairs: every directed pair, `N × (N - 1)`
- Relation logits: full background-inclusive softmax; rank foreground scores
- Inference fusion: model/retrieval/frequency coefficients `1/0/0` and
  `PREDICT_USE_BIAS=false`
- Evaluator: graph-constrained recall and SGMeanRecall
- Metrics: R@20/50/100 and mR@20/50/100
- Top-k rules: one globally ranked prediction list per image; mR then computes
  per-GT-predicate recall and averages all 50 foreground predicates
- Constraint settings: `MULTIPLE_PREDS=false`; PredCls/SGCls relation-NMS from
  the referenced PE-Net evaluator, IoU threshold `0.6`, L21 threshold `0.7`

## Local Protocol After This Audit

- Dataset/split/task: same VG150 full-test PredCls target
- Ground truth: official H5 order and coordinates, with all 183,642 raw relation
  tuples. This restores 31,416 duplicates omitted by the 152,226-relation COCO
  conversion.
- Dual targets: evaluator GT has 325,570 boxes; model input has 325,563 after
  removing seven degenerate boxes across three images. Two relations incident
  to those boxes are absent from the model-view relation labels but remain in
  evaluator GT, matching the official loader/evaluator split.
- Preprocessing: configured 600/1000; padded tensors are cropped to image size,
  converted RGB to BGR Caffe space, then zero-padded to a multiple of 32
- Inference inputs: GT boxes and object labels; relation labels are zeroed before
  evaluation forward
- Candidate pairs: all directed pairs from `generate_object_pairs`
- Relation logits: softmax across 51 logits, then remove background column 0
- Inference fusion: cosine model logits only; frequency bias follows the false
  config and retrieval is training-only
- Evaluator: OpenSGG `SceneGraphEvaluator` with a shared global prediction list
  for each predicate's mean-recall accumulator
- Relation-NMS: enabled for PredCls/SGCls with IoU `0.6` and L21 `0.7`, using a
  memory-bounded vectorized implementation equivalent to the PE-Net upstream
  algorithm. It caches object-level inclusive IoU, evaluates pair overlaps in
  blocks, and replaces repeated flattened argmax scans with a tie-compatible
  row-maximum heap. With full softmax rows, the L21 `0.7` condition is always
  true by a `sqrt(2)` lower bound; a general blocked L21 path remains available.
- Mean-recall denominator: fixed 50 foreground predicates
- Verification: `tests/reproduction/test_rasgg_predcls_protocol.py` covers
  all-pair/global Top-K, fixed 50-class mR, dual targets, exact inclusive-IoU
  parity, background assignment, and randomized selection equivalence against
  a brute-force transcription of upstream relation-NMS.

## Completed Full Evaluation

- Run directory:
  `outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715`
- Scope: full 26,446-image PredCls test split, batch size one
- Recorded runtime: Python 3.10.8, PyTorch 2.5.1+cu121, RTX 2080 Ti
- Environment construction: `hsg` Python/NumPy/Lightning plus a temporary
  compatibility overlay for PyTorch 2.5.1, torchvision 0.20.1, and vendored
  `pkg_resources`
- State: completed

| Metric | Candidate | Bundled official | Signed delta |
|---|---:|---:|---:|
| R@20 | 0.5547371506690979 | 0.5546 | +0.0001371507 |
| R@50 | 0.6218554973602295 | 0.6217 | +0.0001554974 |
| R@100 | 0.6410786509513855 | 0.6410 | +0.0000786510 |
| mR@20 | 0.2872789204120636 | 0.2873 | -0.0000210796 |
| mR@50 | 0.36152592301368713 | 0.3615 | +0.0000259230 |
| mR@100 | 0.3910185992717743 | 0.3910 | +0.0000185993 |

Maximum absolute delta: `0.0001554974`. Evidence is persisted in
`eval/summary.json`, `eval/predcls/metrics.json`, `lightning/metrics.csv`,
`metadata.json`, and `checkpoint_load_report.json` under the run directory.

## v14 Protocol

The historical v14 run was created from dirty commit `035958e` without an
archived diff. The checked-in base constructed predictions only for GT relation
pairs, and the evaluator filtered predictions by predicate before applying K.
Its hparams also omitted 600/1000 test-size keys. The exact dirty state cannot be
reconstructed, so v14 is diagnostic evidence only.

## Alignment Verdict

`mismatch`

## Mismatches

| Area | Official | Current local status | Impact |
|---|---|---|---|
| Source identity | Exact released predictor revision | Bundled config names a non-pinned variant | Exact implementation parity unproven |
| Relation-NMS provenance | RA-SGG omits evaluator through its dataset ignore; README refers to PE-Net | Implemented from PE-Net's hard-coded PredCls/SGCls IoU `0.6`, L21 `0.7` evaluator | Semantics aligned from upstream, but direct RA archive still absent |
| Checkpoint provenance | Release identity and checksum | Local renamed file with no source manifest | Identity not independently verified |
| Evaluator implementation | maskrcnn-benchmark/RU-Net family | Semantics covered by synthetic tests, not a golden official prediction dump | End-to-end parity unproven |
| Runtime identity | Released code's original dependency stack | Temporary hybrid environment; run metadata is dirty | Numerical/runtime parity and exact rerun are not yet independently established |

The completed checkpoint-backed candidate matches the bundled official metrics
extremely closely, but it must not be labeled a reproduction until the remaining
source/provenance and direct-evaluator evidence gaps are closed.
