# FREQ Deviations

Recorded deviations:

1. Paper-scale full VisualGenome evaluation was not run in this phase.
   - Reason: goal permits avoiding large full VG runs when resources are not
     available.
   - Impact: no paper-number alignment is claimed.
2. FREQ is exposed as an OpenSGG method using the local `PairFrequencyBias`
   implementation rather than importing an external official baseline script.
   - Reason: the repo already has the needed pair-prior code.
   - Impact: minimal implementation is OpenSGG-compatible and evaluator-stable,
     but official prior-table parity is not established.
3. Hidden-positive evaluation is represented by the existing GT-aligned JSONL
   predicate-recall schema, not by modifying SGG evaluator semantics.
   - Reason: hidden-positive protocol is an analysis-layer adaptation.
   - Impact: results are comparable within the local hidden-positive tooling,
     not automatically paper-standard SGG R@K.
4. CPU smoke hides CUDA via `CUDA_VISIBLE_DEVICES=""`.
   - Reason: host reports CUDA but fails driver initialization.
   - Impact: CPU smoke is valid; CUDA training/inference remains unvalidated in
     this environment.
5. The OpenSGG prior is rebuilt from local COCO-style `train.json` and
   `rel.json`, while the SGB reference builds `statistics['pred_dist']` from
   VG roidb/dict/image files with `must_overlap=True` and explicit background
   pair counts.
   - Reason: the local OpenSGG data format differs from the SGB reference
     implementation.
   - Impact: FREQ cannot be called paper/repo-aligned until the two prior
     tables are numerically compared on the same split and class mapping.
6. The current workspace does not contain the SGB-format roidb file
   `VG-SGG-with-attri.h5`.
   - Evidence: `tools/reproduction/check_sgb_freq_inputs.py` wrote
     `sgb_freq_input_check.json` with
     `BLOCKED_MISSING_SGB_VG_INPUTS`.
   - Impact: official SGB `statistics['pred_dist']` cannot be exported from
     the current workspace.
