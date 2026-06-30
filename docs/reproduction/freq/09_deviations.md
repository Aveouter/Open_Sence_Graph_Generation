# FREQ Deviations

Recorded deviations:

1. Paper-scale full VisualGenome evaluation was not run in this phase.
   - Reason: goal permits avoiding large full VG runs when resources are not
     available.
   - Impact: no paper-number alignment is claimed.
2. FREQ is exposed as an OpenSGG method using the local `PairFrequencyBias`
   implementation rather than importing an external official baseline script.
   - Reason: the repo already has the needed pair-prior code.
   - Impact: minimal implementation is OpenSGG-compatible and evaluator-stable.
3. Hidden-positive evaluation is represented by the existing GT-aligned JSONL
   predicate-recall schema, not by modifying SGG evaluator semantics.
   - Reason: hidden-positive protocol is an analysis-layer adaptation.
   - Impact: results are comparable within the local hidden-positive tooling,
     not automatically paper-standard SGG R@K.
4. CPU smoke hides CUDA via `CUDA_VISIBLE_DEVICES=""`.
   - Reason: host reports CUDA but fails driver initialization.
   - Impact: CPU smoke is valid; CUDA training/inference remains unvalidated in
     this environment.
