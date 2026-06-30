# TDE Deviations

Recorded deviations:

1. Official TDE checkpoint unavailable/unverified.
   - Existing local checkpoints are not accepted as official TDE checkpoints.
   - Fallback used: explicit random-init inference/export.
   - Impact: no paper-number or checkpoint-performance claim.
2. Full official repository execution was not run in `hsg`.
   - Reason: official code targets an older Python/PyTorch/maskrcnn-benchmark
     stack, while the active objective requires `conda hsg`.
   - Impact: code-logic mapping is documented, but official runtime parity is
     not established.
3. Standard metric validation used a tiny 2-image CPU slice.
   - Impact: validates output schema and evaluator plumbing only.
4. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: hidden-positive adaptation is local-analysis evidence, not official
     TDE paper evidence.

Related legacy notes:

- `04_checkpoint.md` records official checkpoint URLs and prior download
  failures.
- `05_mapping.md` records official-to-OpenSGG logic mapping.
- `06_deviations.md` records earlier open implementation/checkpoint deviations.
