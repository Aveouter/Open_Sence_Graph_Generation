# Motifs Deviations

Recorded deviations:

1. Full VG paper-scale evaluation was not run.
   - Reason: current objective prioritizes automated test/evaluation and forbids
     training; large full benchmark runs are outside this tiny validation pass.
   - Impact: no paper-number alignment is claimed.
2. Local checkpoint only partially maps into OpenSGG Motifs.
   - Observed: 627 external keys remapped to 48 model keys.
   - Missing: `input_visual_proj.weight`, `input_visual_proj.bias`.
   - Impact: checkpoint-backed plumbing is validated, but strict official
     checkpoint parity is not established.
3. Evaluation was CPU-only.
   - Reason: host CUDA visibility/driver behavior is unreliable across
     environments; `conda hsg` CPU evaluation completed.
   - Impact: CUDA throughput or numerical parity is not claimed.
4. Hidden-positive metrics use the local GT-aligned JSONL predicate-recall
   analysis, not a change to standard SGG evaluator semantics.
   - Impact: hidden-positive results are local-analysis claims, not paper SGG
     benchmark claims.
