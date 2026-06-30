# RA-SGG Deviations

Recorded deviations:

1. No official RA-SGG checkpoint available locally.
   - Fallback: explicit random-init inference/export.
2. No relation memory bank available locally.
   - Fallback: PENet-compatible no-memory adapter path.
3. OpenSGG adapter is minimal, not strict official ReTAG parity.
   - Official code uses RA-PENet predictor variants, retrieval top-k, memory
     bank features, and prototype/mixup losses.
4. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
5. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
6. Independent draft GitHub PR submitted: #65.
   - Branch: `features-codex/repro-ra-sgg`.
   - Base: `feat/analysis-tools`.
