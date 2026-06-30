# VCTree Deviations

Recorded deviations:

1. No VCTree checkpoint available locally.
   - Fallback: explicit random-init inference/export.
   - Impact: no performance or paper-number claim.
2. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
3. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: local-analysis evidence only.
