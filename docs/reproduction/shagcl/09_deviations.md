# SHA-GCL Deviations

Recorded deviations:

1. No SHA-GCL checkpoint available locally.
   - Fallback: explicit random-init inference/export.
   - Impact: no performance or paper-number claim.
2. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
3. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: local-analysis evidence only.
4. Independent draft GitHub PR submitted: #64.
   - Branch: `features-codex/repro-shagcl`.
   - Base: `feat/analysis-tools`.
