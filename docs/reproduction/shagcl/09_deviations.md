# SHA-GCL Deviations

Recorded deviations:

1. Official source was found, but local OpenSGG code is not official-code parity.
   - Official: `TransLike_GCL`, `Hybrid-Attention`, `divide4`,
     `KL_logit_TopDown`, grouped auxiliary classifiers, `FrequencyBias_GCL`.
   - Local: simplified `SHAGCLModel` with pair-level attention, learned group
     prototypes, and MSE-style `gcl_loss`.
   - Impact: implementation audit only.
2. No SHA-GCL checkpoint available locally.
   - Fallback: explicit random-init inference/export.
   - Impact: no performance or paper-number claim.
3. Missing official VG inputs and pretrained detector.
   - Impact: official config/evaluator path cannot be run.
4. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
5. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: local-analysis evidence only.
6. Independent draft GitHub PR submitted: #64.
   - Branch: `features-codex/repro-shagcl`.
   - Base: `feat/analysis-tools`.
