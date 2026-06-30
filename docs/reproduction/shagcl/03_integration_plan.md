# SHA-GCL Integration Plan

Plan:

1. Reuse the existing OpenSGG SHA-GCL implementation.
2. Run no-training synthetic forward smoke under `conda hsg`.
3. Because no checkpoint is available, run explicit random-init fallback metric
   and JSONL export tests.
4. Record checkpoint absence and avoid paper-number claims.
