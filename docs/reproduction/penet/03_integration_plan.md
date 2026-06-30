# PENet Integration Plan

Plan:

1. Reuse the existing OpenSGG PENet implementation.
2. Run no-training synthetic forward smoke under `conda hsg`.
3. Because no checkpoint is available, run explicit random-init fallback metric
   and JSONL export tests.
4. Fix config alias mismatch for exporter (`PENet` -> `PE_NET.py`).
5. Record checkpoint absence and avoid paper-number claims.
