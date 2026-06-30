# RA-SGG Integration Plan

Plan:

1. Audit official RA-SGG source and config.
2. Add a minimal OpenSGG-compatible RA-SGG adapter rather than treating local
   `REACT` as RA-SGG.
3. Reuse PENet base logic and expose explicit memory-bank fusion fields.
4. Because no checkpoint or memory bank is available, run explicit
   random-init/no-memory fallback tests.
5. Record all deviations and avoid paper-number claims.
