# TDE Integration Plan

Plan:

1. Reuse the existing OpenSGG `TDEModel` and `TDE_Method`.
2. Treat official checkpoint absence as a deviation, not a reason to alter
   evaluator semantics.
3. Prefer verified official/local TDE checkpoint if available.
4. If unavailable, run explicit random-init fallback tests:
   - method instantiation and synthetic forward,
   - tiny standard PredCls metric slice,
   - relation JSONL export,
   - GT-aligned predicate recall.
5. Keep paper-number alignment unclaimed.

No training plan:

- The active objective forbids training.
- TDE evidence is inference/test/export only.
