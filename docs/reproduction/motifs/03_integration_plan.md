# Motifs Integration Plan

Plan:

1. Reuse the existing OpenSGG Motifs implementation and local checkpoint rather
   than rewriting the model.
2. Validate import, instantiation, and inference-only synthetic forward under
   `conda hsg`.
3. Validate a tiny PredCls standard metric slice with the local checkpoint.
4. Validate relation JSONL export with the local checkpoint.
5. Compute GT-aligned predicate recall from the exported JSONL.
6. Record checkpoint mapping deviations and avoid paper-number claims.

No training plan:

- The active objective forbids model training.
- Any previous train-smoke command is not used as accepted evidence for this
  method closure.
