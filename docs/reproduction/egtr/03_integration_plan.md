# EGTR Integration Plan

Plan:

1. Reuse the existing EGTR method and Visual Genome config.
2. Verify the local archive/checkpoint hashes.
3. Run no-training smoke under `conda hsg`.
4. Run checkpoint-backed relation export through `tools/analysis/export_egtr_predictions.py`.
5. Compute hidden-positive predicate recall from the exported JSONL.
6. Document that EGTR's validated path is SGDet query inference plus post-hoc GT alignment, not a Motifs-style PredCls forward.

Decision:

- No new EGTR model code is required for this goal.
- No protocol-changing patch should be applied merely to make EGTR look like a predicate-classification baseline.
