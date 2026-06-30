# RelTR Integration Plan

Plan:

1. Reuse existing OpenSGG RelTR implementation.
2. Use local checkpoint for all validation.
3. Run no-training smoke under `conda hsg`.
4. Export relation JSONL with Hungarian matcher alignment.
5. Run GT-aligned predicate recall.
6. Run standard PredCls metric slice with `official_metric_eval.py`.
