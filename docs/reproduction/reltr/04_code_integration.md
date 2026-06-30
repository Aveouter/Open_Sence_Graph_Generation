# RelTR Code Integration

RelTR-specific model code already existed and was reused.

Supporting integration change:

- `tools/analysis/official_metric_eval.py`
  - Added project-root insertion to `sys.path` so it can run as a script from
    the repo root.

No dataset labels, ground truth, or evaluator semantics were changed.
