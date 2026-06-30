# EGTR Standard Metric Validation

Attempted direct metric probe:

```bash
conda run -n hsg bash -lc 'python - <<PY ... build EGTR, load checkpoint, run one test batch, call metric(...) ... PY'
```

Result:

- FAILED as a direct Motifs-style PredCls metric probe.
- Error: `KeyError: 'class_labels'` during EGTR criterion/matcher construction.

Interpretation:

- This is a protocol/target-schema mismatch, not evidence that the checkpoint cannot run.
- EGTR's OpenSGG path is SGDet query inference. The robust checkpoint-backed validation for this goal is therefore the EGTR-specific GT-aligned export plus hidden-positive recall.

Standard metric status:

- `NOT_CLAIMED_FOR_EGTR_DIRECT_PREDCLS`.
- No paper-number alignment is claimed.
- The suite row is closed as `HIDDEN_EVAL_ADAPTED`, not `PAPER_ALIGNED` or full benchmark validated.
