# RelTR Code Review

Review status: PASS for checkpoint-backed no-training reproduction.

Checks:

- Existing RelTR implementation is reused.
- Checkpoint is loaded through the existing project checkpoint adaptation path.
- Relation JSONL export uses Hungarian matcher indices for GT relation rows.
- Standard metrics use existing evaluator helpers.

Residual risk:

- Tiny CPU slice validates plumbing, not paper-level accuracy.
