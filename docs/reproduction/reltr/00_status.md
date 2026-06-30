# RelTR Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#66](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/66), branch `features-codex/repro-reltr` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- Local checkpoint: `outputs/pretrained/reltr/reltr_vg.pth`.
- SHA256: `2b3601e5cc5d835d9538d78b698dd7b3f4f1646bdb0e5c459d2cbe959d5906b1`.
- Synthetic forward smoke passed.
- Checkpoint-backed relation JSONL export passed on 2 rows.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.
- Checkpoint-backed standard metric slice processed 1 batch; all PredCls R/mR
  metrics were 0.0.

Next action: review draft PR [#66](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/66) and merge when ready.
