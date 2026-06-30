# SHA-GCL Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#64](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/64), branch `features-codex/repro-shagcl` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- No local SHA-GCL checkpoint found in `outputs/pretrained`.
- Synthetic forward smoke passed with loss `3.7902`.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10: 0.0 / 0.0 / 1.0.
- Predicate mean recall mR@1/mR@5/mR@10: 0.0 / 0.0 / 1.0.

Next action: review draft PR [#64](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/64) and merge when ready.
