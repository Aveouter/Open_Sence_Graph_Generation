# PENet Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#63](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/63), branch `features-codex/repro-penet` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- No local PENet checkpoint found in `outputs/pretrained`.
- Synthetic forward smoke passed with loss `4.4942`.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.

Next action: review draft PR [#63](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/63) and merge when ready.
