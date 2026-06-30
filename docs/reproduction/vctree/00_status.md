# VCTree Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#62](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/62), branch `features-codex/repro-vctree` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- No local VCTree checkpoint found in `outputs/pretrained`.
- Synthetic forward smoke passed with loss `4.9152`.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.

Next action: review draft PR [#62](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/62) and merge when ready.
