# RA-SGG Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#65](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/65), branch `features-codex/repro-ra-sgg` -> `feat/analysis-tools`)

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- Official repo cloned for source audit:
  `/workspace/external/ra_sgg_official/torch-rasgg@e8be01b9fde5c694243606e73931a4c8a8b1bf41`.
- No local RA-SGG checkpoint or memory bank found.
- Minimal OpenSGG-compatible RA-SGG adapter added.
- Synthetic forward smoke passed with loss `4.7866`.
- Random-init/no-memory fallback standard PredCls metric slice accepted outputs;
  all reported R/mR values were 0.0.
- Random-init/no-memory fallback relation JSONL validation passed on 2 GT
  relations.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.

Next action: review draft PR [#65](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/65) and merge when ready.
