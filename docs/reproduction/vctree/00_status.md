# VCTree Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#62](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/62))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- No verified official checkpoint was available.
- Random-init fallback is not reproduction.

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
- Official input check is blocked by missing SGB-format
  `VG-SGG-with-attri.h5` and missing trusted VCTree checkpoint; see
  `vctree_official_input_check.json`.
- Source audit shows the local OpenSGG VCTree is a simplified implementation,
  not SGB `VCTreePredictor` parity.

Next action: choose/pin the target official implementation, provide SGB-format
VG input plus a trusted VCTree checkpoint, run official VCTree evaluation, then
decide whether OpenSGG should import/adapt the official structure or keep the
current local implementation as audit/smoke only.
