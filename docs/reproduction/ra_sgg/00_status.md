# RA-SGG Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#65](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/65))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Official ReTAG/RA-SGG source is pinned at commit
  `e8be01b9fde5c694243606e73931a4c8a8b1bf41`.
- Official resources require a pretrained PE-Net, ReTAG checkpoint, and
  memory-bank feature files.
- The adapter is minimal/no-memory and not official RA-SGG/ReTAG parity.
- Reopen only with official checkpoint and memory-bank behavior aligned.

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- `python tools/reproduction/check_rasgg_official_inputs.py --output docs/reproduction/ra_sgg/rasgg_official_input_check.json`
  returned `BLOCKED` with missing VG inputs, ReTAG checkpoints, pretrained
  PE-Net checkpoints, and memory-bank feature files.
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

Next action: provide official VG inputs, pretrained PE-Net, ReTAG checkpoint,
and memory-bank feature files before attempting checkpoint-backed evaluation.
