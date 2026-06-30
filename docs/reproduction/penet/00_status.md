# PENet Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#63](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/63))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Official PENet source is pinned at commit
  `9c9f50777c66647799cb7a17dfb855aa98aecd1e`.
- No verified official checkpoint, pretrained detector checkpoint, or
  PENET-format VG input files are available locally.
- OpenSGG `PENetContext` is a simplified adapter and not official
  `PrototypeEmbeddingNetwork` parity.
- Random-init fallback is not reproduction.

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- `python tools/reproduction/check_penet_official_inputs.py --output docs/reproduction/penet/penet_official_input_check.json`
  returned `BLOCKED` with missing PENET VG inputs, pretrained detector, and
  official PENet checkpoints.
- No local PENet checkpoint found in `outputs/pretrained`.
- Synthetic forward smoke passed with loss `4.4942`.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.

Next action: provide official PENET-format VG inputs, the official pretrained
detector, and trusted PredCls/SGCls/SGDet checkpoints before attempting
checkpoint-backed evaluation.
