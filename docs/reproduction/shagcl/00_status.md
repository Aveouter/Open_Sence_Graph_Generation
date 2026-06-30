# SHA-GCL Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#64](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/64))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Official SHA-GCL source is pinned at commit
  `8acfb818a0b2a88f9dd4a8ed64591ef856e66bf5`.
- No verified official checkpoint, pretrained detector checkpoint, or
  SHA-GCL-format VG input files are available locally.
- OpenSGG `SHAGCLModel` is a simplified adapter and not official
  `TransLike_GCL` parity.
- Random-init fallback is not reproduction.

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- `python tools/reproduction/check_shagcl_official_inputs.py --output docs/reproduction/shagcl/shagcl_official_input_check.json`
  returned `BLOCKED` with missing VG inputs, pretrained detector, and official
  SHA-GCL checkpoint.
- No local SHA-GCL checkpoint found in `outputs/pretrained`.
- Synthetic forward smoke passed with loss `3.7902`.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10: 0.0 / 0.0 / 1.0.
- Predicate mean recall mR@1/mR@5/mR@10: 0.0 / 0.0 / 1.0.

Next action: provide official VG inputs, pretrained detector, and trusted
SHA-GCL checkpoint before attempting checkpoint-backed evaluation.
