# PENet Status

Current status: `IMPLEMENTATION_AUDIT` / `PROTOCOL_MISMATCH`

PR status: `PR_CLOSED_DEFERRED` ([#63](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/63))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Official PENet source is pinned at commit
  `9c9f50777c66647799cb7a17dfb855aa98aecd1e`.
- The official SGDet checkpoint is available locally at
  `outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth`.
- SGDet checkpoint SHA256:
  `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183`.
- Official pretrained detector checkpoint, PENET-format VG input files, and
  `VG_100K` image directory are still missing locally.
- OpenSGG `PENetContext` now mirrors the official
  `PrototypeEmbeddingNetwork` relation-predictor modules and can remap
  `roi_heads.relation.predictor.*` weights from the official checkpoint.
- SGDet evaluation is guarded against missing detector proposal fields
  (`boxes_per_cls` and `predict_logits`/`scores_all`) to prevent accidental
  PredCls/SGCls protocol fallback.
- Random-init fallback is not reproduction.

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- `python tools/reproduction/check_penet_official_inputs.py --protocol sgdet --output docs/reproduction/penet/penet_official_input_check.json`
  returned `BLOCKED` with missing PENET VG inputs, missing `VG_100K` images,
  and missing pretrained detector. The target SGDet checkpoint is present and
  checksum-recorded.
- Synthetic forward smoke passed with finite synthetic loss across local
  reruns. The value is stochastic because the smoke uses random tensors.
- Clean integration regression coverage now verifies PENet registration,
  builder config mapping, Motifs-compatible forward schema, and no-pair output
  handling.
- Random-init fallback standard PredCls metric slice accepted outputs; all
  reported R/mR values were 0.0.
- Random-init fallback relation JSONL validation passed on 2 GT relations.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.

Current SGDet parity report:

- `docs/reproduction/penet/12_sgdet_official_parity.md`

Next action: provide official PENET-format VG inputs, `VG_100K` images, and
the official pretrained detector before attempting checkpoint-backed SGDet
evaluation.
