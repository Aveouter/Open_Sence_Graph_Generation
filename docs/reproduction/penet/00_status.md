# PENet Status

Current status: `IMPLEMENTATION_AUDIT` / `PROTOCOL_MISMATCH` /
`NOT_REPRODUCTION_READY`

PR status: `PR_OPEN_AUDIT` ([#81](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/81))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Official PENet source is pinned at commit
  `9c9f50777c66647799cb7a17dfb855aa98aecd1e`.
- The official SGDet checkpoint is available locally at
  `outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth`.
- SGDet checkpoint SHA256:
  `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183`.
- Official pretrained detector checkpoint, PENET-format VG input files, and
  `VG_100K` image directory are now present from official/backup sources; the
  SGDet input gate passes.
- OpenSGG `PENetContext` now mirrors the official
  `PrototypeEmbeddingNetwork` relation-predictor modules and can remap
  `roi_heads.relation.predictor.*` weights from the official checkpoint.
- Official detector checkpoint weights now map into local PE-NET detector
  feature modules with load counts `backbone=520`, `fpn=16`,
  `box_extractor=4`, `rpn_head=6`, `box_predictor=4`.
- Local PE-NET SGDet eval now has an eval-only detector proposal path that
  emits `boxes_per_cls` and `obj_dists`/`predict_logits` from the official
  detector checkpoint.
- Full local OpenSGG SGDet evaluation completed on the full Visual Genome test
  split, but the checkpoint-backed result does not align with the official
  PE-NET SGDet table.
- `--penet_detector_ckpt` can now be supplied at evaluation time instead of
  hardcoding the local detector path in `configs/VisualGenome/PE_NET.py`.
- SGDet evaluation is guarded against missing detector proposal fields
  (`boxes_per_cls` and `predict_logits`/`scores_all`) to prevent accidental
  PredCls/SGCls protocol fallback.
- Random-init fallback is not reproduction.

Completed phases: 0-10.

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last result:

- Full local OpenSGG SGDet eval-only run completed with the official relation
  checkpoint and official-backup Faster R-CNN detector checkpoint:
  `26446/26446` test images, `sgdet_R@50 = 0.08229778707027435`,
  `sgdet_mR@50 = 0.018936751410365105`. These do not align with the official
  README targets `R@50 = 0.3041` and `mR@50 = 0.1225`, so the result is
  `protocol_mismatch` / `not_reproduction_ready`, not reproduction success.
- Data gap audit found that local OpenSGG VG JSON uses a zero-relation-filtered
  subset of the official PENET/SGB H5 split, deduplicates exact test
  relationship triples, and stores integer COCO-style boxes with up to `1.0`
  pixel difference from official H5-derived boxes.
- `python tools/reproduction/check_penet_official_inputs.py --protocol sgdet --output docs/reproduction/penet/penet_official_input_check.json`
  returned `PASS` after retrieving the official-backup VG H5 and pretrained
  Faster R-CNN detector from Weiyun and linking the existing local VG image
  pack used by RELTR.
- Local OpenSGG eval-only probe loaded the official detector checkpoint and
  official SGDet relation checkpoint, generated SGDet proposals, reached the
  evaluator, and produced a 1-image `pipeline_smoke_only` metric table with
  all R/mR values equal to `0.0`.
- Official detector checkpoint load probe succeeded for local PE-NET detector
  modules:
  `{'backbone': 520, 'fpn': 16, 'box_extractor': 4, 'rpn_head': 6, 'box_predictor': 4}`.
- Official PENET evaluator execution is blocked in `conda hsg` because the
  legacy maskrcnn-benchmark extension does not build under the current
  Python/PyTorch/CUDA stack and `apex` is unavailable.
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

Next action: isolate the full-VG parity gap. Known suspects are local VG JSON
input semantics versus official H5 loader semantics, the local OpenSGG eval
resize path (`800/1333`) versus the official PENET `600/1000` preprocessing,
the local PE-NET-only detector proposal adapter versus official
maskrcnn-benchmark `GeneralizedRCNN`, and unproven local evaluator parity.
Official PENET evaluator parity remains blocked in the current `hsg`
environment by legacy maskrcnn-benchmark/APEX runtime issues.
