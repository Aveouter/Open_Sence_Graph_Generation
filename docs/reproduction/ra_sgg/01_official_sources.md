# RA-SGG Official Sources

Primary source:

- Official repository: https://github.com/KanghoonYoon/torch-rasgg
- Local audit checkout:
  `/workspace/external/ra_sgg_official/torch-rasgg`
- Commit audited: `e8be01b9fde5c694243606e73931a4c8a8b1bf41`.

Official README summary:

- RA-SGG is `Retrieval-Augmented Scene Graph Generation Framework via
  Multi-Prototype Learning`.
- It refers to PE-Net as the base implementation.
- ReTAG requires a pretrained PE-Net and a relation-embedding memory bank
  populated from the training dataset.
- Official PredCls script uses predictor `ReTAGPENet`, `RASGG.MEMORY_SIZE 8`,
  `RASGG.NUM_RETRIEVALS 20`, `RASGG.THRESHOLD 0.3`, mixup alpha/beta `20/5`,
  GT boxes, GT object labels, and `DTYPE float32`.
- Official config uses predictor names such as
  `RA-PENetCorrectProtoBetaEnhanceBG`.
- Official resources are hosted in Google Drive folders for pretrained models
  and memory-bank features.
- Official `Model_Zoo.md` lists ReTAG `model.pth` and `result.txt` artifacts
  for PredCls, SGCls, and SGDet.

OpenSGG interpretation:

- The local adapter uses PENet-style predicate prediction and exposes an
  optional memory-distribution fusion hook.
- Without official checkpoint/memory bank, only no-memory fallback testing is
  claimed.
- The local adapter does not implement official feature-bank top-k retrieval,
  reliable selection, BG correction, mixup, or prototype-loss behavior.
