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
- Official config uses predictor names such as
  `RA-PENetCorrectProtoBetaEnhanceBG`.
- Official resources are hosted in Google Drive folders for pretrained models
  and memory-bank features.

OpenSGG interpretation:

- The local adapter uses PENet-style predicate prediction and exposes an
  optional memory-distribution fusion hook.
- Without official checkpoint/memory bank, only no-memory fallback testing is
  claimed.
