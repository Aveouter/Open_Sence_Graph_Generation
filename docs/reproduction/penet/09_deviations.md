# PENet Deviations

Recorded deviations:

1. Official source was found, but local OpenSGG code is not official-code parity.
   - Official: `PrototypeEmbeddingNetwork` with 2048-dim prototype scoring,
     300d GloVe predicate prototypes, cosine logit scale, and prototype losses.
   - Local: `PENetContext` with 512 hidden dim, 200d embeddings, direct
     classifier, and no official prototype loss stack.
   - Impact: implementation audit only.
2. No official PENet checkpoint available locally.
   - Official Google Drive IDs are recorded, but no local trusted files are
     present under `outputs/pretrained/penet_official`.
   - Impact: no performance or paper-number claim.
3. No official pretrained detector checkpoint available locally.
   - Missing:
     `/workspace/external/penet_official/PENET/checkpoints/pretrained_faster_rcnn/model_final.pth`.
   - Impact: official test command cannot be run.
4. Official PENET-format VG inputs are unavailable.
   - Missing `VG-SGG-with-attri.h5`, `VG-SGG-dicts-with-attri.json`, and
     `image_data.json` under the official checkout.
   - Impact: official config/evaluator path cannot be validated.
5. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
6. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: local-analysis evidence only.
7. Independent draft GitHub PR submitted: #63.
   - Branch: `features-codex/repro-penet`.
   - Base: `feat/analysis-tools`.
8. Official `rel_nms` evaluator detail is identified but not aligned.
   - Impact: metric protocol remains unverified.
