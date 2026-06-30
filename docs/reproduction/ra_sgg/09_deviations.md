# RA-SGG Deviations

Recorded deviations:

1. No official RA-SGG checkpoint available locally.
   - Fallback: explicit random-init inference/export.
2. No relation memory bank available locally.
   - Fallback: PENet-compatible no-memory adapter path.
3. No official pretrained PE-Net checkpoint available locally.
   - Official scripts require `checkpoints/PE-NET_PredCls/model_final.pth`
     and analogous SGCls/SGDet PE-Net checkpoints.
4. Official VG inputs are unavailable under the official checkout.
   - Missing `VG-SGG-with-attri.h5`, `VG-SGG-dicts-with-attri.json`, and
     `image_data.json`.
5. OpenSGG adapter is minimal, not strict official ReTAG parity.
   - Official code uses RA-PENet predictor variants, retrieval top-k, memory
     bank features, and prototype/mixup losses.
6. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
7. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
8. Independent draft GitHub PR submitted: #65.
   - Branch: `features-codex/repro-ra-sgg`.
   - Base: `feat/analysis-tools`.
