# USG-Par Implementation Audit

## Scope

This audit compares the local OpenSGG USG candidate against the official
repository at https://github.com/ChocoWu/USG. It targets algorithmic alignment
of the image-only SGDet path, not a checkpoint-backed reproduction.

## Aligned in Code

- Detection head now follows the official CLIP-style cosine classifier:
  query projection, class-name text embeddings, learnable no-object embedding,
  and learnable logit scale.
- Class-name embeddings are built from VG category names with OpenCLIP when the
  text tower is available, then projected to the model hidden dimension.
- Object matching now uses sigmoid class probabilities, matching official USG
  loss semantics.
- Object classification now uses per-query sigmoid BCE with matched/no-object
  weights instead of softmax CE.
- RPC now follows official top-k selection over the full pair matrix and the
  two-way RAC layer reads previous-layer subject/object states in parallel.
- Relation targets now preserve multiple predicates per subject-object pair via
  multi-hot targets.
- Predicate loss now uses BCEWithLogits, and pair-confidence loss uses weighted
  BCE, matching official `L_rel = L_predicate + L_pair`.
- USG SGDet evaluation now routes through a USG-specific adapter using sigmoid
  object/predicate scores and VG foreground label offsets.

## Deliberate Dataset Adaptations

- Official PSG uses mask losses; the OpenSGG VG path keeps L1/GIoU box losses.
- Official universal/multimodal training includes association and contrastive
  terms; this local baseline currently covers the image-only SGDet path.

## Remaining Non-Alignment

- No official checkpoint identity or SHA256 is recorded.
- Official PSG split/preprocessing/evaluator parity has not been verified.
- Visual Genome protocol cannot be described as official USG reproduction unless
  an official VG configuration/checkpoint/protocol is documented.

## Claim

Current claim is `implementation_audit` only.
