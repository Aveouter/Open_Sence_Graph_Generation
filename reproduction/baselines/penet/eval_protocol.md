# Evaluation Protocol: PENet SGDet

## Official Protocol

- Dataset: PENET/SGB Visual Genome
- Split: official VG test split
- Task: SGDet
- Preprocessing: BGR image tensor with detector pixel mean/std, 600/1000 eval resize, and size-divisible-by-32 image-list padding
- Inference inputs: raw image only; detector proposals, object labels, boxes, and predicates predicted by the model
- Evaluator: PENET/SGB Visual Genome SGDet evaluator
- Metrics: R@20/50/100 and mR@20/50/100
- Top-k rules: relation triplets sorted by relation score times subject/object scores
- Constraint settings: graph-constrained SGDet

## Local Protocol

- Dataset: OpenSGG Visual Genome dataloader for the current smoke/full runs
- Split: OpenSGG VG test split
- Task: SGDet
- Preprocessing: PE-NET method-local BGR255 conversion, detector pixel-mean subtraction, and official-space pad32
- Inference inputs: raw image, with eval-only detector proposal adapter
- Evaluator: PE-NET-specific OpenSGG compact SGDet evaluator using `evaluate_recall`
- Metrics: `sgdet_R@20/50/100` and `sgdet_mR@20/50/100`
- Top-k rules: detected-box relation pairs sorted by relation score times subject/object scores, capped at top 100
- Constraint settings: graph-constrained SGDet

## Alignment Verdict

partial

The current adapter is suitable for checkpoint-backed SGDet evaluation, but the
full run remains below the official table: R@50 `28.82` vs `30.41`, mR@50
`11.94` vs `12.25`. The remaining protocol work is therefore treated as a
parity gap, not as a successful reproduction.

## Mismatches

| Area | Official | Local | Impact |
|---|---|---|---|
| Runtime process | Original PENET/SGB test script | OpenSGG Lightning test wrapper | May affect data ordering, transforms, and evaluator aggregation |
| Dataset interface | PENET/SGB h5/json dataset | OpenSGG VG dataloader; h5 test split parity checked for `26446` test images | Residual image-file/path preprocessing parity still needs official-process comparison |
| Detector implementation | Original maskrcnn-benchmark detector modules | Eval-only local adapter with official tensors, official anchors/config/NMS, and pad32 preprocessing | Remaining risk is lower-level ROIAlign/FPN numeric parity |
| Evaluator implementation | Official PENET/SGB evaluator | Local compact evaluator | Semantic parity must be validated by outputs or official process |

## Latest Gap Probes

Two additional 64-image probes were run after the full result and then reverted:

| Probe | R@20/50/100 | mR@20/50/100 | Result | Decision |
|---|---:|---:|---|---|
| Absolute-`xyxy` propagation through detector/relation/union box paths | `20.77 / 24.96 / 28.72` | `6.05 / 6.46 / 8.59` | Mixed versus current smoke row | not adopted |
| Absolute-`xyxy` plus ROIAlign `aligned=False` | `19.16 / 24.41 / 27.07` | `5.96 / 6.57 / 7.65` | Worse on R@K and mR@100 | not adopted |

These are protocol diagnostics only. The remaining full-run gap is still
R@50 `-1.59` points and mR@50 `-0.31` points versus the official table.
