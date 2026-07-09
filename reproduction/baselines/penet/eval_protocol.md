# Evaluation Protocol: PENet SGDet

## Official Protocol

- Dataset: PENET/SGB Visual Genome
- Split: official VG test split
- Task: SGDet
- Preprocessing: BGR image tensor with detector pixel mean/std and 600/1000 eval resize
- Inference inputs: raw image only; detector proposals, object labels, boxes, and predicates predicted by the model
- Evaluator: PENET/SGB Visual Genome SGDet evaluator
- Metrics: R@20/50/100 and mR@20/50/100
- Top-k rules: relation triplets sorted by relation score times subject/object scores
- Constraint settings: graph-constrained SGDet

## Local Protocol

- Dataset: OpenSGG Visual Genome dataloader for the current smoke/full runs
- Split: OpenSGG VG test split
- Task: SGDet
- Preprocessing: PE-NET method-local BGR conversion and detector normalization
- Inference inputs: raw image, with eval-only detector proposal adapter
- Evaluator: PE-NET-specific OpenSGG compact SGDet evaluator using `evaluate_recall`
- Metrics: `sgdet_R@20/50/100` and `sgdet_mR@20/50/100`
- Top-k rules: overlapping detected-box relation pairs sorted by relation score times subject/object scores, capped at top 100
- Constraint settings: graph-constrained SGDet

## Alignment Verdict

partial

The current adapter is suitable for checkpoint-backed SGDet evaluation, but the
full run remains below the official table: R@50 `28.40` vs `30.41`, mR@50
`11.50` vs `12.25`. The remaining protocol work is therefore treated as a
parity gap, not as a successful reproduction.

## Mismatches

| Area | Official | Local | Impact |
|---|---|---|---|
| Runtime process | Original PENET/SGB test script | OpenSGG Lightning test wrapper | May affect data ordering, transforms, and evaluator aggregation |
| Dataset interface | PENET/SGB h5/json dataset | OpenSGG VG dataloader for current smoke/full runs | Needs side-by-side validation before table comparison |
| Detector implementation | Original maskrcnn-benchmark detector modules | Eval-only local adapter with official tensors | Proposal parity must be tested beyond tensor loading |
| Evaluator implementation | Official PENET/SGB evaluator | Local compact evaluator | Semantic parity must be validated by outputs or official process |
