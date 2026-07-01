# Evaluation Protocol: USG-Par

## Official Protocol

- Dataset: PSG for `configs/psg.yaml`
- Split: unresolved
- Task: SGDet
- Preprocessing: OpenCLIP ConvNeXt-L image preprocessing at 320px
- Inference inputs: image modality with class text embeddings
- Evaluator: official USG evaluator, not yet proven equivalent to OpenSGG
- Metrics: SG recall family
- Top-k rules: RPC top-k pairs, default 100
- Constraint settings: unresolved

## Local Protocol

- Dataset: Visual Genome through OpenSGG dataloader
- Split: OpenSGG VG split
- Task: SGDet
- Preprocessing: OpenSGG image preprocessing; OpenCLIP preprocessing parity not yet audited
- Inference inputs: image modality with VG category-name text embeddings when OpenCLIP is available
- Evaluator: OpenSGG `SceneGraphEvaluator`
- Metrics: `sgdet_R@10/20/50`, `sgdet_mR@10/20/50`
- Top-k rules: RPC top-k pairs, default 100
- Constraint settings: OpenSGG evaluator defaults

## Alignment Verdict

mismatch

## Mismatches

| Area | Official | Local | Impact |
|---|---|---|---|
| Dataset | PSG | Visual Genome | Paper numbers are not directly comparable |
| Entity supervision | masks | boxes | Local bbox losses are a necessary VG adaptation, not official PSG mask loss |
| Checkpoint | unresolved | none verified | No reproduction metric can be claimed |
| Evaluator | official USG evaluator | OpenSGG evaluator | Metric parity must be audited before claims |
| Preprocessing | OpenCLIP transform | OpenSGG dataloader path | May affect checkpoint compatibility |
