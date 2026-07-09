# Baseline Reproduction Issue: PENet SGDet

## Baseline

- Method: PE-NET / PENet
- Paper: Prototype-based Embedding Network for Scene Graph Generation
- Official repository: https://github.com/VL-Group/PENET
- Official commit/tag: `9c9f50777c66647799cb7a17dfb855aa98aecd1e`
- Target dataset/split: Visual Genome test split used by PENET/SGB
- Target task: SGDet
- Target metrics: R@20/50/100 and mR@20/50/100

## Official Evidence

- Paper/README table: SGDet R@20/50/100 `23.36 / 30.41 / 34.84`, mR@20/50/100 `9.16 / 12.25 / 14.34`
- Official config: `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- Official inference command: `tools/relation_test_net.py` with `MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork`
- Official evaluator: PENET/SGB Visual Genome relation evaluator
- Reported checkpoint: Google Drive id `1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C`

## Checkpoint Evidence

- Checkpoint path or URL: `outputs/pretrained/penet_official/PE-NET_SGDet/model_final.pth`
- Source owner: VL-Group/PENET
- Download date: local file timestamp `2023-04-02 09:15:21 +0000`
- SHA256: `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183`
- Expected architecture: `PrototypeEmbeddingNetwork` with ResNeXt-101-FPN detector
- Expected config: `e2e_relation_X_101_32_8_FPN_1x.yaml`
- Compatibility verdict: partial, with tensor-level relation/detector extractor parity covered by tests

## OpenSGG Candidate Implementation

- Local files:
  - `src/methods/penet_method.py`
  - `src/models/penet.py`
  - `src/models/penet_detector.py`
  - `utils/penet_weights.py`
  - `src/core/metrics.py`
- Claimed method name: `PENet`
- Differences from official implementation:
  - OpenSGG still uses its own Lightning/dataloader wrapper.
  - SGDet proposal generation is an eval-only local adapter for the official detector tensors.
  - The local evaluator computes official-style SGDet recall from compact PE-NET outputs, but it is not the original PENET process.
- Required code fixes:
  - Keep detector and relation ROI box extractors separate.
  - Load relation extractor, union extractor, detector RPN, detector box predictor, and detector box extractor from the SGDet checkpoint.
  - Remap official `roi_heads.relation.predictor.*` tensors into the local PE-NET predictor.
  - Guard relation-score background column handling when epoch-end passes `rel_nums=50`.
  - Add tensor parity tests for official relation/detector extractor keys.

## Decision Records

- ADRs required before any success-status change: a final SGDet parity decision after full Visual Genome metrics and official-run comparison.
- ADRs required before deferring reproduction: record remaining proposal/evaluator/data-protocol mismatches if full metrics stay below the official table.
- Existing ADR links: `reproduction/adr/0004-penet-sgdet-pr84-adapter.md`

## Protocol Alignment Checklist

- [ ] dataset split matches official protocol
- [x] object labels/classes match official protocol
- [x] predicate labels/classes match official protocol
- [x] preprocessing matches official detector BGR mean/std flow inside PE-NET method
- [x] inference mode matches SGDet image-only proposal flow
- [ ] evaluator semantics match official protocol exactly
- [x] metric keys match official SGDet R/mR@K
- [ ] checkpoint loads without unexplained missing or unexpected keys

## Success Conditions

- Full Visual Genome SGDet metrics are comparable to the official table under the same checkpoint, config, preprocessing, proposal flow, and evaluator semantics.
- Tensor-level tests prove relation ROI extractor and detector ROI extractor tensors equal their official checkpoint keys.
- The report distinguishes smoke tests from baseline evidence.

## Failure / Defer Conditions

- Full SGDet R@50/mR@50 remains materially below the official `30.41 / 12.25` table row.
- Any remaining data preprocessing, detector proposal, or evaluator semantic mismatch is not resolved.

## Current Outcome

`checkpoint_backed_evaluation`

## Notes

- The 64-image smoke run is diagnostic only and is not baseline evidence.
- Full VG SGDet eval with official overlap-pair filtering completed with
  R@50 `28.40` and mR@50 `11.50`, below the official `30.41` and `12.25`.
- The result is close enough to continue parity diagnostics, but this issue does
  not mark PE-NET SGDet as reproduced.
