# Baseline Reproduction Issue: USG-Par

## Baseline

- Method: USG-Par
- Paper: Universal Scene Graph Parser
- Official repository: https://github.com/ChocoWu/USG
- Official commit/tag: unresolved
- Target dataset/split: Visual Genome in OpenSGG for local baseline work; official image config targets PSG
- Target task: SGDet
- Target metrics: `sgdet_R@10/20/50`, `sgdet_mR@10/20/50`

## Official Evidence

- Paper table/section: unresolved
- Official config: `configs/psg.yaml`
- Official inference command: unresolved
- Official evaluator: official `eval.py` / SG recall evaluator, not yet proven identical to OpenSGG evaluator
- Reported checkpoint: unavailable in this audit

## Checkpoint Evidence

- Checkpoint path or URL: unavailable
- Source owner: unavailable
- Download date: unavailable
- SHA256: unavailable
- Expected architecture: USG-Par image path
- Expected config: official PSG config plus Visual Genome adaptation decision
- Compatibility verdict: unavailable

## OpenSGG Candidate Implementation

- Local files:
  - `src/models/usg.py`
  - `src/methods/usg_method.py`
  - `src/core/metrics.py`
  - `configs/VisualGenome/USG.py`
- Claimed method name: USG-Par candidate implementation
- Differences from official implementation:
  - Official image path supervises masks on PSG; OpenSGG Visual Genome path uses boxes.
  - Official code is universal/multimodal; local baseline currently uses image-only SGDet.
  - Official checkpoint/config/evaluator parity is not verified.
- Required code fixes:
  - Open-vocabulary cosine object classifier with no-object embedding.
  - Sigmoid BCE object classification and sigmoid-class Hungarian cost.
  - Multi-hot predicate targets with BCE loss.
  - Weighted BCE pair-confidence loss.
  - Official RPC pair selection and two-way RAC update order.
  - USG-specific SGDet metric adapter for sigmoid scores and VG label offsets.

## Decision Records

- ADRs required before claiming reproduction:
  - `reproduction/adr/0003-usg-par-official-algorithm-alignment.md`
- ADRs required before deferring reproduction:
  - checkpoint/protocol ADR if official weights or evaluator remain unavailable
- Existing ADR links:
  - `reproduction/adr/0001-reproduction-claim-standard.md`
  - `reproduction/adr/0002-random-init-is-not-reproduction.md`

## Protocol Alignment Checklist

- [ ] dataset split matches official protocol
- [ ] object labels/classes match official protocol
- [ ] predicate labels/classes match official protocol
- [ ] preprocessing matches official protocol
- [ ] inference mode matches official protocol
- [ ] evaluator semantics match official protocol
- [ ] metric keys match official protocol
- [ ] checkpoint loads without unexplained missing or unexpected keys

## Success Conditions

- Official checkpoint identity and SHA256 are recorded.
- Official config and preprocessing are mapped exactly or every adaptation is justified.
- Evaluator semantics are shown to match the target paper/official repository.
- A checkpoint-backed SGDet run is reported with non-random weights.

## Failure / Defer Conditions

- No official checkpoint or no verified compatible checkpoint.
- PSG mask protocol cannot be compared to Visual Genome box protocol.
- Evaluator semantics differ and cannot be reconciled without changing benchmark rules.

## Current Outcome

`implementation_audit`

## Notes

- Do not report random-init, tiny-slice, or all-zero metrics as USG-Par reproduction.
