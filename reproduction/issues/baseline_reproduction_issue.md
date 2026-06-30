# Baseline Reproduction Issue: <METHOD>

## Baseline

- Method:
- Paper:
- Official repository:
- Official commit/tag:
- Target dataset/split:
- Target task: PredCls / SGCls / SGDet / other
- Target metrics:

## Official Evidence

- Paper table/section:
- Official config:
- Official inference command:
- Official evaluator:
- Reported checkpoint:

## Checkpoint Evidence

- Checkpoint path or URL:
- Source owner:
- Download date:
- SHA256:
- Expected architecture:
- Expected config:
- Compatibility verdict: trusted / partial / rejected / unavailable

## OpenSGG Candidate Implementation

- Local files:
- Claimed method name:
- Differences from official implementation:
- Required code fixes:

## Decision Records

- ADRs required before claiming reproduction:
- ADRs required before deferring reproduction:
- Existing ADR links:

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

- Required evidence before calling this reproduced:

## Failure / Defer Conditions

- Conditions that force `deferred_reproduction`:

## Current Outcome

Choose one:

- `reproduction_ready`
- `checkpoint_backed_evaluation`
- `implementation_audit`
- `pipeline_smoke_only`
- `checkpoint_unavailable`
- `protocol_mismatch`
- `deferred_reproduction`

## Notes

- Do not use random-init or tiny-slice zero metrics as reproduction evidence.
