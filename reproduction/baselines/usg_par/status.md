# Baseline Status: USG-Par

## Outcome

`implementation_audit`

## Claim Boundary

This status records algorithm-alignment work against the official USG repository.
It is not a reproduction result because checkpoint provenance, official evaluator
semantics, official split/preprocessing, and Visual Genome versus PSG protocol
alignment remain unresolved.

## Evidence Summary

| Requirement | Evidence | Verdict |
|---|---|---|
| Official paper identified | USG-Par / Universal Scene Graph Parser | partial |
| Official repository identified | https://github.com/ChocoWu/USG | present |
| Checkpoint provenance verified | No checkpoint manifest yet | missing |
| Config aligned | Local config mirrors core official model/loss knobs, with VG bbox adaptation | partial |
| Inference flow aligned | Local SGDet adapter now uses USG sigmoid semantics | partial |
| Evaluator semantics aligned | OpenSGG evaluator not proven identical to official evaluator | missing |
| Metrics comparable to paper | Dataset/protocol/checkpoint mismatch unresolved | missing |

## Current Blocker

- Official checkpoint identity and evaluation protocol parity are not established.
- Official image-path supervision is mask-based PSG; OpenSGG local target is VG boxes.

## Next Action

- Obtain official checkpoint/protocol evidence or keep the baseline at
  `implementation_audit`.
- Run only smoke/shape checks until checkpoint/config/evaluator compatibility is verified.
