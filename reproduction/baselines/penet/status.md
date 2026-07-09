# Baseline Status: PENet SGDet

## Outcome

`checkpoint_backed_evaluation`

## Claim Boundary

This is not a reproduction-success report. The SGDet path now loads the official
checkpoint and passes tensor-level adapter tests. The full Visual Genome run is
close to the official PENET table but remains below it, so the status stays at
checkpoint-backed evaluation rather than reproduced.

## Evidence Summary

| Requirement | Evidence | Verdict |
|---|---|---|
| Official paper identified | PE-NET / Prototype-based Embedding Network for SGG | present |
| Official repository identified | VL-Group/PENET at `9c9f50777c66647799cb7a17dfb855aa98aecd1e` | present |
| Checkpoint provenance verified | SGDet `model_final.pth`, SHA256 `ca7009b404f845ed989f799d1dc426be28a33db89b6b8d16649309d338948183` | present |
| Config aligned | Local `configs/VisualGenome/PE_NET.py` targets SGDet and official X-101-FPN hyperparameters | partial |
| Inference flow aligned | Eval-only detector proposal adapter loads RPN/box predictor, relation predictor loads from official SGDet keys, and relation ROI features are re-extracted | partial |
| Evaluator semantics aligned | PE-NET SGDet compact evaluator uses global top-K relation ranking | partial |
| Metrics comparable to official table | Full VG SGDet R@50 `0.2839694917201996`, mR@50 `0.11496535688638687` vs official `0.3041`, `0.1225` | close but below target |

## Current Gap

- Full Visual Genome SGDet R@50 is about `2.01` points below the official table.
- Full Visual Genome SGDet mR@50 is about `0.75` points below the official table.
- Official PENET process has not been run side-by-side from the same local inputs.

## Next Action

- Continue with detector proposal, data-interface, and evaluator parity diagnostics
  before upgrading this baseline to a reproduction claim.
