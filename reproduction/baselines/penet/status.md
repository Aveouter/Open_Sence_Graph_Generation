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
| Config aligned | Local `configs/VisualGenome/PE_NET.py` targets SGDet and official X-101-FPN hyperparameters, including `TEST.RELATION.REQUIRE_OVERLAP=False` | present |
| Inference flow aligned | Eval-only detector proposal adapter loads RPN/box predictor, relation predictor loads from official SGDet keys, relation ROI features are re-extracted, and BGR255 pad32 preprocessing is applied | partial |
| Evaluator semantics aligned | PE-NET SGDet compact evaluator uses global top-K relation ranking | partial |
| Metrics comparable to official table | Full VG SGDet R@50 `0.2882086932659149`, mR@50 `0.11937950551509857` vs official `0.3041`, `0.1225` | close but below target |

## Current Gap

| Gap | Current evidence | Status |
|---|---|---|
| R@50 vs official table | Full VG R@50 `28.82` vs official `30.41`, gap `-1.59` points | open |
| mR@50 vs official table | Full VG mR@50 `11.94` vs official `12.25`, gap `-0.31` points | open |
| Official process parity | Original PENET/SGB process has not been run side-by-side from the same local inputs | open |
| Absolute-`xyxy` proposal propagation | 64-image smoke produced mixed R/mR movement | checked, not adopted |
| `xyxy` plus ROIAlign `aligned=False` | 64-image smoke was worse on R@K and mR@100 | checked, not adopted |

## Next Action

| Priority | Diagnostic | Rationale |
|---|---|---|
| P0 | Official-process side-by-side comparison | Compare detector proposals, ROI features, relation scores, and evaluator matches from the same local images/checkpoint before another full run. |
| P1 | ROIAlign/FPN tensor-level parity | Treat numeric parity as a tensor-level test target rather than a smoke-only code change. |
