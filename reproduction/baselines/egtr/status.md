# Baseline Status: EGTR

## Outcome

`checkpoint_backed_evaluation`

## Claim Boundary

Official EGTR SGDet full Visual Genome evaluation completed with the locally
downloaded official checkpoint. The local OpenSGG wrapper was audited against
the official repository and adjusted to match the official single-predicate
SGDet evaluator semantics before reporting metrics.

This status covers evaluation only. No training was run.

## Key Result

| Metric | Local full VG DDP | Official archive reference | Delta |
|---|---:|---:|---:|
| SGDet R@50 | 0.3008347452 | 0.3021079916 | -0.0012732464 |
| SGDet mR@50 | 0.0792875737 | 0.0793714169 | -0.0000838432 |

## Evidence

- Run: `outputs/runs/egtr/2026-07-08_EGTR_official_parity_full_vg_ddp6`
- Metrics: `outputs/runs/egtr/2026-07-08_EGTR_official_parity_full_vg_ddp6/eval/sgdet/metrics.json`
- Parity report: `reproduction/baselines/egtr/parity_report.md`
- Checkpoint manifest: `reproduction/baselines/egtr/checkpoint_manifest.md`
- Evaluation protocol: `reproduction/baselines/egtr/eval_protocol.md`

## Notes

- Local metrics use OpenSGG's 1-indexed VG label convention internally and bridge
  to EGTR's 0-indexed official logits at the adapter boundary.
- The official archive reference reports `(single)R@*` and `(single)mR@*`; the
  local report maps these to constrained single-predicate `sgdet_R@*` and
  `sgdet_mR@*`.
