# Checkpoint Manifest: RA-SGG PredCls

## Identity

- File/path: `checkpoints/RASGG/model_final.pth`
- Source URL: not recorded locally
- Source owner: presumed RA-SGG authors; provenance is not yet verified
- Download command: unknown
- Download date: unknown
- SHA256: `2488946f213686cad43a06f718735067f02a9fcbf966e2108cccfb1cd0edd8db`
- File size: `2,718,467,375` bytes
- Container: `model` (648 tensors), optimizer state, iteration `60000`
- Bundled config: `checkpoints/RASGG/config.yml`
- Bundled reference result: `checkpoints/RASGG/result.txt`

## Expected Compatibility

- Method: RA-SGG / ReTAG
- Architecture: ResNeXt-101-32x8d FPN plus RA-PENet
- Dataset: Visual Genome VG150
- Task: PredCls
- Config predictor name: `RA-PENetCorrectProtoBeta`
- Number of object classes: 151 including background
- Number of predicate classes: 51 including background

## Load Report

- Loader path: `BaseExperiment.test` external `.pth` routing through
  `RASGGModel.remap_external_state_dict` and the separate visual extractor.
- Current run report:
  `outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715/checkpoint_load_report.json`.
- Key normalization/remap: 643/648 checkpoint tensors; exact-shape load:
  637/648 checkpoint tensors.
- Model/predictor tensors: 63 matched.
- Visual extractor tensors: 574 matched, including the two all-level union
  `reduce_channel` tensors.
- Element coverage: 411,581,390 / 436,755,486 (`94.236%`).
- Shape mismatches: none among the 637 mapped tensors.
- Unmapped predictor tensors: six `fusion_attn_{q,k,v}` weight/bias tensors.
  They are defined by the checkpoint source variant but are not used by the
  audited public ReTAG forward path.
- Unmapped detector buffers: five RPN anchor buffers; irrelevant to PredCls.
- Local model-only entries: six deterministic/runtime buffers
  (`head_ids`, `body_ids`, `tail_ids`, `rel_loss_weight`, `fb_keys`, and
  `fb_values`); they are not checkpoint predictor weights. BatchNorm counters
  are also supplied by the local runtime where needed.
- Shape adaptation: none.
- Verdict: partial.

## Completed Evaluation

The checkpoint-backed full-test candidate completed at
`outputs/runs/ra_sgg/2026-07-15_rasgg_predcls_h5_repro_20260715` with:

- R@20/50/100:
  `0.5547371506690979 / 0.6218554973602295 / 0.6410786509513855`
- mR@20/50/100:
  `0.2872789204120636 / 0.36152592301368713 / 0.3910185992717743`
- Bundled official R@20/50/100: `0.5546 / 0.6217 / 0.6410`
- Bundled official mR@20/50/100: `0.2873 / 0.3615 / 0.3910`
- Maximum absolute delta: `0.0001554974`

The exact outputs are persisted in `eval/summary.json`,
`eval/predcls/metrics.json`, and `lightning/metrics.csv`; runtime and load
evidence are in `metadata.json` and `checkpoint_load_report.json` in the same
run directory. Their exact hashes and the six-metric vector are tracked in
`reproduction/evidence/ra_sgg/predcls_candidate_2026-07-15.json` so the PR does
not depend solely on ignored local output files.

The metadata reports Python 3.10.8, PyTorch 2.5.1+cu121, one RTX 2080 Ti, and a
dirty source tree. PyTorch/torchvision are supplied through a temporary
compatibility overlay on the `hsg` Python/Lightning environment. The extremely
close metric match supports checkpoint compatibility, but this remains an
implementation audit because the runtime is not clean/pinned and the original
predictor revision and checkpoint provenance remain incomplete.

## Historical v14 Load

The v14 code path did not route the external checkpoint into the separate visual
extractor and did not implement the configured all-level union reduction. Its
logged metrics therefore cannot establish checkpoint-compatible evaluation.

## Notes

The unmapped attention tensors have an explained forward-path impact of zero in
the pinned public source, but the predictor class named in the bundled config is
not present at that pinned commit. Until the original source revision and file
provenance are recovered, this manifest is not `trusted`.
