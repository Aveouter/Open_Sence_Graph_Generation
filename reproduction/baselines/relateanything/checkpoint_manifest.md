# Checkpoint Manifest: RelateAnything ViT-S/16+

## Identity

- Source URL: https://huggingface.co/maelic/relsgg-vits16plus
- Source owner: maelic / Maëlic Neau.
- Target snapshot: `2db90096be5217bdc7a9003c042950f45723d105`.
- File/path: acquired externally at `<external-assets>/vits16plus`;
  weight files are excluded from the repository.
- Required A1 assets: `model.pth`, `text_student.pt`, `tokenizer.json`,
  `tokenizer_config.json`; retain the release config and notices for provenance.
- A3 additionally requires `predicate_embeddings.npz` and a matcher calibration
  explicitly bound to that text space. Released `calibration.json` calibrates
  deployment scores and cannot substitute for the matcher record.
- Download: Hugging Face `snapshot_download`, 2026-10-09; local SHA256 below.
- Provenance lead: release manifest names source
  `e9ea42aed60f766f12ad19d51709129c50110a3b`; compatibility with the issue's
  pinned official source must be established, not assumed.

## Expected Compatibility

| Asset | SHA256 |
|---|---|
| model.pth | `5d281bf0d89f2bbfd72ff5a14f9a40ce12534e790b0402e2ca970539c7bcc294` |
| text_student.pt | `e0317830b68ea51e6711fc90d4a35954d0528e5bd78a8d5afd966601ce4ed119` |
| tokenizer.json | `6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35` |
| tokenizer_config.json | `5beb842134088c9dd96b796b748bd612bd51b3649736cc465e3f300fb3da8084` |
| predicate_embeddings.npz | `e85ce8690df07771d0db8139281e4b364fce79423973e7ab2ef41fcffef332f4` |

Hashes were computed from acquired files using SHA256, not copied from Hub metadata.

- Method: complete official RelateAnything / RelSGG relation model.
- Architecture: released DINOv3 ViT-S/16+ configuration embedded in checkpoint.
- Dataset/task: official VG150 pack, test, supplied GT regions without object
  label model inputs; target closed vocabulary for A1.
- Config: official checkpoint args, EMA selection and official preprocessing.
- Object classes: not an inference classifier input; annotations stay in evaluator.
- Predicate classes/order: pack vocabulary for A1; exact checkpoint training
  vocabulary and embedding order for A3; no invented background channel.
- Text encoder: released student; do not substitute CLIP or another text space.

## Load Report

- Loader: integration test `test_release_checkpoint_predicts_the_same_triplets_as_official_api`
  in the isolated Python 3.13 runtime, against Apache source parent in ADR 0013.
- Strict loader: official `RelateAnything.from_checkpoint(..., strict=True)`.
- Both strict loaders completed; nonempty predictions, scores, masks,
  decomposition and vocabulary switching matched.
- Local state inspection (2026-10-09) selected `model`: this release has no
  `ema_model` entry, so the official `weights="ema"` preference falls back to
  its released `model` state. Do not infer its pre-stripping EMA provenance.
- Missing keys, unexpected keys and shape mismatches: all empty arrays.
  Obsolete keys removed: empty array. No adapter key remapping was applied.
- Released predicate count: 19,103, matching the sidecar order. NumPy stores
  `names` as an object array; the official API enables pickle for that trusted
  release bank. A3 port loading does likewise after the input gate.
- Official loader removes documented obsolete keys and handles optional `W_obj`;
  report these exceptions explicitly rather than hiding them as adapter remaps.
- Verdict: **strict-load and public inference parity verified**; full protocol
  compatibility and baseline reproduction remain unverified.

## Sources

- [Pinned checkpoint loader](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/checkpoint.py)
- [Pinned API](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/relsgg/api.py)
- [Pinned release manifest](https://github.com/Maelic/RelateAnything/blob/06766fdf56752ca535fc9b971fca99ce563676d0/deploy/dist/relsgg-vits16plus/relateanything.json)
