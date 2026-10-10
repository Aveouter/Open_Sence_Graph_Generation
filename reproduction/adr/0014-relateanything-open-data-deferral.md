# ADR 0014: Defer the RelateAnything A1 baseline on unavailable official evaluation data

## Status

Accepted 2026-10-10. Extends ADR 0012 and ADR 0013.

## Context

The port is verified (PR #128: byte-identical source, strict load, numerical
parity against the official API) and the evaluation environment was configured
on 2026-10-10: CUDA torch in the isolated runtime, strict-load attestation
produced, official runner importing, GPU inference demonstrated. The remaining
prerequisite for the checkpoint-backed A1 baseline is the official VG150
evaluation pack.

That pack is not publicly obtainable. Measured 2026-10-10:

- `https://huggingface.co/datasets/maelic/OV-SGG-Bench` returns 401 to
  anonymous API and page requests, while the same endpoints return 200 for the
  author's public datasets. That public listing contains PSG-coco-format,
  IndoorVG-coco-format, VG150-coco-format, GQA200-coco-format and RA-4M --
  OV-SGG-Bench is absent, and a site-wide search for "OV-SGG" returns zero
  datasets (no mirror or fork).
- `maelic/RA-4M` holds only `packs/megasg/{train,val}` (training packs).
- `Maelic/RelateAnything` publishes no releases; `Maelic/SGG-Benchmark`
  releases are REACT++ only.
- The in-repo VG150 copy (`data/VisualGenome`, gitignored, images complete) is
  not a substitute. Its test split lists 26,446 images, every one with >=2
  boxes and >=1 relation, while the released pack's audited denominator is
  26,404. The 42-image difference matches no packer rule (`min_rels` is 0 for
  test and boxes are never dropped), and the public `maelic/VG150-coco-format`
  parquet reports a third count (test 31,876 rows). Three sources, three
  denominators: a locally rebuilt pack would evaluate a different dataset than
  the released artifact.

Environment gate, recorded because it shaped the attempt: Microsoft Smart App
Control began blocking the isolated runtime's torch DLLs mid-work
(`WinError 4551`, CodeIntegrity event 3118). Installing the CUDA build
(`torch==2.14.1+cu130`, from the documented wheel index) produced files SAC
allowed; that build also matches the official runner's CUDA/bf16 precision
policy. SAC verdicts are per-file and may flip again, so long runs re-verify
`import torch` first. Disabling SAC (irreversible without a Windows reset) or
moving to WSL2 was declined as unnecessary once CUDA worked.

## Decision

Record `deferred_reproduction` for the RelateAnything baseline. Do not run a
substituted pack, do not present a local-split evaluation as the official
protocol, and do not claim any metric. Reopen when the official pack is
obtainable again -- author access restored, a mirror published, or the author
naming a replacement location -- and then run the documented flow unchanged.

## Evidence

- Blocker measurements: the HTTP/API checks above, reproducible with `curl`
  against `huggingface.co/api/datasets/...` (commands recorded in the PR body).
- Strict-load attestation, external and re-derivable with
  `tools/reproduction/produce_relateanything_evidence.py`:
  `D:\Code\RelateAnything_evidence\relateanything_evidence.json`, sha256
  `e29f460709a1c963345a64950e3b4235ca8344fbf94d63d9484fa1c1d786a32e` -- the four
  mismatch lists all empty, `used_ema_model=false` (this release stores EMA
  weights under `model`), snapshot revision
  `2db90096be5217bdc7a9003c042950f45723d105`, and the six pinned source hashes.
- A3 stays deferred for its own reasons: matcher tau calibration is missing and
  the released NPZ layout is incompatible with the official runner (ADR 0012).

## Consequences

- Cannot be claimed: reproduction, paper parity, or any A1/A3 metric.
- Can be claimed: the implementation audit, strict-load and inference parity,
  and the input-gate design, all already recorded.
- Next: restore data access. The prepared pipeline (pack -> link images ->
  full-split audit -> input gate -> official A1 -> adapter leg -> comparison)
  then runs as planned, with the +-0.005 comparison tolerance preregistered
  before any number is seen.
