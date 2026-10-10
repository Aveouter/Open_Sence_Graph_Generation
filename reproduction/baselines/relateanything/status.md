# Baseline Status: RelateAnything

## Outcome

`implementation_audit` / `deferred_reproduction`. The port audit stands on its
own evidence; the checkpoint-backed A1 run is deferred because the official
evaluation pack is not publicly obtainable (see Data Access Blocker below and
[ADR 0014](../../adr/0014-relateanything-open-data-deferral.md)).

Issue: [#124](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/124).
The requested work is a complete OpenSGG integration of the released relation
model, not a replacement architecture or a random-initialization demonstration.

## Evidence Summary

| Requirement | Evidence | Verdict |
|---|---|---|
| Official paper identified | arXiv:2609.12552v1; exact result table still to map | partial |
| Official repository identified | Maelic/RelateAnything at `06766fdf56752ca535fc9b971fca99ce563676d0` | identified |
| Checkpoint provenance verified | External snapshot hashes recorded; strict loaders and empty key/shape mismatch reports verified | local release compatibility verified |
| Config aligned | Source audit identifies defaults; no executed reference | partial |
| Inference flow aligned | Released weights produce matching nonempty triplets and scores against official API; vocabulary switching checked | measured public API parity |
| Evaluator semantics aligned | A1/A3 aggregation and release-layout gaps recorded | partial |
| Metrics comparable to paper | No local full-split checkpoint-backed run | missing |

## Current Blockers

- Apache parent source and isolated runtime are recorded in ADR 0013; full data/protocol evidence remains pending.
- **Official evaluation pack is not publicly obtainable** (2026-10-10); full
  input inventory and scored denominators are consequently blocked.
- A3 runner does not consume released embeddings unchanged; deployment calibration
  is not a synonym-matcher tau calibration.
- Exact paper variant/table and code-to-release compatibility are pending.

## Data Access Blocker (2026-10-10)

The evaluation environment is configured (CUDA torch in the isolated runtime,
strict-load attestation produced, official runner importing, GPU inference
demonstrated), but the official VG150 evaluation pack cannot be downloaded:

- `maelic/OV-SGG-Bench` returns 401 to anonymous API and page requests while
  the same endpoints return 200 for the author's public datasets; that listing
  (PSG/IndoorVG/VG150/GQA200-coco-format, RA-4M) does not include it, and no
  mirror exists (site-wide search returns zero datasets).
- `maelic/RA-4M` carries only `packs/megasg/{train,val}`; the GitHub repos
  publish no data releases.
- The in-repo VG150 copy is not a substitute: its test split has 26,446 images
  (all with >=2 boxes and >=1 relation) against the pack's audited 26,404, and
  the public `maelic/VG150-coco-format` parquet reports 31,876 test rows --
  three sources, three denominators, none reproducible from the others.

Strict-load attestation produced before the blocker was hit (external,
re-derivable with `tools/reproduction/produce_relateanything_evidence.py`):
`D:\Code\RelateAnything_evidence\relateanything_evidence.json`, sha256
`e29f460709a1c963345a64950e3b4235ca8344fbf94d63d9484fa1c1d786a32e` -- the four
mismatch lists all empty, `used_ema_model=false`, snapshot revision
`2db90096be5217bdc7a9003c042950f45723d105`, six pinned source hashes.

## Next Action

Reopen when the official pack is obtainable again (author access restored, a
mirror published, or a replacement location confirmed). The prepared pipeline
then runs unchanged: pack -> hardlink images -> full-split audit -> input gate
-> official A1 -> adapter leg -> comparison against the 0.533 / 0.282 / 0.369
reference within a preregistered +-0.005 tolerance. The input checker is a gate
over supplied attestations, not proof of model or evaluation parity. Its
synthetic tests are not baseline results.

## Port Verification (2026-10-09)

The method registry and main CLI expose `RelateAnything` for supplied-region
prediction. The Apache source parent is pinned by `UPSTREAM.json`; model source
changes are rejected before loading weights. Evaluation uses a separate metric
namespace, rejects subsets and lost denominators, and requires provenance gates.

47 local integration/input/registry tests passed in the isolated Python 3.13
runtime, including strict loading of the released ViT-S/16+ weights and numerical
triplet parity with the official API. Earlier, 37 official tests passed against
the ported package, including training losses and gradients. These are code
verification results, not full-split baseline evaluation. CPU was used; CUDA
execution and full benchmark metrics remain unverified.

These include full released vocabulary and NPZ object-array loading, A3 loop
comparison with the official soft evaluator, official-format pack loading,
main CLI prediction and CI runtime routing. A3 benchmark text encoding uses the
official training template ensemble. 19 stable characterization tests passed in
the existing OpenSGG environment. Bare-Python port/input tests passed with 13
optional tests skipped explicitly. Aggregate reproduction guardrails remain failing on the
pre-existing extraction tip mismatch and the legacy environment's SciPy DLL
application-control error. Independent documentation, claim, boundary, registry
validation and changed-file lint checks passed; aggregate success is not claimed.

## Code Port Acceptance

| Deliverable | Verified evidence |
|---|---|
| Complete relation network, text student, region encoding, scoring and losses | Source files compared to the Apache Git revision; official training/gradient tests pass |
| Registry, config and main CLI | Registry tests, CPU prediction subprocess and independent-runtime class resolution pass |
| Dynamic vocabulary, masks, confidence and graph decomposition | Released-checkpoint triplet and score comparisons against official API pass |
| A1/A3 evaluation integration | Packed-batch loop and evaluator comparisons pass; input provenance/full-split gates are enforced |
| Dependency/license boundary | Optional requirements file, component license/notices, source manifest and ADR 0013 |
| Reproduction claim boundary | Documentation/claims/boundary checks pass; full-split reproduction remains pending |

The code port is accepted on this evidence. This does not close the full
reproduction issue: official data access, matcher calibration, exact paper-table
mapping, complete reference/local split comparison and CUDA verification remain.

## Claim Boundary

No reproduced result, supported training recipe, SGDet result or standard PredCls
result is claimed. See [implementation audit](implementation_audit.md),
[checkpoint manifest](checkpoint_manifest.md), [evaluation protocol](eval_protocol.md)
and [task contract ADR](../../adr/0012-relateanything-region-input-contract.md).
