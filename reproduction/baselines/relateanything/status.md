# Baseline Status: RelateAnything

## Outcome

`implementation_audit` / `not_reproduction_ready`.

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
- Official dataset pack access, full input inventory and scored denominators are pending.
- A3 runner does not consume released embeddings unchanged; deployment calibration
  is not a synonym-matcher tau calibration.
- Exact paper variant/table and code-to-release compatibility are pending.

## Next Action

Complete the checkpoint and evaluation evidence gates before running baseline
metrics.
The input checker is a gate over supplied attestations, not proof of model or
evaluation parity. Its synthetic tests are not baseline results.

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
