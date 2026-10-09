# Issue #110 — VG150 `pair_known` cross-split audit (historical reports preserved)

**Decision:** `MEASURED_FAIL` for the historical `pair_known` fit/evaluation isolation gate, **as reported by the checkout audit** in [Issue #110 comment 6079523851](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/110#issuecomment-6079523851). This PR **versions that reported measurement and a separate read-only check harness**; it did **not** re-execute the 315,642-row VG150 audit because the raw split/data files are excluded from Git and were not supplied to this PR workspace.

## What changed / not changed

- Added `measured_finding.json` (typed and provenance-qualified transcription of the source issue's measurement).
- Added `audit_cross_split.py` and `test_audit_cross_split.py` (stdlib-only portable index/image audit and **5 synthetic fixture tests**).
- **No changes** to historical `outputs/reports/ontology_probe/01_*`, `02_*`, `03_*`, `05_*` or training/evaluation implementation. No retraining, GPU, new SGG model, checkpoint rewriting or audit-data release.
- The scripts were originally tested on **toy fixtures**; their inclusion does **not** turn the issue comment into a second independent real-data reproduction.

## Verified finding as *reported* by the source checkout

| Evaluation cross-check | Shared rows | Shared images |
| --- | ---: | ---: |
| `pair_ood.train` ∩ `pair_known.eval` | 41,626 | 8,271 |
| `pair_ood.dev` ∩ `pair_known.eval` | 4,780 | 901 |
| **actual fitting+validation** ∩ `pair_known.eval` | **46,406 / 62,428 (74.34%)** | **9,172 / 11,520 (79.62%)** |
| Clean remainder | **16,022** | **2,348** |

The historical pipeline fits/calibrates every probe/prior on `pair_ood.train/dev` even in the `pair_known` evaluation arm: [`eval_matrix.py:314–329`](https://github.com/Aveouter/Open_Sence_Graph_Generation/blob/44c21cc1cfaa457979bd2977ac5bbdffa5e79754/tools/ontology_probe/eval_matrix.py#L314-L329) and [`train_probes.py:333–359`](https://github.com/Aveouter/Open_Sence_Graph_Generation/blob/44c21cc1cfaa457979bd2977ac5bbdffa5e79754/tools/ontology_probe/train_probes.py#L333-L359). Auditing `pair_known.train` versus `pair_known.eval` alone misses this cross-split path.

Under the shipped *unordered seen-pair + wrong-argmax* definition, **27,368** `pair_known` historical relations are conflict candidates, but the clean remainder has **6,079** unordered / **4,377** ordered conflict candidates. Under the historical unordered smoothing/calibration, **zero** clean candidates have prior confidence at least approximately 0.9; **281** survive the reported 0.7 threshold. The method's actual strict comparator uses `>` at thresholds; boundary policy and exact `≥` counts must be reconfirmed from output before publication as calibrated high-confidence findings. **Do not call the entire 27,368 cohort high-confidence.**

The contaminated portion's reported conflict fraction is 45.87%, while the clean remainder is 37.94%. This observed difference is **not** a causal estimate of contamination's effect: the clean portion is selected by the pair-OOD holdout design and may differ systematically.

## Reproduce the isolation check in the source checkout

Place the script in or invoke it from the original checkout with:
- `data/VisualGenome/rel.json`
- `outputs/analysis/ontology_probe/splits/split_pair_ood.json`
- `outputs/analysis/ontology_probe/splits/split_pair_known.json`
- Optional: `data/VisualGenome/train.json` to compute directed/undirected pair support
- Optional: `outputs/analysis/ontology_probe/matrix/prior_cells.json` for dev-selected VG50 lookup `alpha`/`temperature`

```bash
python reproduction/audits/issue110_pair_known/audit_cross_split.py \
  --splits-dir outputs/analysis/ontology_probe/splits \
  --rel-json data/VisualGenome/rel.json \
  --train-json data/VisualGenome/train.json \
  --prior-cells outputs/analysis/ontology_probe/matrix/prior_cells.json \
  --output /tmp/issue110_actual_split_audit.json
echo "exit status: $?"  # 3 = measured nonempty train or dev/eval overlap
python -m unittest discover -v \
  -s reproduction/audits/issue110_pair_known -p 'test_*.py'
```

**Fail-closed checks:** the script verifies each split's `relation_order_sha256` against the ordered original `rel.json` relation table. Status 0: no actual train/dev image or row intersections; 3: measured contamination; 2: missing/inconsistent inputs. It measures both train and validation overlap. It **does not** verify the independent SHA256 of the input files against an external full-hash manifest; the original audited checkout stated these hashes matched, but full values are not included in the issue comment.

## Limits and PR acceptance checklist

- [ ] Attach or independently fetch the **full raw-input SHA256 manifest** from the reporting checkout, not just abbreviated hashes.
- [ ] Re-run the committed verifier on the same inputs and commit its **complete output JSON** or privacy-safe structured derived result (hashes, counts and protocol IDs), with raw VG data remaining untracked.
- [ ] Cross-check every reported cell, including `pair_known` clean cohort numbers and confidence threshold inclusivity. The present helper checks raw index intersections and conflicts on the full evaluation set; it does **not** itself compute every clean-subset statistic reported in the issue.
- [ ] Verify alignment of **saved B2/B3/B4 predictions** with clean row/image indices, then compute corrected held-out metrics **in a subsequent reviewable change**; **no model retraining** by default.
- [ ] Preserve the initial source's exact provenance limitations: archived `training_subset` stamps and HEAD code agree, but historical command invocation / complete executed source were not independently reconstructed.
- [ ] Investigate potential **selection bias** (the clean cohort is not a random held-out split); compute independent-image confidence intervals and stratified predicate counts.
- [x] Preserve all historical reports unchanged and link this audit as the correction.

**Do not merge until enough of the original measurement provenance is captured for a second person to audit the load-bearing numbers.** Leaving this PR in **Draft** is intentional.

Decision: #110 remains OPEN. #111 evidence taxonomy is refined; #122 and #123 remain OPEN, #112 DEFERRED, and new large-scale GPU work is not authorized.
