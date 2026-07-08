# FREQ Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#59](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/59))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- Existing evidence is pipeline/export smoke only.
- Reopen only after paper/repo-aligned FREQ protocol is verified.

Completed phases:

- Phase 0: status audit
- Phase 1: official source audit
- Phase 2: OpenSGG code inventory
- Phase 3: integration plan
- Phase 4: minimal code integration
- Phase 5: smoke test
- Phase 6: reproduction code review
- Phase 7: standard metric validation
- Phase 8: hidden-positive evaluation adaptation
- Phase 9: deviation report
- Phase 10: final reproduction report

Last command:

```bash
python tools/analysis/export_relation_predictions.py --method FREQ --model FREQ --task PredCLS --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/freq/hidden_export && python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/freq/hidden_export/relation_predictions.jsonl --output_dir outputs/reproduction/freq/predicate_recall --ks 1 5 10
```

Last result:

- Relation JSONL validation passed.
- Exported 2 GT relation rows from a 1-batch, 2-image cap.
- GT-aligned predicate recall summary: R@1 0.0, R@5 1.0, R@10 1.0; mR@1 0.0, mR@5 1.0, mR@10 1.0.
- FREQ official-prior input check is currently blocked because SGB-format
  `VG-SGG-with-attri.h5` is missing; see `sgb_freq_input_check.json`.

Next action: provide or locate SGB-format `VG-SGG-with-attri.h5`, export
official SGB `statistics['pred_dist']`, then run
`tools/reproduction/compare_freq_prior.py` as described in
`11_evidence_gate_audit.md`. Keep FREQ deferred until prior-table parity and
benchmark-scale PredCls evaluation are both established.
