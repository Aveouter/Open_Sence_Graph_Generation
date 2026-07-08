# TDE Status

Current status: `DEFERRED_NOT_REPRODUCED`

PR status: `PR_CLOSED_DEFERRED` ([#61](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/61))

Alignment audit:

- Not counted as reproduced under the stricter original-paper/original-repo standard.
- No verified official checkpoint was available.
- Random-init fallback is not reproduction.

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

Environment:

- `conda hsg`
- Test/evaluation only. No model training.

Last command:

```bash
conda run -n hsg bash -lc 'python ... metric slice ...; python tools/analysis/export_relation_predictions.py --method TDE --model TDE --task PredCLS --allow_random_init --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/tde/hidden_export_random; python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/tde/hidden_export_random/relation_predictions.jsonl --output_dir outputs/reproduction/tde/predicate_recall_random --ks 1 5 10'
```

Last result:

- No verified official/local TDE checkpoint was available.
- Random-init fallback inference was explicitly labeled with
  `--allow_random_init`.
- Synthetic forward smoke passed with loss `3.9619`.
- Standard PredCls metric slice accepted outputs; R@20/R@50/R@100 and
  mR@20/mR@50/mR@100 were all 0.0.
- Hidden/GT-aligned JSONL validation passed on 2 rows.
- Predicate recall R@1/R@5/R@10 and mR@1/mR@5/mR@10 were all 0.0.
- Official input check is blocked by missing SGB-format
  `VG-SGG-with-attri.h5` and missing official TDE checkpoints for PredCls,
  SGCls, and SGDet; see `tde_official_input_check.json`.

Next action: provide SGB-format `VG-SGG-with-attri.h5` and official TDE
checkpoints under `outputs/pretrained/tde_official/{predcls,sgcls,sgdet}/`,
then inspect causal keys and run official SGB evaluation before attempting
OpenSGG checkpoint mapping.
