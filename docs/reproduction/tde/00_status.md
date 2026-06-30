# TDE Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#61](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/61), branch `features-codex/repro-tde` -> `feat/analysis-tools`)

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

Next action: review draft PR [#61](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/61) and merge when ready.
