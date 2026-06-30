# Motifs Status

Current status: `HIDDEN_EVAL_ADAPTED`

PR status: `PR_SUBMITTED_DRAFT` ([#60](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/60), branch `features-codex/repro-motifs` -> `feat/analysis-tools`)

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
- Test/evaluation only. No model training is part of the accepted Motifs evidence.

Last command:

```bash
conda run -n hsg bash -lc 'python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/motifs/hidden_export_ckpt/relation_predictions.jsonl --output_dir outputs/reproduction/motifs/predicate_recall_ckpt --ks 1 5 10 && python - <<PY ... PY'
```

Last result:

- Local checkpoint `outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth` loaded through OpenSGG checkpoint adaptation.
- External checkpoint key remap: 627 input keys -> 48 loaded model keys.
- Missing keys: `input_visual_proj.weight`, `input_visual_proj.bias`.
- Relation JSONL validation passed on 2 GT relations.
- GT-aligned predicate recall: R@1 0.0, R@5 1.0, R@10 1.0; mR@1 0.0, mR@5 1.0, mR@10 1.0.
- Standard PredCls metric slice accepted outputs; R@20/R@50/R@100 and mR@20/mR@50/mR@100 were all 0.0 on the tiny slice.

Next action: review draft PR [#60](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/60) and merge when ready.
