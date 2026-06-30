# Motifs Hidden-Positive Evaluation

Export command:

```bash
conda run -n hsg python tools/analysis/export_relation_predictions.py --method Motifs --model Motifs --task PredCLS --ckpt_path outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --output_dir outputs/reproduction/motifs/hidden_export_ckpt
```

Export result:

- JSONL validation: PASS.
- Total GT relation rows: 2.
- Matched rows: 2.
- Unmatched rows: 0.
- Score dimension: 50.
- Predicate checksum prefix: `8dc025ea105b`.

Predicate recall command:

```bash
conda run -n hsg python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/motifs/hidden_export_ckpt/relation_predictions.jsonl --output_dir outputs/reproduction/motifs/predicate_recall_ckpt --ks 1 5 10
```

Predicate recall result:

- R@1: 0.0
- R@5: 1.0
- R@10: 1.0
- mR@1: 0.0
- mR@5: 1.0
- mR@10: 1.0

Interpretation:

- Hidden/GT-aligned evaluation is adapted at the schema and execution level for
  checkpoint-backed Motifs outputs.
- The numbers are tiny-slice sanity results only.
