# EGTR Hidden-Positive Evaluation

Command:

```bash
conda run -n hsg bash -lc 'python tools/analysis/export_egtr_predictions.py --ckpt_path outputs/pretrained/egtr/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/epoch=03-validation_loss=1.71.ckpt --device cpu --test_dataset_size 2 --val_batch_size 1 --num_workers 0 --max_batches 1 --top_k 10 --output_dir outputs/reproduction/egtr/hidden_export_ckpt && python tools/analysis/compute_predicate_recall_from_jsonl.py --predictions outputs/reproduction/egtr/hidden_export_ckpt/relation_predictions.jsonl --output_dir outputs/reproduction/egtr/predicate_recall_ckpt --ks 1 5 10'
```

Export summary:

- Mode: `checkpoint`.
- Export method: `egtr_gt_aligned_compact`.
- Total GT relations: 2.
- Exported rows: 2.
- Matched rows: 2.
- Unmatched rows: 0.
- Score dimension: 50.
- Errors: 0.
- Warnings: 0.

Predicate recall:

- R@1: 0.0
- R@5: 0.5
- R@10: 1.0
- mR@1: 0.0
- mR@5: 0.5
- mR@10: 1.0

Scope:

- Two-image CPU slice.
- GT-aligned hidden-positive pipeline validation only.
