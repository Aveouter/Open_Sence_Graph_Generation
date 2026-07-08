# Evaluation Protocol: EGTR

## Official Protocol

- Dataset: Visual Genome
- Split: `test`
- Task: SGDet
- Evaluator: official EGTR `evaluate_egtr.py` / `train_egtr.evaluate_batch`
- Metrics: constrained single-predicate SGDet recall and mean recall
- Batch size: 1
- Number of object queries: 200
- Image preprocessing: `DeformableDetrFeatureExtractor`, `min_size=800`, `max_size=1333`
- Object scoring: `pred_logits.softmax(-1)[:, :num_labels]`
- Relation scoring: `pred_rel` clamped to `[0, 1]`, multiplied by `pred_connectivity`
- Single-predicate top-k: rank object-query pairs by
  `pred_rel.max(-1)[0] * outer(object_scores, object_scores)`, excluding self pairs
- Official reference: archive JSON reports `test` length `26446`

## Local Protocol

- Dataset: `data/VisualGenome`
- Split: `test`
- Task: SGDet
- Evaluator: OpenSGG `SceneGraphEvaluator` via EGTR compact SGDet adapter
- Metrics: `sgdet_R@10/20/50`, `sgdet_mR@10/20/50`
- Batch size: 1 per rank
- DDP devices: 6 visible GPUs through `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`
- Image preprocessing: local `data/dataloaders/egtr/visual_genome.py`, byte-identical to official local checkout
- Object scoring: aligned to official softmax query scoring
- Relation scoring: aligned to official `pred_rel * pred_connectivity`
- Single-predicate top-k: aligned to official query-pair ranking
- Label bridge: local evaluator uses VG IDs `1..150` and predicates `1..50`; EGTR logits remain `0..149` and `0..49`

## Alignment Verdict

aligned

## Remaining Differences

| Area | Official | Local | Impact |
|---|---|---|---|
| Evaluator implementation file | `BasicSceneGraphEvaluator` | OpenSGG `SceneGraphEvaluator` | Semantics aligned by adapter; not byte-identical |
| Label indexing | 0-indexed EGTR arrays | 1-indexed VG evaluator entries | Explicit offset bridge |
| Execution | single V100 in official JSON | 6-GPU DDP mixed A100/2080Ti | Same checkpoint/config/evaluator semantics; faster wall time |
| Persisted metrics | official JSON includes R@100/mR@100 | local config persisted through @50 | User-requested R@50/mR@50 are persisted |
