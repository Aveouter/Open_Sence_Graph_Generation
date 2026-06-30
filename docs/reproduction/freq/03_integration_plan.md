# FREQ Integration Plan

Plan:

1. Reuse `PairFrequencyBias` rather than duplicating a prior table.
2. Add `FREQModel` that enumerates all directed non-self object pairs and emits
   `rel_logits`, `pair_indices`, `sub_boxes`, `obj_boxes`, `obj_labels`,
   `predicate_bg_index`, and `relation_softmax_scope`.
3. Add `FREQ_Method` as a Lightning wrapper with `MotifsCriterion`, so training,
   validation, and metric paths see the same output contract as Motifs.
4. Register `freq` in `src/methods/__init__.py`.
5. Add `configs/VisualGenome/FREQ.py`.
6. Wire CLI/config resolution for `freq`.
7. Extend the relation exporter with a no-checkpoint path only for FREQ.
8. Validate with synthetic smoke, minimal CPU train smoke, standard metric slice,
   and JSONL predicate-recall slice.
