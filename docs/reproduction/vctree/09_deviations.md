# VCTree Deviations

Recorded deviations:

1. No VCTree checkpoint available locally.
   - Fallback: explicit random-init inference/export.
   - Impact: no performance or paper-number claim.
2. Tiny CPU slice only.
   - Impact: validates plumbing, not benchmark accuracy.
3. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: local-analysis evidence only.
4. Official SGB-format VG input is incomplete.
   - Evidence: `tools/reproduction/check_vctree_official_inputs.py` wrote
     `vctree_official_input_check.json` with `missing_sgb_vg_inputs`.
   - Missing file:
     `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5`.
   - Impact: official SGB VCTree evaluation/evaluator parity cannot be run.
5. Trusted VCTree checkpoint is absent.
   - Evidence: `vctree_official_input_check.json` reports zero checkpoint
     candidates under `outputs/pretrained/vctree_official`.
   - Impact: no checkpoint-backed VCTree claim is allowed.
6. Local OpenSGG VCTree is not SGB `VCTreePredictor` parity.
   - Evidence: SGB uses `VCTreeLSTMContext`, `vctree_score_net`,
     `generate_forest`, `arbForest_to_biForest`, and SGB frequency statistics.
     Local OpenSGG uses a simplified `TreeConstructor` and local TreeLSTM.
   - Impact: local smoke evidence remains implementation audit only.
