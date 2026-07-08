# TDE Deviations

Recorded deviations:

1. Official TDE checkpoint unavailable/unverified.
   - Existing local checkpoints are not accepted as official TDE checkpoints.
   - Fallback used: explicit random-init inference/export.
   - Impact: no paper-number or checkpoint-performance claim.
2. Full official repository execution was not run in `hsg`.
   - Reason: official code targets an older Python/PyTorch/maskrcnn-benchmark
     stack, while the active objective requires `conda hsg`.
   - Impact: code-logic mapping is documented, but official runtime parity is
     not established.
3. Standard metric validation used a tiny 2-image CPU slice.
   - Impact: validates output schema and evaluator plumbing only.
4. Hidden-positive evaluation uses local JSONL predicate-recall tooling.
   - Impact: hidden-positive adaptation is local-analysis evidence, not official
     TDE paper evidence.
5. Official SGB-format VG input is incomplete.
   - Evidence: `tools/reproduction/check_tde_official_inputs.py` wrote
     `tde_official_input_check.json` with `missing_sgb_vg_inputs`.
   - Missing file:
     `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5`.
   - Impact: official SGB evaluation and official statistics/evaluator parity
     cannot be run from the current workspace.
6. Official TDE checkpoints are absent.
   - Evidence: `tde_official_input_check.json` reports no candidate checkpoint
     files under `outputs/pretrained/tde_official/predcls`, `sgcls`, or `sgdet`.
   - Impact: no checkpoint-backed TDE claim is allowed.

Related legacy notes:

- `04_checkpoint.md` records official checkpoint URLs and prior download
  failures.
- `05_mapping.md` records official-to-OpenSGG logic mapping.
- `06_deviations.md` records earlier open implementation/checkpoint deviations.
- `11_evidence_gate_audit.md` records current gate status and blockers.
