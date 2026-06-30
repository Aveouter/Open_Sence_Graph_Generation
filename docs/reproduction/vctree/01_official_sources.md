# VCTree Official Sources

Primary source:

- `Learning to Compose Dynamic Tree Structures for Visual Contexts`, Tang et
  al., CVPR 2019.
- Original paper code: https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation
- SGB reference implementation:
  https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
  - Local path:
    `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
  - Local commit: `ceb71fa88461c2a97a6258a80f47669d89207296`
  - SGB predictor:
    `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py:VCTreePredictor`
  - SGB context:
    `maskrcnn_benchmark/modeling/roi_heads/relation_head/model_vctree.py:VCTreeLSTMContext`
  - SGB tree utilities:
    `maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_vctree.py`

Implementation interpretation:

- VCTree replaces sequential Motifs context with dynamic tree construction and
  TreeLSTM-style context propagation.
- OpenSGG validates PredCls first using GT boxes and labels.

Checkpoint status:

- No local VCTree checkpoint was found in `outputs/pretrained`.
- SGB README says most SGG model checkpoints are not uploaded; VCTree normally
  requires training unless a trusted checkpoint is supplied.
- No paper-number alignment is claimed.

See `11_evidence_gate_audit.md` for the gate status and current blockers.
