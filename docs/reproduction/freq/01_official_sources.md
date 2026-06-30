# FREQ Official Sources

Primary sources:

- Neural Motifs paper: `Scene Graph Parsing with Global Context`, Zellers et al.,
  CVPR 2018. The paper reports simple frequency baselines such as FREQ and
  FREQ+OVERLAP alongside Neural Motifs.
- Neural Motifs project: https://rowanzellers.com/neuralmotifs/
- Neural Motifs official code: https://github.com/rowanz/neural-motifs
- CVF paper page: https://openaccess.thecvf.com/content_cvpr_2018/html/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.html
- Scene-Graph-Benchmark reference used for local source audit:
  https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
  - Local path:
    `/workspace/external/tde_official/Scene-Graph-Benchmark.pytorch`
  - Local commit: `ceb71fa88461c2a97a6258a80f47669d89207296`
  - Official-style prior implementation:
    `maskrcnn_benchmark/modeling/roi_heads/relation_head/model_motifs.py:FrequencyBias`
  - Official-style statistics construction:
    `maskrcnn_benchmark/data/datasets/visual_genome.py:get_statistics`

Interpretation used for this reproduction:

- FREQ is treated as the non-visual pair prior
  `P(predicate | subject class, object class)`.
- It is evaluated in PredCls mode because it requires ground-truth object labels
  and boxes to form subject-object pairs.
- Current OpenSGG evidence is only structurally aligned with the SGB
  `FrequencyBias` API. Numerical parity of the prior table is not established.
- No paper-number alignment is claimed from the tiny local smoke/metric slice.

See `11_evidence_gate_audit.md` for the current gate-by-gate status.
