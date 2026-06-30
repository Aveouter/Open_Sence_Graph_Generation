# FREQ Official Sources

Primary sources:

- Neural Motifs paper: `Scene Graph Parsing with Global Context`, Zellers et al.,
  CVPR 2018. The paper reports simple frequency baselines such as FREQ and
  FREQ+OVERLAP alongside Neural Motifs.
- Neural Motifs project: https://rowanzellers.com/neuralmotifs/
- CVF paper page: https://openaccess.thecvf.com/content_cvpr_2018/html/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.html
- Scene-Graph-Benchmark style implementations commonly expose this prior as
  `FrequencyBias` / `freq_bias`, used by Motifs and TDE-style relation heads.

Interpretation used for this reproduction:

- FREQ is treated as the non-visual pair prior
  `P(predicate | subject class, object class)`.
- It is evaluated in PredCls mode because it requires ground-truth object labels
  and boxes to form subject-object pairs.
- No paper-number alignment is claimed from the tiny local smoke/metric slice.
