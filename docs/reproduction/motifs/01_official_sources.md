# Motifs Official Sources

Primary sources:

- Paper: `Neural Motifs: Scene Graph Parsing with Global Context`, Zellers et
  al., CVPR 2018.
- Project page: https://rowanzellers.com/neuralmotifs/
- CVF paper page: https://openaccess.thecvf.com/content_cvpr_2018/html/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.html
- Common implementation lineage: Scene-Graph-Benchmark / Kaihua-style Motif
  predictor with object context LSTM, edge context LSTM, visual relation branch,
  and pair-frequency bias.

Audit interpretation:

- Motifs is a two-stage SGG relation-head baseline.
- The OpenSGG implementation targets PredCls first: GT boxes and GT object
  labels are used to predict relation predicates.
- The local checkpoint is treated as an available checkpoint fallback; full
  paper-number alignment is not claimed because only a tiny test slice was run.

Skill note:

- The `literature-query` skill instructions were read before source-audit work.
  No local paper wiki page for Motifs was available in this repository, so the
  audit uses the paper/project sources above and local code inventory.
