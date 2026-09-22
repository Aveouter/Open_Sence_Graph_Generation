# 2025–2026 Visual Genome and Open Images literature audit

Checked on **2026-09-22**. Status: `implementation_audit` / literature survey.
The [README result tables](../README.md#visualgenome-paper-results) contain
transcribed paper values. This document records the selection and its limits.
No model was trained or evaluated for this survey; no checkpoint or
OpenSGG-to-author-code equivalence was established.

## Selection rule

Include a method in the new README catalogue when a primary source identifies
a 2025 or 2026 publication, the work concerns static-image SGG on VG or Open
Images, and an author GitHub repository contains method source code. A released
component can qualify as code, but incomplete integration must be disclosed.
A README, paper PDF, figures, upstream benchmark link, third-party
implementation, or an OpenSGG adapter alone is insufficient.

The conference scope is CVPR, ICCV, ECCV, ICLR, ICML, NeurIPS, AAAI, IJCAI and
ACM MM main tracks. The core journal scope is TPAMI, IJCV, TIP and TMM; JAS is
recorded as an additional related journal. WACV/BMVC and other journals are
not silently treated as this core scope. This explicit venue list makes the
otherwise subjective phrase “top conferences/journals” reviewable.

Conference year takes precedence over arXiv upload or a later proceedings
deposit. Journal online and issue years are recorded separately where they
differ. The 2026 coverage ends at the audit date; later publications and code
releases cannot be covered by this snapshot.

## Search coverage

Search terms were `scene graph generation`, `visual relationship detection`,
`open vocabulary`, `Visual Genome`, `Open Images`, method names and exact paper
titles. Community lists and general search were discovery aids; acceptance,
numbers and code decisions use publishers, proceedings, author papers and
author repositories. Negative search results mean **not verified in this
audit**, not proof that no repository exists anywhere.

| Search surface | Primary entry points | Outcome / limitation |
|---|---|---|
| CVPR 2025/2026, ICCV 2025 | [CVPR 2025](https://openaccess.thecvf.com/CVPR2025?day=all), [CVPR 2026](https://openaccess.thecvf.com/CVPR2026?day=all), [ICCV 2025](https://openaccess.thecvf.com/ICCV2025?day=all) | Conference indexes screened; paper PDFs checked for static-image VG/OI evaluation and code links. HQSG and VL-IRM qualify; 2026 candidates have code gaps below. |
| ICLR 2025/2026 | [2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025), [2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026) | Hydra-SGG repository is a placeholder; APT has prompt-component source. |
| ICML 2025/2026 | [PMLR 267](https://proceedings.mlr.press/v267/), [ICML 2026 downloads](https://icml.cc/Downloads/2026) | NoDIS qualifies. HSGG has code, but its OpenReview PDF was inaccessible. RPC has no verified author code link. Scene-graph retrieval is outside scope. |
| AAAI 2025/2026 | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/issue/archive) and exact-title searches | RA-SGG/RAHP verified from publisher PDFs; SSC-SGG lacks verified author code. Targeted 2026 searches did not establish another eligible VG/OI method. |
| NeurIPS 2025/2026 | [NeurIPS proceedings](https://proceedings.neurips.cc/) and exact-title searches | ACC verified in the 2025 main proceedings. No unverified 2026 acceptance is assumed. Video/embodied-only results are excluded. |
| ECCV 2026 | [ECVA proceedings](https://www.ecva.net/papers.php), [Dual-SGG publisher record](https://doi.org/10.1007/978-3-032-37098-3_23) | Dual-SGG paper found; its embedded author GitHub URL returned 404. |
| IJCAI / ACM MM | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/), publisher DOI discovery and title/code searches | No additional method met all three gates in this search. This is not a claim that every 2026 paper has been indexed or checked. UniQ is MM 2024, not 2025. |
| Journals | IEEE Xplore, Springer, JAS publisher pages and DOI metadata discovery | MCL/GCM code verified; journal numerical tables remain unavailable. HMSC2 has a named-predictor mismatch. Other candidates and exclusions are listed below. |

## Included methods and source evidence

GitHub links refer to the default branches inspected on the audit date; source
availability can change. This is a file-level screening, not a full scientific
implementation audit. A linked training script is not evidence that a run has
been executed successfully.

| Method and full paper title | Primary paper | Source evidence inspected | Result extraction |
|---|---|---|---|
| **HQSG** — Hybrid Reciprocal Transformer with Triplet Feature Alignment for Scene Graph Generation | [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Hybrid_Reciprocal_Transformer_with_Triplet_Feature_Alignment_for_Scene_Graph_CVPR_2025_paper.pdf) | [Author repository](https://github.com/fujiawei0724/HQSG): `modeling/`, `engine/`, `evaluation/`, `configs/`, `train.py` | Table 1, PDF p. 6 / printed p. 8958; HQSG ResNet-101 row. The older project-page code-release notice does not override the now-populated repository. |
| **RA-SGG** — Retrieval-Augmented Scene Graph Generation Framework via Multi-Prototype Learning | [AAAI 2025 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33036/35191) | [Author repository](https://github.com/KanghoonYoon/torch-rasgg): `maskrcnn_benchmark/`, `scripts/`, `Model_Zoo.md` | Table 1, PDF p. 6 / printed p. 9567, RA-SGG row. Author README documents relation NMS in PredCls and SGCls. |
| **RAHP** — Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation | [AAAI 2025 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32594/34749) | [Author repository](https://github.com/Leon022/RAHP): `maskrcnn_benchmark/modeling/roi_heads/relation_head/ov_classifier.py`, `tools/generate_relation_aware_embedding.py` | VG Tables 1/3, OI Table 2; PDF pp. 6–7. Paper variants retain their own base architectures; release of this repository does not verify every adapted variant. |
| **NoDIS** — Noise-Guided Predicate Representation Extraction and Diffusion-Enhanced Discretization for Scene Graph Generation | [ICML 2025 publisher page](https://proceedings.mlr.press/v267/zhang25ak.html), [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhang25ak/zhang25ak.pdf) | [Author repository](https://github.com/gavin-gqzhang/NoDIS): relation predictors, `diffusion_utils.py`, `model_utils.py`, configurations and train/test scripts | Table 1, PDF p. 7, four NoDIS base-model variants; Table 2, PDF p. 9, Transformer-NoDIS, separate OI V4/V6 rows. |
| **VL-IRM** — Vision-Language Interactive Relation Mining for Open-Vocabulary Scene Graph Generation | [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Min_Vision-Language_Interactive_Relation_Mining_for_Open-Vocabulary_Scene_Graph_Generation_ICCV_2025_paper.pdf) | [Author repository](https://github.com/myukzzz/VL-IRM): `maskrcnn_benchmark/`, `configs/`, `tools/`, `VG150_split_GLIPunseen.py` | Tables 1–4, PDF p. 6 / printed p. 16760. Preserve the paper's `Ours`, `Ours*`, and `OvSGTR+Ours*` distinction. |
| **ACC** — Interaction-Centric Knowledge Infusion and Transfer for Open Vocabulary Scene Graph Generation | [NeurIPS 2025 accepted paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/f7b118ed1bfd2a9f366d55021a8bc1e0-Paper-Conference.pdf) | [Author repository](https://github.com/HKUST-LongGroup/ACC): `models/`, `datasets/`, `config/`, `engine.py`, `main.py` | Tables 1/2, PDF p. 8, both Swin-T and Swin-B. The README badge's older INOVA preprint is not used as the numerical source. |
| **MCL** — Multi-Concept Learning for Scene Graph Generation | [TIP 2025, DOI 10.1109/TIP.2025.3540296](https://doi.org/10.1109/TIP.2025.3540296) | [Author repository](https://github.com/XinyuLyu/G-USGG): multi-prototype/memory relation-predictor code and Motif/PE-Net training instructions | **Pending final-paper access.** Do not substitute the withdrawn 2023 preprint linked by the repository; see below. |
| **GCM** — Human-Inspired Scene Understanding: A Grounded Cognition Method for Unbiased Scene Graph Generation | [TPAMI, DOI 10.1109/TPAMI.2025.3635152](https://doi.org/10.1109/TPAMI.2025.3635152) | [Author repository](https://github.com/Nora-Zhang98/GCM): `VTransEPredictorGCM` and other GCM predictors; VG/GQA/OIV6 instructions and `tools/cal_mean_std.py` | **Pending final-paper access.** The repository's dataset instructions and detector links are not numerical paper results. Online 2025, issue 2026. |
| **APT** — APT: Towards Universal Scene Graph Generation via Plug-in Adaptive Prompt Tuning | [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/23bce04d18001e4b4eee05aba89706ee-Paper-Conference.pdf) | [Prompt-component source](https://github.com/CGCL-codes/APT/blob/HEAD/maskrcnn_benchmark/modeling/adpative_modeling.py) contains adaptive prompts and `CompositionalGeneralizationPrompter`; the filename is spelled `adpative_modeling.py` upstream | Representative variants from Table 2, PDF p. 7, and Table 8, p. 15. Integration and protocol caveats below. |
| **HSGG** — HSGG: Training-Free Hierarchical Scene Graph Generation with Geometry-Guided Relation Reasoning | [ICML 2026 official presentation](https://icml.cc/virtual/2026/poster/65975), [OpenReview paper](https://openreview.net/forum?id=8Hmanb9Pum) | [Author-designated implementation](https://github.com/lllyz789/HSGG): `scene_graph_generator.py`, `modules/`, `GICD/`, VG150/PSG dataset instructions | **Pending primary PDF access.** OpenReview returned a browser-verification page. No values are imported from third-party summaries or reproduction logbooks. |
| **BiRef** — Bimodal Predicate Refinement With Decoupled Entity-Predicate Representations for Scene Graph Generation | [JAS 2026, vol. 13(6), pp. 1470–1491](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2026.125786) | [Author repository linked by the journal](https://github.com/gavin-gqzhang/BiRef): relation predictor registered as `BiRef`, modeling code and scripts | **Pending numerical full text.** Publisher abstract names VG/GQA/Open Images; the accessible page did not expose its tables. Added as a related journal, separately from the core venue screen. |

## Protocol and source decisions

### Visual Genome

VG150 ordinarily refers to 150 object and 50 predicate categories, but the name
alone does not determine preprocessing, split membership, graph constraints,
detector, or postprocessing. PredCls supplies both boxes and object labels;
SGCls supplies boxes; SGDet supplies neither. Their numbers belong in separate
columns or tables.

The README preserves reported values without recomputing or filling in missing
metrics. Closed-vocabulary, open-relation (OvR), and open-object-plus-relation
(OvD+R) experiments are separate. An unseen-triplet zero-shot metric is also
not the same as holding out predicate classes.

For RAHP, PredCls uses the EPIC relation split while its OvSGTR SGDet variant
uses the OvSGTR split. VL-IRM explicitly presents both 50% novel PGSG and 30%
novel OvSGTR settings. Its GLIP/Flan-T5 model and GroundingDINO/OvSGTR
augmentation have different pretraining. ACC §4.1 further removes test images
overlapping GroundingDINO pretraining and retains **14,700** images. These are
material comparison boundaries, not cosmetic method labels.

### Open Images

The user's “Image” is interpreted as the repository's **Open Images V6**.
OI V4 is also recorded when the same eligible paper reports it, but its scores
are not pooled with V6.

The NoDIS table reports `wmAP_rel`, `wmAP_phr`, and a weighted score, with
`score = 0.2 R@50 + 0.4 wmAP_rel + 0.4 wmAP_phr`. Its V6 row yields
`0.2 × 74.11 + 0.4 × 38.87 + 0.4 × 38.95 = 45.95` after rounding.
RAHP and VL-IRM instead report total/base/novel recall under their PGSG-derived
open-vocabulary protocol. A recall value must not be presented as weighted mAP.

### APT

APT is included because its repository contains actual adaptive-prompt and
compositional-generalization components, beyond its two-line README. However,
the inspected relation predictors are benchmark predictors; a repository-wide
source search found prompt definitions in `adpative_modeling.py`, without
establishing integrations for all the architectures in the paper. The source
release is not evidence that each tabulated variant is runnable.

Two independent paper issues also need clarification:

- §4.1 says OI V6 has **288 entity / 30 relation classes and 5,322 test images**;
  Appendix B says **301 object / 31 predicate classes and 6,322 test images**.
  The README therefore isolates Table 8 with a `protocol_mismatch` warning.
- Some reported F scores do not equal the harmonic mean of the accompanying
  R/mR cells, despite that metric definition. For example, VG Table 2's
  HQSG+APT SGDet R@50 = 36.5 and mR@50 = 18.2 imply about 24.29, whereas
  the paper prints F@50 = 22.6. The README transcribes R/mR only and neither
  silently repairs nor endorses the F column.

APT's Tables 3 and 9 also report open-vocabulary base/novel results. Their
method integrations and task alignment need clarification before adding them
to the mixed-method OvR tables. The README's APT rows are explicitly
representative paper variants, not an exhaustive transcription of every table.

### MCL source identity

The journal article is TIP 2025, DOI `10.1109/TIP.2025.3540296`. Its author
repository is named G-USGG and links [arXiv:2308.04802](https://arxiv.org/abs/2308.04802),
whose title is *Generalized Unbiased Scene Graph Generation*. That arXiv record
is marked withdrawn. The withdrawn version must not supply numbers attributed
to the 2025 journal article. MCL's concept-level `mCR@K` must also remain
distinct from predicate-level `mR@K` when final tables become available.

## Excluded or unresolved candidates

These rows explain screening decisions; they are **not** included methods or
baseline results. “No verified code” describes this search's evidence limit.

| Candidate | Primary evidence | Decision at cutoff |
|---|---|---|
| Hydra-SGG, ICLR 2025 | [Accepted paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/10831838f1cdd8fcba90188e88e7a121-Paper-Conference.pdf), [author GitHub](https://github.com/Dreamer312/Hydra-SGG) | Repository contains only `README.md` and `LICENSE`; paper/README numbers do not satisfy the actual-code filter. |
| Robo-SGG, CVPR 2026 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Lv_Robo-SGG_Exploiting_Layout-Oriented_Normalization_and_Restitution_Can_Improve_Robust_Scene_CVPR_2026_paper.pdf), [author GitHub](https://github.com/MICLAB-BUPT/Robo-SGG) | Inspected repository contains `README.md`, `Fig1.png`, `Fig2.png`; no implementation. |
| FlowSG, CVPR 2026 | [Author paper, arXiv:2604.18623](https://arxiv.org/abs/2604.18623), [CVF record](https://openaccess.thecvf.com/CVPR2026?day=all) | No verified author implementation. The local adapter is insufficient. Corrected the README's unrelated arXiv link; no paper-results row added. |
| MoE feature decoupling, CVPR 2026 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Mixture-of-Experts_based_Feature_Decoupling_for_Open_Vocabulary_Scene_Graph_Generation_CVPR_2026_paper.pdf) | No verified author GitHub implementation in paper/title searches. |
| Dual-SGG, ECCV 2026 | [Publisher record](https://doi.org/10.1007/978-3-032-37098-3_23), [author preprint](https://arxiv.org/abs/2607.06176) | Embedded [author code URL](https://github.com/runfeng-q/Revisiting-Scene-Graph-Generation-from-the-Perspective-of-Detector-Conditioned-Reachability) returned 404. Do not substitute the author's Salience-SGG repository. |
| RPC, ICML 2026 | [Official presentation](https://icml.cc/virtual/2026/poster/65758) | *Beyond Unidirectional Bias: Reciprocal Perspective Calibration in Scene Graph Generation* found; no verified author implementation; primary PDF access also blocked. |
| CAPSGG / Navigating the Unseen, CVPR 2025 | [CVF proceedings](https://openaccess.thecvf.com/CVPR2025?day=all) | Paper checked; no verified author GitHub implementation. Baseline code references do not qualify. |
| Conformal Prediction and MLLM aided Uncertainty Quantification in SGG, CVPR 2025 | [CVF proceedings](https://openaccess.thecvf.com/CVPR2025?day=all) | No verified author GitHub implementation. Conformal-set metrics would additionally require a separate protocol. |
| SSC-SGG, AAAI 2025 | [Publisher PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32998/35153) | No author code URL established in the paper or exact-name repository search. |
| USG-Par, CVPR 2025 | [Original paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Universal_Scene_Graph_Generation_CVPR_2025_paper.pdf), [author GitHub](https://github.com/ChocoWu/USG) | Image results in Table 2 are **PSG**, not VG or OI. Existing OpenSGG support does not change the paper's benchmark. |
| GSGG, IJCV 2025 | [Publisher](https://doi.org/10.1007/s11263-025-02499-z), [author GitHub](https://github.com/ZHUXUHAN/GSGG) | Author repository inspected as README-only; no implementation verified. |
| DB-SGG, IJCV 2025 | [Informative Scene Graph Generation via Debiasing](https://doi.org/10.1007/s11263-025-02365-y) | No verified method-specific author code; upstream captioning/benchmark links in the article do not qualify. |
| MCFA, IJCV 2026 | [Publisher full text](https://doi.org/10.1007/s11263-026-02855-7) | No verified MCFA author implementation. The older CFA/ICCV 2023 repository must not be relabeled as this method. |
| HMSC2, TIP 2026 | [Publisher](https://doi.org/10.1109/TIP.2025.3650668), [author source](https://github.com/Nora-Zhang98/HMSC2/blob/HEAD/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py) | Source files exist, but README requests `HMSC2Predictor` and the inspected registry file does not define/register that predictor. `implementation_audit`; method-code correspondence unresolved, so no numerical row. |
| CAModule, TPAMI 2025 | [A Causal Adjustment Module for Debiasing SGG](https://doi.org/10.1109/TPAMI.2025.3537283) | Publication found; no verified author implementation. |
| RcSGG, TPAMI 2025 | [A Reverse Causal Framework to Mitigate Spurious Correlations for Debiasing SGG](https://doi.org/10.1109/TPAMI.2025.3568644) | Publication found; no verified author implementation. A same-name third-party repository is insufficient. |
| CFEN, TPAMI 2025 | [Exploring the Essence of Relationships for SGG via Causal Features Enhancement Network](https://doi.org/10.1109/TPAMI.2025.3559995) | Only an unofficial simplified implementation found; excluded by author-code criterion. |
| FMSGG, TPAMI 2026 | [Foundation Models based Scene Graph Generation](https://doi.org/10.1109/TPAMI.2026.3712094) | Publication found; no verified author implementation. |
| RelationLMM, TPAMI 2025 | [Large Multimodal Model as Open and Versatile Visual Relationship Generalist](https://doi.org/10.1109/TPAMI.2025.3531452) | Publication found; no verified author implementation. |
| Hierarchical SGG With Coarse-to-Fine Reasoning, TMM 2026 | [Publisher](https://doi.org/10.1109/TMM.2026.3664986) | Publication found; no verified author implementation. |
| Relationship-Incremental SGG, TIP issue 2025 | [Publisher, DOI with 2024 online year](https://doi.org/10.1109/TIP.2024.3384096) | Incremental protocol; no verified author implementation in this audit. |
| UniQ | [Publisher record](https://doi.org/10.1145/3664647.3681542) | The paper is ACM MM **2024** despite a 2025 arXiv upload; outside the publication window. |
| SDSGG | [Author preprint](https://arxiv.org/abs/2410.15364) | NeurIPS **2024**, outside the publication window. |
| Diff-VRD | [Author preprint](https://arxiv.org/abs/2504.12100) | Inspected version says under review at TCSVT; do not treat that as acceptance. |
| SPOT | [ICLR 2026 submission](https://openreview.net/pdf?id=JeqXsxUTkn) | Accessible source is an anonymous under-review version; main-track acceptance and eligible author code were not established. |

Video/Action Genome, 3D point-cloud scene graphs, PSG-only segmentation graphs,
scene-graph-conditioned image generation and graph retrieval are excluded from
these VG/OI results. For example, [ESCA](https://proceedings.nips.cc/paper_files/paper/2025/hash/0418973e545b932939302cb605d06f43-Abstract-Conference.html)
and [SCENIR](https://github.com/nickhaidos/scenir-icml2025) address other evaluation
settings. Workshop acceptance is not main-track acceptance.

## Related venues outside the core selection

These relevant public-code works are recorded for follow-up, without silently
expanding the README's top-venue result tables or importing their baselines.

| Work | Venue / primary source | Author code |
|---|---|---|
| Effective SGG by Statistical Relation Distillation (SRD) | [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Nguyen_Effective_Scene_Graph_Generation_by_Statistical_Relation_Distillation_WACV_2025_paper.pdf) | [LUNAProject22/SRD](https://github.com/LUNAProject22/SRD) |
| Enhancing SGG with Hierarchical Relationships and Commonsense Knowledge (HIERCOM) | [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Jiang_Enhancing_Scene_Graph_Generation_with_Hierarchical_Relationships_and_Commonsense_Knowledge_WACV_2025_paper.pdf) | [bowen-upenn/scene_graph_commonsense](https://github.com/bowen-upenn/scene_graph_commonsense) |
| REACT | [BMVC 2025 proceedings](https://bmvc2025.bmva.org/proceedings/239/) | [Maelic/SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark); current REACT++ results must not be attributed to the 2025 paper |
| Salience-SGG | [WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Qu_Salience-SGG_Enhancing_Unbiased_Scene_Graph_Generation_with_Iterative_Salience_Estimation_WACV_2026_paper.pdf) | [runfeng-q/Salience-SGG](https://github.com/runfeng-q/Salience-SGG) |

## Remaining evidence gaps and update procedure

The catalogue has eleven methods with source code, of which seven have
transcribed numerical results. Final-paper numerical evidence remains pending
for **MCL, GCM, HSGG and BiRef**. This limitation is visible in the README;
their missing values are not reconstructed from another paper's baseline row,
an abstract's relative gain, a withdrawn preprint, or a third-party run.

When updating this survey:

1. Verify the venue and final version from a primary publication record.
2. Inspect the author repository's actual method source; record changed code
   availability and implementation gaps.
3. Read the original table and its experimental setup, including supplements.
   Record task, category split, test set, backbone, supervision and metric.
4. Keep variants and incompatible protocols separate. Verify numeric column
   order against the PDF; several source tables extract poorly as plain text.
5. Use the repository's reproduction workflow for any later executable baseline
   work. This literature record alone grants no reproduction status.

The survey does not rule out undiscovered papers, repositories or inaccessible
publisher tables. It records what was verified and what still needs evidence.
