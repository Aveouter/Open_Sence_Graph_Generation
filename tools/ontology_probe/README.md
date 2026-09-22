# `tools/ontology_probe` — VG150 跨 object-pair 泛化诊断工具

**状态：诊断实验工具。** 这是一套数据集/本体测量工具，不是复现实验。
它不加载任何 SGG checkpoint，也不报告 benchmark 数字。所有产出的 artifact
都带 `status: diagnostic_experiment` 与 `not_a_reproduction: true`。

本目录只包含**工具代码**。实验产物全部写在 `outputs/analysis/ontology_probe/`
（gitignored），不属于本仓库内容。

## 它测量什么

三个量，各自独立：

- `Δ_visual` = 在 `(c_s, c_o) + geometry` 之上加视觉特征带来的变化
- `Δ_ontology` = 换用 canonical 标签空间带来的变化（在 pair-OOD 下）
- `Δ_shuffle` = 打乱视觉特征与关系的对应关系造成的性能变化

以及每 predicate 的分解与 Visual Rescue Rate（prior 判断错误时视觉纠正的比例）。

## 流水线

各阶段可单独运行：`bash run_probe_pipeline.sh s3`。`s1`–`s4` 不需要 GPU。

| 阶段 | 模块 | 做什么 |
|---|---|---|
| s1 | `ontology_stats.py` | predicate 频率、pair 计数、`H(R \| c_s, c_o)`；同时写出 5 处已有代码依赖的 `data/VisualGenome/predicate_frequencies.json` |
| s2 | `build_canonical_map.py`、`validate_canonical_map.py` | 两档 canonical 标签空间（L1 同义变体 / L2 加蕴含映射） |
| s3 | `build_splits.py`（`ood_split.py`） | 严格 pair-OOD 与 relaxed split，附带泄漏审计（`assert`，不是 warning） |
| s4 | `eval_matrix.py` | 无需 GPU 的先验半边矩阵（B0、B1_lookup） |
| s5 | `visual_sanity.py` | **门槛检查**：先用 object linear probe 等 6 项证明特征确实带信号 |
| s6 | `extract_features.py`（`feature_cache.py`） | 冻结编码器特征缓存（分片、可续跑） |
| s7 | `train_probes.py`（`probe_models.py`） | B1_add … B4 探针 |
| s8 | `shuffle_test.py`、`rescue_rate.py` | 视觉干预与 rescue rate |
| s9 | `pool_compare.py` | 池化后的 `Δ_ontology`，含 touched/untouched 分解 |
| s10 | `diagnostics_table.py` | 每 predicate 诊断表，判定规则写在 artifact 里可复核 |

支撑模块：`vg_annotations.py`（标注访问 + 索引语义校验）、`canonical_map.py`、
`splits.py`、`prior_baselines.py`、`probe_metrics.py`（指标与配对检验，纯 stdlib）、
`common.py`。

## 方法上的几个约定（先于结果确定）

这些是分析协议，写在代码里由机器执行，不是对结果的解读：

- **pair-OOD 按图像切分。** holdout 单位是**无向**类别对，`(a,b)` 与 `(b,a)` 一起
  扣掉；只扣单向会从反方向泄漏。`audit_split` 对 pair 与 image 双重不相交做
  `assert`。
- **所有 split 共用同一个训练子集。** IID 与 pair-OOD 的探针都在同一个缩减后的
  子集上训练，两者的差异因此只来自评测分布，而不是训练数据量。
- **macro 指标只在同一类集上比较。** 跨标签空间比较时，先把较细空间按
  fine→coarse 求和池化到较粗空间的类上；`vg50` 也用 `identity_map` 表示，
  不在任何地方特殊处理。
- **`B4` vs `B4_shuffled` 是参数量对照**，不是 `B4` vs `B2`。shuffle 保持结构
  与参数量完全不变，只破坏特征与关系的对应。
- **`C_prior_conflict` 无法在严格 pair-OOD 上测量。** 该评测集每个 pair 在训练中
  零支撑，先验退化为常数向量，"先验错了" 不携带信息。该子集在 IID val 与
  relaxed split 上测量。
- **报告纪律。** `on` 占训练集 28.5%，常数先验的 micro accuracy 就是 28.5%，
  因此每个报告都带 `micro_metric_warning`，`mR@1` 明确标注为 macro recall。

## 已知的方法限制

- 单 seed（0）。±0.02 macro recall 量级在 seed 方差范围内。
- 冻结 CLIP ViT-B/16，224 px，context scale 1.5。未测试其他编码器。
- `V_u` 是 subject/object 并集的裁剪图，**不含**两者之间的空间配置编码；
  部分负结果可能来自这个表示上限而非数据本身。
- pair-OOD 下有 23 个 predicate（`n_eval < 50`）被排除在每-predicate 结论之外；
  macro 指标只覆盖存活下来的类，且类集随标签空间变化。

## 实验结果

**→ [结果报告总目录](../../outputs/reports/ontology_probe/README.md)**

总目录按阶段分册，并可跳转到各份报告：

| 报告 | 阶段 |
|---|---|
| [01_dataset_statistics.md](../../outputs/reports/ontology_probe/01_dataset_statistics.md) | M1、M3 |
| [02_prior_baselines.md](../../outputs/reports/ontology_probe/02_prior_baselines.md) | M4 |
| [03_visual_probes.md](../../outputs/reports/ontology_probe/03_visual_probes.md) | M5、M6 |
| [04_interventions.md](../../outputs/reports/ontology_probe/04_interventions.md) | M7、M8 |
| [05_ontology.md](../../outputs/reports/ontology_probe/05_ontology.md) | M9、M10 |

报告只陈述测量结果、测量协议与限制，不做进一步推论。

报告是手写文档，放在 `outputs/reports/` 下并**纳入版本控制**（`.gitignore` 只
排除该目录下的 csv/json/png）。报告引用的原始产物是生成物，在
`outputs/analysis/ontology_probe/` 下（`predicate_stats/ splits/ matrix/ sanity/`
`cache/ probes/ shuffle/ rescue/ diagnostics/`），该目录被 `.gitignore` 排除，
需跑过流水线后才会出现。
