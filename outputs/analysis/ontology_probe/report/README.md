# VG150 跨 object-pair 泛化诊断 —— 结果报告总目录

**状态：diagnostic experiment（诊断实验），非复现实验。** 不加载任何 SGG
checkpoint，不报告 benchmark 数字。

全部读数均由 `tools/ontology_probe/` 下的工具产生，报告只陈述测量结果与
测量协议，不做进一步推论。

## 报告清单

| 报告 | 对应阶段 | 内容 |
|---|---|---|
| [01_dataset_statistics.md](01_dataset_statistics.md) | M1、M3 | VG150 语料统计、pair / 熵、split 构造与泄漏审计 |
| [02_prior_baselines.md](02_prior_baselines.md) | M4 | B0 / B1_lookup 先验基线，pair-OOD 与 IID 上的读数 |
| [03_visual_probes.md](03_visual_probes.md) | M5、M6 | 视觉特征门槛检查；B3 / B4 探针矩阵与 `Δ_visual` |
| [04_interventions.md](04_interventions.md) | M7、M8 | 视觉干预（shuffle）；Visual Rescue Rate |
| [05_ontology.md](05_ontology.md) | M9、M10 | 跨标签空间池化比较；每 predicate 分解 |

## 运行配置（全部读数共用）

| 项 | 值 |
|---|---|
| 数据 | `data/VisualGenome/`（train 57,723 图 / 315,642 关系；val 5,000 / 26,721；test 26,446 / 152,226） |
| 编码器 | `openai/clip-vit-base-patch16`，冻结，crop-and-encode，224 px，context scale 1.5 |
| 探针 | 2 层 MLP `(512, 256)`，LayerNorm 按 block 归一化；各 cell 结构与超参完全一致 |
| 优化 | AdamW，lr 3e-4，wd 1e-4，batch 8192，max 100 epoch，按 dev NLL early stop（patience 5） |
| 训练子集 | 所有 split 共用 `pair_ood.train`（211,219 条关系），因此 split 间差异只来自评测分布 |
| seed | 0（单 seed；多 seed 复现未做） |

## 产物位置

各阶段原始产物在 `outputs/analysis/ontology_probe/` 下：
`predicate_stats/ splits/ matrix/ sanity/ cache/ probes/ shuffle/ rescue/ diagnostics/`。
`outputs/` 被 `.gitignore` 排除，不属于仓库提交内容；报告链接为本地路径。
