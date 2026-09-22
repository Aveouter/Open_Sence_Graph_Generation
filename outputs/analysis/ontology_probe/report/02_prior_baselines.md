# 02 — 先验基线（B0 / B1_lookup）

对应阶段：M4（无需 GPU）
产物：`matrix/prior_cells.{json,csv}`、`matrix/eval_subsets.json`

---

## 2.1 两个基线

| 模型 | 输入 | 形式 |
|---|---|---|
| `B0` | — | 全局 `P(r)`，常数，按 dev NLL 标定温度 |
| `B1_lookup` | `(c_s, c_o)` | Dirichlet backoff：`(n(r,pair) + α·P_global(r)) / (n(pair) + α)` |

计数**只**从该 split 自己的训练子集构建；`α` 与温度在 dev 上网格搜索
（`α ∈ {0.5,1,5,20}`，`T ∈ {0.25,0.5,1,2,4}`），不在评测集上调参。
所有 split 共用 `pair_ood.train` 作为训练子集。

## 2.2 读数（macro recall）

| split | 空间 | B0 | B1_lookup | 可用类数 |
|---|---|---:|---:|---:|
| iid | vg50 | 0.0400 | 0.2253 | 25 |
| iid | L1_noise | 0.0417 | 0.2347 | 24 |
| iid | L2_entail | 0.0500 | 0.2570 | 20 |
| **pair_ood** | vg50 | 0.0370 | **0.0370** | 27 |
| **pair_ood** | L1_noise | 0.0385 | **0.0385** | 26 |
| **pair_ood** | L2_entail | 0.0435 | **0.0435** | 23 |
| pair_known | vg50 | 0.0256 | 0.1863 | 39 |
| pair_known | L1_noise | 0.0270 | 0.1963 | 37 |
| pair_known | L2_entail | 0.0323 | 0.2194 | 31 |

对应 micro accuracy 与 NLL 见 `matrix/prior_cells.csv`。

## 2.3 pair_ood 上 B1_lookup 与 B0 完全相同

vg50 上两者均为 acc 0.3374 / macro 0.0370 / NLL 2.6897，三个标签空间下逐位相同。
原因：严格 pair-OOD 下评测集的每个 pair 在训练中零支撑，`n(pair)=0`，backoff 式子
退化为 `P_global(r)`，与 B0 的定义一致。这是构造的必然结果，不是拟合失败。

同一原因使 `C_prior_conflict` 在 pair_ood 上为 0（见报告 01 第 1.5 节）。

## 2.4 池化恒等（用于 M9 的比较口径）

把 vg50 的 fine 概率按 `p_canon(C) = Σ_{r∈C} p_fine(r)` 池化后，与直接在
canonical 空间从计数重建的先验**逐位相同**（实测 6 个 split×空间组合，误差 < 1e-12）。

这是 backoff 先验的代数性质：平滑与池化都对计数线性，
`Σ_{r∈C}(n(r,pair)+α·Pg(r))/(n(pair)+α) = (n(C,pair)+α·Pg(C))/(n(pair)+α)`。
该恒等式在 `tests/analysis/test_ontology_probe_metrics.py` 中以断言固定。

推论：对先验类模型，池化不引入任何差异，`Δ_ontology` 在纯先验 cell 上恒为 0。

## 2.5 限制

- B0 的 argmax 恒为全局最高频 predicate；在 val 上该 predicate 是 `on`
  （占比 0.4458），故 B0 的 val micro accuracy 等于 0.4458，而非训练集占比 0.2849。
- macro 分母 = 评测集上支撑 ≥50 的类，随空间变化（25 / 24 / 20），
  因此上表跨空间的数字不可直接比较；跨空间比较见报告 05。
