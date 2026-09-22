# 04 — 视觉干预（shuffle）与 Visual Rescue Rate

对应阶段：M7（shuffle）、M8（rescue）
产物：`shuffle/shuffle_results.json`、`shuffle/shuffle_runs.csv`、
`shuffle/shuffle_by_predicate.csv`、`rescue/rescue_results.json`、
`rescue/rescue_summary.csv`、`rescue/rescue_by_predicate.csv`

---

## 4.1 干预设计

对已训练好的 B4 固定权重，只重排视觉张量，`(c_s, c_o)`、`G`、标签、target
全部保持为同一对象（代码中以 `assert` 检查）。因此架构与参数量完全不变，
差异只能来自"哪个特征属于哪条关系"。

| 变体 | 被重排的块 |
|---|---|
| `union` | `V_u` |
| `sub_obj` | `V_s`, `V_o`（同一置换） |
| `all` | `V_s`, `V_o`, `V_u` |

| 分组 | 含义 |
|---|---|
| `global` | 在全评测集内重排，`P(V\|r)` 被破坏，视觉证据变为误导 |
| `within_predicate` | 只在同一 GT predicate 内重排，`P(V\|r)` 保持不变，只破坏实例对应 |

每个组合 5 次重复，不同 seed；报告均值与极值。
`Δacc = acc(原) − acc(重排后)`，取正表示重排造成下降。

## 4.2 读数（pair_ood）

### vg50，baseline macro = 0.1945 / acc = 0.4669

| 变体 | 分组 | Δacc（5 次均值，[min,max]） | 重排后 macro | 全部 McNemar p<0.05 |
|---|---|---:|---:|---|
| union | global | +0.0183 [+0.0171, +0.0198] | 0.1626 | 是 |
| union | within_predicate | +0.0072 [+0.0055, +0.0088] | 0.1812 | 是 |
| sub_obj | global | +0.0913 [+0.0893, +0.0935] | 0.1102 | 是 |
| sub_obj | within_predicate | −0.0046 [−0.0059, −0.0029] | 0.1865 | 否 |
| all | global | **+0.1061** [+0.1052, +0.1067] | 0.1032 | 是 |
| all | within_predicate | **−0.0064** [−0.0096, −0.0041] | **0.1947** | 否 |

### L2_entail，baseline macro = 0.2150

| 变体 | 分组 | Δacc | 重排后 macro | 全部 p<0.05 |
|---|---|---:|---:|---|
| union | global | +0.0206 | 0.1777 | 是 |
| union | within_predicate | +0.0101 | 0.2020 | 是 |
| sub_obj | global | +0.0946 | 0.1214 | 是 |
| sub_obj | within_predicate | −0.0044 | 0.2034 | 否 |
| all | global | **+0.1038** | 0.1122 | 是 |
| all | within_predicate | **−0.0079** | **0.2105** | 否 |

两个标签空间下的读数一致。

## 4.3 事实陈述

- `all__global` 的重排使 accuracy 下降约 0.104–0.106，macro recall 从
  0.1945 降到 0.1032（vg50）；即模型对"特征是否来自本关系所属图像之外的
  其他关系"是敏感的。
- `all__within_predicate` 的重排使 accuracy 变化 −0.0064（vg50）与 −0.0079
  （L2），方向为轻微上升，且 5 次重复的 McNemar 均不显著；macro recall
  0.1947 与 baseline 0.1945 在三位小数内相同。
- 该变体保留了 `P(V|r)`，只切断了"这条关系 ↔ 这幅图"的对应。

## 4.4 Visual Rescue Rate（M8）

定义：`VRR = P(ŷ_B4 = y | ŷ_B2 ≠ y)`；
`harm = P(ŷ_B4 ≠ y | ŷ_B2 = y)`；
`net = (rescue 数 − harm 数) / N`。
`τ` 子集 = `{ŷ_B2 ≠ y 且 p_B2(ŷ) > τ}`。

| split | 空间 | B2 acc | B4 acc | B2 错 | VRR | harm | net | VRR@0.5 | @0.7 | @0.9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair_ood | vg50 | 0.4697 | 0.4669 | 9,920 | 0.1549 | **0.1809** | −0.0028 | 0.0965 | 0.0471 | 0.0070 |
| pair_ood | L2_entail | 0.5032 | 0.5005 | 9,292 | 0.1591 | 0.1624 | −0.0027 | 0.1039 | 0.0485 | 0.0055 |
| iid | vg50 | 0.6855 | 0.6770 | 8,405 | 0.1171 | 0.0661 | −0.0085 | 0.0656 | 0.0147 | **0.0000** |
| iid | L2_entail | 0.7273 | 0.7143 | 7,286 | 0.0983 | 0.0548 | −0.0131 | 0.0536 | 0.0126 | 0.0030 |

`τ` 子集规模：pair_ood vg50 为 4,374 / 1,593 / 143（τ=0.5/0.7/0.9）；
iid vg50 为 5,731 / 3,187 / 201。

### 同一套机制在几何特征上的读数（对照）

用相同代码把 B2（`c_s,c_o,G`）当作"被救援方"、B1_add（`c_s,c_o`）当作"救援方"，
即测几何相对纯标签先验的 rescue：

| split | 空间 | VRR | harm | net | VRR@0.5 | @0.7 | @0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| pair_ood | vg50 | 0.1363 | 0.0945 | +0.0355 | 0.0702 | 0.0375 | 0.0360 |
| pair_ood | L2_entail | 0.1475 | 0.0883 | +0.0380 | 0.0752 | 0.0412 | 0.0343 |
| iid | vg50 | 0.0803 | 0.0231 | +0.0106 | 0.0252 | 0.0056 | **0.0000** |

## 4.5 事实陈述

- pair_ood 上 VRR 与 harm 同量级（0.155 vs 0.181），net 为负（−0.0028）。
- 随 `τ` 提高 VRR 单调下降；τ=0.9 时 pair_ood 为 0.0070，iid vg50 为 0.0000。
- 几何特征在同一套机制下也随 `τ` 下降（0.1363 → 0.0360 → 0.0000）。

## 4.6 限制

- shuffle 只在 `eval_only` 范围做（固定已训练权重），没有做"用打乱后的训练
  特征重新训练"的 `train_eval` 变体，因此读数反映的是**已训练模型对视觉证据的
  依赖度**，不是视觉信号的可学性上限。
- `within_predicate` 分组按 GT predicate，故 `on` 这类大类内部的重排样本量远大于
  尾部 predicate，均值由大类主导。
- harm 与 VRR 都是单 seed 下的配对统计；McNemar 给出的是同一 seed 内两个模型的
  配对显著性，不覆盖 seed 间的波动。
