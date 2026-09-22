# 05 — 跨标签空间比较与每 predicate 分解

对应阶段：M9（池化比较）、M10（诊断表）
产物：`matrix/pool_compare_*.{json,csv}`、`diagnostics/predicate_diagnostics_*.{csv,json,md}`

---

## 5.1 两档 canonical 标签空间

| 空间 | 类数 | 合并内容 |
|---|---:|---|
| `vg50` | 50 | 原始空间（`identity_map`，全 singleton） |
| `L1_noise` | 48 | 仅同词位变体：`lying on`+`laying on`；`wearing`+`wears` |
| `L2_entail` | 41 | L1 + 冻结的 8 条强蕴含映射：7 个 on-家族子项 → `on`，`covered in` → `in` |

L2 为 41 类而非 42：`lying on`/`laying on` 在 L1 已合并，故 on-家族在 L2 是
7 个单元并为 1，减 6；`covered in` 再减 1。

`configs/predicate_semantic_map_vg150.json` 中 28 条映射只有 8 条 strong；
`behind→near`、`wearing→has` 等 weak 映射**未**并入，因其会抹掉方向与语义信息。
排除的 pair 及理由记在 `configs/predicate_canonical_map_vg150.json` 的
`excluded_pairs` 字段。

## 5.2 比较口径

跨标签空间不能各自在自己的类集上取 macro。本报告所有跨空间数字按以下方式得到：

1. 把较细空间**按 fine→coarse 求和**池化到较粗空间的类（对 argmax 与 NLL 都精确）；
2. 两个模型在**同一个类集**上取 macro，类集为较粗空间中评测支撑 ≥50 的类；
3. 标记每条关系是否落在被合并类内：
   - **untouched**：其 vg50 predicate 在粗空间是 singleton，`canon(pred)==canon(gt)`
     与 `pred==gt` 等价，故该子集上的差值不含换标签成分；
   - **touched**：落在被合并类内，差值含重标成分。

`pool_compare` 在计算前断言池化后的 vg50 标签与 canonical 模型自身的 target
逐行相同，否则拒绝比较。

注意：对 backoff 先验类模型池化是恒等的（见报告 02 第 2.4 节），
故下表只列**学习到的**探针（B2、B4）。

## 5.3 B2（`c_s, c_o, G`）

| split | 空间 | 池化 vg50 acc | 原生 canonical acc | Δ | McNemar p | untouched Δ (n) | touched Δ (n) |
|---|---|---:|---:|---:|---:|---:|---:|
| iid | L1_noise | 0.6926 | 0.6920 | −0.0006 | 0.574 | −0.0005 (24,546) | −0.0009 (2,175) |
| iid | L2_entail | 0.7273 | 0.7257 | −0.0016 | 0.075 | −0.0023 (9,871) | −0.0012 (16,850) |
| pair_ood | L1_noise | 0.4741 | 0.4725 | −0.0016 | 0.421 | −0.0011 (17,979) | −0.0124 (726) |
| pair_ood | L2_entail | 0.5032 | 0.5000 | −0.0032 | 0.124 | +0.0021 (9,125) | −0.0081 (9,580) |
| pair_known | L1_noise | 0.6782 | 0.6786 | +0.0004 | 0.473 | +0.0008 (54,398) | −0.0019 (8,030) |
| pair_known | L2_entail | 0.7046 | 0.7027 | −0.0019 | 0.0028 | −0.0032 (31,211) | −0.0005 (31,217) |

B2 上 6 个组合的 |Δ| 均 ≤ 0.0032，其中 1 个达 p<0.01（pair_known / L2，方向为负）。

## 5.4 B4（`c_s, c_o, G, V`）

| split | 空间 | 池化 vg50 acc | 原生 canonical acc | Δ | McNemar p | untouched Δ (n) | untouched p | touched Δ (n) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid | L1_noise | 0.6770 | 0.6785 | +0.0015 | 0.282 | +0.0015 (24,546) | 0.319 | +0.0014 (2,175) |
| iid | L2_entail | 0.7143 | 0.7181 | +0.0038 | 0.0063 | **+0.0199 (9,871)** | 2.7e−12 | −0.0056 (16,850) |
| pair_ood | L1_noise | 0.4669 | 0.4699 | +0.0030 | 0.188 | +0.0042 (17,979) | 0.072 | −0.0262 (726) |
| **pair_ood** | **L2_entail** | **0.5005** | **0.4967** | **−0.0038** | **0.095** | **+0.0122 (9,125)** | **6.1e−05** | −0.0190 (9,580) |
| pair_known | L1_noise | 0.6800 | 0.6808 | +0.0008 | 0.422 | +0.0014 (54,398) | 0.202 | −0.0034 (8,030) |
| pair_known | L2_entail | 0.7087 | 0.7157 | +0.0070 | 4.2e−13 | **+0.0221 (31,211)** | 1.4e−49 | −0.0081 (31,217) |

事实陈述：

- pair_ood / L2 的整体 Δ 为 −0.0038（p=0.095），untouched 子集为 +0.0122
  （p=6.1e−05），touched 子集为 −0.0190。
- iid 与 pair_known 的 L2 上，untouched 子集 +0.0199 与 +0.0221，整体 Δ 为
  +0.0038 与 +0.0070。
- L1 上全部 6 个组合的 |Δ| ≤ 0.0030 且无一项达 p<0.05。

## 5.5 每 predicate 诊断（M10）

判定规则写在产物里可复核：`vision_generalizes` 需同时满足
`ΔV ≥ 0.05` **且** shuffle drop ≥ 0.05，且 support ≥ 50；否则归入
`prior_or_geometry_dominated` / `delta_without_correspondence` /
`degenerate_support` / `inconclusive`。

`ΔV = macro(B4) − macro(B2)`，shuffle drop 取 `all` 变体、`global` 分组的 5 次均值。

### pair_ood / vg50（阈值：support ≥ 50，ΔV ≥ 0.05，drop ≥ 0.05）

| 判定 | 个数 |
|---|---:|
| `vision_generalizes` | **1** |
| `prior_or_geometry_dominated` | 26 |
| `degenerate_support` | 23 |

满足两项阈值的 predicate：`in`（ΔV +0.0796，shuffle drop +0.1004，support 2,047）。

未达阈值但 ΔV 为正且 support 充足的：
`riding` +0.0435（drop 0.1304，support 115）、
`has` +0.0311（drop 0.2824，support 1,674）、
`near` +0.0029（drop 0.0824，support 1,706）、
`in front of` +0.0146、`standing on` +0.0154、`attached to` +0.0086。

ΔV 为负且 support 充足的（节选）：
`on` −0.0128（support 6,311）、`of` −0.0053（756）、`holding` −0.0159（757）、
`behind` −0.0573（873）、`eating` −0.1831（71）、`above` −0.0549（692）。

### pair_ood / L2_entail

| 判定 | 个数 |
|---|---:|
| `vision_generalizes` | **2** |
| `prior_or_geometry_dominated` | 21 |
| `degenerate_support` | 27 |

完整表格见 `diagnostics/predicate_diagnostics_vg50_pair_ood.md` 与
`diagnostics/predicate_diagnostics_L2_entail_pair_ood.md`。

## 5.6 尚未测量

- `Δ_ontology` 的**从头训练**含义已由本节覆盖（canonical 空间是独立训练的模型，
  不是 vg50 输出的后处理池化）。
- 未做 `B4` 在 canonical 空间下的 shuffle 干预（M9 表格中"B4 visual shuffle /
  canonical"一格为空）。
- 单 seed；上表中 p 值均为同一 seed 内两模型的配对检验，不覆盖 seed 间波动。
