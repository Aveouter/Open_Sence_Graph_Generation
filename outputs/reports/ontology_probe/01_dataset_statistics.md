# 01 — VG150 语料统计与 split 构造

对应阶段：M1（predicate 统计）、M3（pair-OOD split）
产物：`predicate_stats/`、`splits/`

---

## 1.1 语料规模

数据源 `data/VisualGenome/`（COCO 格式 `train/val/test.json` + `rel.json`）。
`rel.json` 的三元组为 `[sub_idx, obj_idx, pred_id]`，前两者是**该图像标注列表的
0-based 位置下标**，不是全局 annotation id；`rel_categories[0]` 为
`__background__`，真实 predicate 为 1..50。

| 量 | train | val | test |
|---|---|---|---|
| 图像 | 57,723 | 5,000 | 26,446 |
| 物体 | 670,591 | 62,754 | 325,570 |
| 关系三元组 | 315,642 | 26,721 | 152,226 |

索引语义由 `vg_annotations.validate_index_semantics` 抽样断言（越界即失败），
train 上通过、0 处问题。

## 1.2 熵与 pair 统计（train）

| 量 | 值 |
|---|---|
| `H(R)` | 3.590 bits |
| `H(R \| c_s, c_o)` | 1.392 bits（Miller–Madow 校正 1.436） |
| `I(R ; c_s, c_o)` | **2.198 bits** |
| `H(R \| c_s)` | 2.583 bits |
| `H(R \| c_o)` | 2.246 bits |
| `I(R ; c_s)` | 1.007 bits |
| `I(R ; c_o)` | 1.344 bits |
| distinct 有向 pair | 9,783 |
| distinct 无向 pair | 6,324 |
| 其中携带 >1 predicate 的无向 pair | 4,552 |

`H(R|c_s,c_o) / H(R)` = 0.388，即 `I(R;pair) / H(R)` = 0.612。

## 1.3 predicate 分布

| predicate | train | val | test |
|---|---:|---:|---:|
| `on` | 0.2849 | 0.4458 | 0.3347 |
| `has` | 0.1640 | 0.0952 | 0.1340 |
| `of` | 0.0850 | 0.0463 | 0.1122 |

三个 split 图像互不重叠（`train∩val`、`train∩test`、`val∩test` 均为 0），
但 predicate 占比不同。常数 `on` 预测器在 val 上的 micro accuracy 即 0.4458。

训练集上频率尾部：`flying in` 5 条、`says` 35 条、`made of` 93 条；
50 个 predicate 中 49 个拥有 ≥20 个不同训练 pair。

## 1.4 pair-OOD split 构造

- **按图像切分**：被扣掉的 pair 所在图像整图移出训练集。
- **holdout 单位是无向 pair**：扣 `(a,b)` 同时扣 `(b,a)`，保证两个方向都未见过。
- 贪心选择：得分 = 该 pair 在 study predicate 下的关系数 / 新触及图像数；
  受 `image_cap_fraction` 预算约束；平局用 `sha1(seed|c_s|c_o)`（不用 `hash()`，
  因 `PYTHONHASHSEED` 逐进程随机）。
- 参数：`max_pair_count=20`，`min_pair_count=2`，`image_cap_fraction=0.20`，
  `dev_fraction=0.10`，`seed=42`。

| 量 | pair_ood | pair_known |
|---|---:|---:|
| train 关系 | 211,219 | 227,869 |
| dev 关系 | 23,384 | 24,785 |
| eval 关系 | 18,705 | 62,428 |
| 扣掉的 pair 数 | 2,676 | 0（保留 pair，只扣实例） |
| 触及图像 | 11,544 | 11,545 |

study predicate 的 eval 支撑（pair_ood）：
`on` 6,311 / `in` 2,047 / `has` 1,674 / `holding` 757 / `of` 756 /
`wearing` 537 / `riding` 115。

低于阈值（`n_train<20` 或 `n_eval<50`）被排除在每-predicate 结论外的
predicate 共 23 个，含 `flying in`（`n_train=4`）。

`split_audit.json` 中对以下各项做 `assert`（不是 warning）：relation 不交、
**image 不交**、`pairs(train) ∩ pairs(eval) = ∅` 在所有方向与所有 predicate 下成立、
train/dev 图像不交、两侧非空。

## 1.5 先验冲突子集的支撑

`C_prior_conflict` = 该 pair 在训练中**见过**、但标定后先验 argmax ≠ GT。

| split | 空间 | n_eval | n_conflict | 占比 |
|---|---|---:|---:|---:|
| iid | vg50 | 26,721 | 10,378 | 0.388 |
| iid | L2_entail | 26,721 | 9,283 | 0.347 |
| **pair_ood** | vg50 | 18,705 | **0** | **0.000** |
| **pair_ood** | L2_entail | 18,705 | **0** | **0.000** |
| pair_known | vg50 | 62,428 | 27,368 | 0.438 |
| pair_known | L2_entail | 62,428 | 25,210 | 0.404 |

pair_ood 上该子集为空：严格 pair-OOD 下每个评测 pair 训练支撑为 0，先验退化为
全局常数向量，其 argmax 与 pair 无关。因此该子集改在 iid 与 pair_known 上测量。

## 1.6 限制

- 阈值 `n_eval<50` 使 pair_ood 上 23 个 predicate（约一半）不进入每-predicate 结论；
  macro 指标只覆盖存活类，且类集随标签空间变化。
- split 构造用的 study predicate 集合为
  `on, has, of, in, holding, wearing, riding`，贪心是按这个集合的增益排序的；
  换集合会得到不同的 holdout。
