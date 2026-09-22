# 03 — 视觉特征门槛检查与 B3 / B4 探针

对应阶段：M5（门槛）、M6（探针）
产物：`sanity/visual_sanity.{json,csv}`、`probes/<cell>/`

---

## 3.1 特征缓存

| 项 | 值 |
|---|---|
| 编码器 | `openai/clip-vit-base-patch16`，冻结，`pooler_output`（768 维） |
| 裁剪方式 | crop-and-encode，直接 resize 到 224（不 ROI-pool 14×14 patch grid） |
| context scale | 1.5（按框中心放大后 clamp 到图像边界） |
| 存储 | fp16 分片 h5，1,000 图/片 |
| train 缓存 | 58 片，670,591 物体 / 315,642 关系 / 1.53 GB |
| val 缓存 | 5 片，62,754 物体 / 26,721 关系 / 0.14 GB |

物体特征按 `(image_id, obj_idx)` **每物体只抽一次**；union 特征按关系抽。
抽取吞吐 132 crops/s（6 线程解码），全量 train+val 用时约 1 小时。

## 3.2 M5 门槛检查（train，前 20,000 张图 / 109,517 条关系）

| 检查 | 结果 |
|---|---|
| `variance` | 通过 — 0 个死维度，mean variance 0.613 |
| `determinism` | 通过 — 同一 crop 重编码两次逐位相同（max abs diff = 0.0） |
| `alignment` | 通过 — 缓存特征与现场重编码的余弦 ≥ 0.9999 |
| `object_probe` | 通过 — **0.4445** vs majority 0.1565（150 类，80/20 划分） |
| `pair_discriminability` | **未通过** — 见下 |
| `pair_predicate_probe` | 通过 — 0.3382 vs majority 0.2819，**+0.0562** |

`pair_discriminability`：同一 pair 下相同 predicate 的两条关系，平均余弦 0.4218；
不同 predicate 的两条，平均余弦 0.4227；差值 **−0.00087**（比较数 7,756,526 与
12,355,108）。该检查比较的是原始余弦。

`pair_predicate_probe`：限定在 695 个多-predicate pair 上（train 65,707 行 /
test 22,440 行，49 个 predicate，按 pair 而非按行划分），对 union 特征 `V_u`
做线性探针预测 predicate，test 准确率 0.3382 对 majority 0.2819。

两项检查测的是不同东西：前者是原始余弦相似度在固定 pair 下的差异，
后者是在固定 pair 下 predicate 是否线性可分。

门槛判定：`object_probe` 通过（阈值 0.20），流程继续。

## 3.3 探针定义

| 模型 | 输入 | pair_ood macro（vg50） |
|---|---|---:|
| B0 | — | 0.0370 |
| B1_lookup | `(c_s, c_o)` 查表 | 0.0370 |
| B1_add | `E_s + E_o`（可组合因式先验） | 0.1876 |
| B2 | `c_s, c_o, G` | 0.2147 |
| B3 | `V_s, V_o, V_u, G`（无 object label） | 0.1563 |
| B4 | `c_s, c_o, G, V_s, V_o, V_u` | 0.1945 |

`G` 为 8 维几何特征，复用 `src/models/motifs.py::_pair_geometry` 的定义
（归一化 cxcywh 上的 dx, dy, log wh 比, 面积比, 中心距, IoU, union 面积）。

## 3.4 三个 split × 三个标签空间的 macro recall

| 空间 | split | B2 | B3 | B4 | **B4 − B2** | 可用类数 |
|---|---|---:|---:|---:|---:|---:|
| vg50 | iid | 0.3172 | 0.2121 | 0.3212 | +0.0040 | 25 |
| vg50 | pair_ood | 0.2147 | 0.1563 | 0.1945 | **−0.0201** | 27 |
| vg50 | pair_known | 0.2656 | 0.1831 | 0.2797 | +0.0141 | 39 |
| L1_noise | iid | 0.3212 | 0.2239 | 0.3328 | +0.0116 | 24 |
| L1_noise | pair_ood | 0.2130 | 0.1626 | 0.2050 | **−0.0080** | 26 |
| L1_noise | pair_known | 0.2722 | 0.1951 | 0.2997 | +0.0275 | 37 |
| L2_entail | iid | 0.3591 | 0.2291 | 0.3378 | −0.0213 | 20 |
| L2_entail | pair_ood | 0.2456 | 0.1656 | 0.2150 | **−0.0306** | 23 |
| L2_entail | pair_known | 0.3094 | 0.2001 | 0.2863 | −0.0231 | 31 |

micro accuracy 对照（同 9 个 cell）：

| 空间 | split | B2 | B3 | B4 | B4 − B2 |
|---|---|---:|---:|---:|---:|
| vg50 | iid | 0.6855 | 0.5962 | 0.6770 | −0.0085 |
| vg50 | pair_ood | 0.4697 | 0.4188 | 0.4669 | −0.0028 |
| vg50 | pair_known | 0.6652 | 0.5912 | 0.6800 | +0.0148 |
| L1_noise | iid | 0.6926 | 0.6037 | 0.6822 | −0.0105 |
| L1_noise | pair_ood | 0.4741 | 0.4223 | 0.4664 | −0.0077 |
| L1_noise | pair_known | 0.6782 | 0.6010 | 0.6927 | +0.0146 |
| L2_entail | iid | 0.7273 | 0.6370 | 0.7143 | −0.0131 |
| L2_entail | pair_ood | 0.5032 | 0.4476 | 0.5005 | −0.0027 |
| L2_entail | pair_known | 0.7046 | 0.6234 | 0.7087 | +0.0041 |

**pair_ood 三行中，B4 的 macro recall 与 micro accuracy 均低于 B2。**
B3 在全部 9 个 cell 上均低于 B2。

## 3.5 参数量对照

B2 与 B4 的隐层完全相同（`(512, 256)`），差异只在输入维度：
B2 = 264（128+128+8），B4 = 2568（264 + 3×768）。
参数量 B2 = 320,066，B4 = 1,509,458（首层差值为主）。

由于 B4 参数量更大，`B4 − B2` 不是参数量对齐的对照；
参数量对齐的对照是 `B4 − B4_shuffled`（见报告 04），因为 shuffle 保持结构与
参数量完全不变。

## 3.6 训练过程记录

每个 cell 记录 `selected_epoch`、`n_epochs_run`、`hit_epoch_cap`、逐 epoch
dev NLL。本次全部 18 个视觉 cell 的 `selected_epoch` 落在 6–7，
`hit_epoch_cap=false`（上限 100），即由 early stopping 终止而非被上限截断。

## 3.7 限制

- 单 seed（0）。未做多 seed 复现；±0.02 macro recall 量级与 seed 方差的相对大小
  未测量。
- `V_u` 是 subject/object 并集框的裁剪图，**不含**两者的空间配置编码。
  报告 3.2 中 `pair_discriminability` 与 `pair_predicate_probe` 两项读数不同，
  说明该表示在原始余弦上区分不出 predicate，但线性探针可读出一定信息；
  两者都不包含位置关系。
- 训练子集固定为 `pair_ood.train`，因此 iid / pair_known 的数字不是在全量
  train 上训练的。
