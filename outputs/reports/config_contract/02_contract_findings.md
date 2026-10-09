# 02 — 配置契约的判读与下一轮切法

状态：`implementation_audit`。**非复现实验**，不加载任何 checkpoint，不报告 benchmark 数字。
本文件不改变任何运行时行为，只对 01 号文件的测量作判读，并给出下一轮 TDD 的切分。

依据：`01_fallback_inventory.md`（同目录）。基线提交
`34b77034ac135943cee5f25186c37a0fe327c7f1`。

---

## 2.1 核心结论

**八个分歧键当前全部处于潜伏状态：没有任何一个会在现有配置下触发。**

判据不是"配置文件恰好设了它所以没事"，而是逐键核对了**读取方所属方法**的配置是否设置该键。
核对方式与逐键结果记录在 2.2，可依 01 号文件的协议复算。

## 2.2 逐键判定

| 键 | 读取方所属方法 | 未设置该键的配置 | 判定 |
|---|---|---|---|
| `data_root` | motifs, ra_sgg, usg | 全部 19 个（该键只由 parser 提供） | **不可达**：在 `default_parser()` 中，argparse 必然填充 |
| `hidden_dim` | 16 处 | 无（19/19 均设置） | 潜伏 |
| `dropout` | usg, cvc, flowsg, gpsnet, imp, react_sgg, shagcl, squat, transformer_sgg, vctree, motifs, penet, ra_sgg | `HSTRNet`（但 HSTRNet 不是 `dropout` 的读取方） | 潜伏 |
| `freq_bias_eps` | motifs×2, gpsnet, imp, penet, ra_sgg, react_sgg, shagcl, squat, transformer_sgg, vctree, egtr_method | `FlowSG`、`HSTRNet`、`RelTR`、`USG`（四者均不是读取方） | 潜伏 |
| `num_queries` | flowsg, usg, egtr_method×2 | 16 个配置未设置，但 **`FlowSG` 与 `USG` 在设置方之列** | 潜伏 |
| `rel_loss_coef` | usg, egtr_method | 16 个配置未设置，但读取方 `USG`、`EGTR` 均设置 | 潜伏 |
| `use_freq_bias` | penet, gpsnet, imp, motifs×2, react_sgg, shagcl, squat, transformer_sgg, vctree, egtr_method | `FlowSG`、`HSTRNet`、`RA_SGG`×3、`RelTR`、`USG`（均不是读取方） | 潜伏 |
| `visual_dim` | gpsnet, imp, motifs×2, ra_sgg, react_sgg, shagcl, squat, transformer_sgg, vctree, penet | `EGTR`、`FlowSG`、`HSTRNet`、`RelTR`、`USG`（均不是读取方） | 潜伏 |

「读取方」= 在 01 号文件 1.4 节中出现的模型或方法文件。「未设置该键的配置」= 该配置文件中
没有对应该键的顶层赋值。一个键只有在其**读取方**的配置集合中存在遗漏时才会生效。

## 2.3 为什么仍然值得处理：遮蔽是偶然的，不是结构性的

这一点是本次审计的主要产出，也是它值得成为架构证据而非一次性检查的原因。

那七个键**都不在 `utils/parser.py` 的 `default_parser()` 中**。也就是说，`argparse` 不会提供它们，
`getattr` 的回退值是代码侧唯一的兜底。当前之所以没有分歧生效，**唯一的原因是每个读取方的方法
碰巧都有一份设置了该键的配置**。这不是由任何结构保证的：

- 新增一份省略该键的配置，就会静默把一个按模型而异的内置值带进运行；
- 该内置值还会随**读取方是哪个模型**而变——同一份配置在 A 模型下拿到 512，在 B 模型下拿到 2048。

按后果排序，激活后影响最大的是：

| 键 | 分歧 | 为什么是这一档 |
|---|---|---|
| `rel_loss_coef` | 0.8 / 15.0 | 损失权重相差约 19 倍，直接改变训练目标 |
| `hidden_dim` | 256 / 512 / 2048 | 决定模型容量，且与参数计数、checkpoint 形状耦合 |
| `dropout` | 0.0 / 0.1 / 0.2 | 三档，影响正则强度 |
| `use_freq_bias` | False / True | 开关量，改变频率先验是否参与 |

## 2.4 对先前说法的更正

先前对话中曾多次把 `freq_bias_eps` 描述为一处**生效中**的分歧（"Motifs 回退到 `1e-3`，
另外九个回退到 `1e-12`"）。按 2.2 的核对，该说法有两处不准确：

1. 读取点是 **12 处**，不是 10 处（其中 `src/methods/egtr_method.py:144` 位于方法层，
   此前未被计入）；
2. `0.001` 与 `1e-12` 的两个分支**都不会生效**：读取该键的每一个方法，其配置都显式设置了它。

分歧在源码中真实存在，但它当前不可达。记录此更正是因为 02 号文件将作为下一轮的输入，
一个被高估为"生效中"的缺陷会误导优先级。

## 2.5 契约应取的形状

由 01、02 的测量直接推得三条约束：

1. **要声明的是那七个非 parser 键。** `data_root` 已由 `default_parser()` 覆盖，纳入契约只会
   制造第二个事实源。
2. **必须是"按模型声明"，不能是标量。** `hidden_dim`（512/2048）、`dropout`（0.1/0.2）、
   `visual_dim`（2048/4096）、`use_freq_bias`（True/False）、`rel_loss_coef`（0.8/15.0）、
   `num_queries`（100/200）、`freq_bias_eps`（1e-3/1e-12）之中，至少一部分看起来是**有意为之**
   （例如 penet 的 `visual_dim=4096` 与其骨干一致）。强行统一会真的改动数字。
   因此形状是 `key -> {模型: 值}`，无分歧的键退化为一元。
3. **17 处非字面量回退无法原样搬运。** 它们依赖运行时量（`entity_nums`、`rel_nums`）或其他键，
   例如 `egtr_num_labels` 的默认值里又嵌了一次 `getattr`。契约要么把这些键显式化，
   要么把推导写成具名函数并单独测试。**这是下一轮开始前需要先定的判断**，
   因为它决定契约是纯数据还是要带一个函数层。

## 2.6 下一轮 TDD 的切法

按垂直切片推进，每片一个测试一个实现，并为每道判据配 must-fire 与 must-pass 控制
（ADR 0007 对"能失败的闸门"的要求）。

**切片 A —— 契约数据本身。**
- 行为：七个键各有声明的按模型回退值，且与源码现值逐一相符。
- must-fire：改动任一源码回退值 → 对不上，报错。
- must-pass：当前工作区全绿。
- 控制手段：与 01 号文件 1.4 节的清单比对；该清单即本切片的期望值来源。

**切片 B —— 防回潮护栏。**
- 行为：源码中出现的回退值与契约不符即为 finding。
- must-fire：临时把某处 `1e-12` 改成 `1e-9` → finding。
- must-pass：现有 279 处全部相符 → 干净。
- 注：该检查应落在能在无依赖 CI job 中运行的位置，参考候选 1 的
  `src/method_registry.py` 先例（纯数据、标准库、可 AST 或直接导入读取）。

**切片 C —— 非字面量回退。** 先按 2.5 第 3 条作出判断（显式化 vs 具名函数），再切。
在判断作出之前不应开始编码，否则会先锁死一个未定的接口。

顺序上 A → B 是连的，C 独立。**本文件不主张** A、B、C 中任何一项已经完成。

## 2.7 未主张

- 未主张任何回退值是**错误的**。哪个值才是模型的正确默认，是建模问题，需要 ADR；
  本审计只判定"是否可达"，不判定"是否正确"。
- 未主张分歧是缺陷。按模型而异可以是刻意的设计，本文件只要求它被**声明**而非内嵌。
- 未主张复现结论。未运行任何训练或评测，未产生任何 benchmark 数字，
  未改动任何基线或证据文件。
- 未主张清点完整覆盖配置读取**全部形态**。只统计了带回退值的 `getattr`
  （见 01 号文件 1.7）。
