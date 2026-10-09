# 01 — SGG 配置回退值清点（只读审计）

状态：`implementation_audit`。**非复现实验**，不加载任何 checkpoint，不报告 benchmark 数字，
不改变任何运行时行为。本文件只陈述测量结果与测量协议。

审计对象：`src/models/` 与 `src/methods/` 中形如 `getattr(<holder>, "<key>", <default>)` 的
配置读取点。这类读取点的回退值**不经过任何集中声明**，是配置解码重构需要先看清的地基。

基线：提交 `34b77034ac135943cee5f25186c37a0fe327c7f1`（`main`），清点在其上、工作区干净。

---

## 1.1 规模

| 量 | 值 |
|---|---|
| 回退式读取点 | **279** |
| 涉及的配置键 | **151** |
| 回退值存在分歧的键 | **8** |
| 回退值不为字面量的读取点 | **17** |

> 先前对话中曾按约 128 处引用本项。清点结果是 279 处；128 是偏低的估数，应以本表为准。

## 1.2 测量协议（可重跑）

下列脚本即产生本文件全部数字的程序。在仓库根目录下以标准库 Python 运行即可复算：

```python
import ast, pathlib, collections
sites = collections.defaultdict(list)
for root in (pathlib.Path("src/models"), pathlib.Path("src/methods")):
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Name) and f.id == "getattr") or len(node.args) != 3:
                continue
            holder, name, default = node.args
            if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
                continue
            if isinstance(default, ast.Name) and default.id == "args":
                continue
            sites[name.value].append(
                (ast.unparse(default), isinstance(default, ast.Constant),
                 path.as_posix(), node.lineno, ast.unparse(holder))
            )
```

判定规则，逐条列出以便复查：

1. 只统计 `getattr` 恰好三参、且第二参为字符串字面量的调用；
2. 跳过回退值本身就是 `args` 的写法（那不是配置回退）；
3. 「分歧」定义为同一键在不同读取点出现**多个不同的回退值表达式**；
4. 「潜伏 / 生效」的判据：该键的每个读取方所属方法，其配置文件是否都显式设置了该键。
   任一读取方的方法**没有**任何配置设置该键，该处回退即为生效；
5. 若该键同时出现在 `utils/parser.py` 的 `default_parser()` 中，则 argparse 必然填充，
   回退不可达，单独标注。

## 1.3 分歧键全表

| 键 | 不同回退值 | 读取点数 | 在 `default_parser()` |
|---|---|---|---|
| `data_root` | `'./data'`, `None` | 4 | 是 |
| `dropout` | `0.0`, `0.1`, `0.2` | 14 | 否 |
| `freq_bias_eps` | `0.001`, `1e-12` | 12 | 否 |
| `hidden_dim` | `2048`, `256`, `512` | 16 | 否 |
| `num_queries` | `100`, `200` | 4 | 否 |
| `rel_loss_coef` | `0.8`, `15.0` | 2 | 否 |
| `use_freq_bias` | `False`, `True` | 11 | 否 |
| `visual_dim` | `2048`, `4096` | 11 | 否 |

## 1.4 分歧键的逐点位置

### `data_root`

| 回退值 | 位置 |
|---|---|
| `'./data'` | `src/models/ra_sgg.py:2075` (holder=`args`) |
| `'./data'` | `src/models/usg.py:1069` (holder=`args`) |
| `None` | `src/models/motifs.py:1493` (holder=`args`) |
| `None` | `src/models/motifs.py:1529` (holder=`args`) |

### `dropout`

| 回退值 | 位置 |
|---|---|
| `0.0` | `src/models/usg.py:1149` (holder=`args`) |
| `0.1` | `src/models/cvc.py:557` (holder=`args`) |
| `0.1` | `src/models/flowsg.py:959` (holder=`args`) |
| `0.1` | `src/models/gpsnet.py:391` (holder=`args`) |
| `0.1` | `src/models/imp.py:230` (holder=`args`) |
| `0.1` | `src/models/react_sgg.py:360` (holder=`args`) |
| `0.1` | `src/models/shagcl.py:377` (holder=`args`) |
| `0.1` | `src/models/squat.py:365` (holder=`args`) |
| `0.1` | `src/models/transformer_sgg.py:291` (holder=`args`) |
| `0.1` | `src/models/vctree.py:440` (holder=`args`) |
| `0.2` | `src/models/motifs.py:1491` (holder=`args`) |
| `0.2` | `src/models/motifs.py:1527` (holder=`args`) |
| `0.2` | `src/models/penet.py:1129` (holder=`args`) |
| `0.2` | `src/models/ra_sgg.py:2092` (holder=`args`) |

### `freq_bias_eps`

| 回退值 | 位置 |
|---|---|
| `0.001` | `src/models/motifs.py:1490` (holder=`args`) |
| `0.001` | `src/models/motifs.py:1526` (holder=`args`) |
| `1e-12` | `src/methods/egtr_method.py:144` (holder=`args`) |
| `1e-12` | `src/models/gpsnet.py:390` (holder=`args`) |
| `1e-12` | `src/models/imp.py:229` (holder=`args`) |
| `1e-12` | `src/models/penet.py:1128` (holder=`args`) |
| `1e-12` | `src/models/ra_sgg.py:2094` (holder=`args`) |
| `1e-12` | `src/models/react_sgg.py:359` (holder=`args`) |
| `1e-12` | `src/models/shagcl.py:376` (holder=`args`) |
| `1e-12` | `src/models/squat.py:364` (holder=`args`) |
| `1e-12` | `src/models/transformer_sgg.py:290` (holder=`args`) |
| `1e-12` | `src/models/vctree.py:439` (holder=`args`) |

### `hidden_dim`

| 回退值 | 位置 |
|---|---|
| `2048` | `src/models/penet.py:1120` (holder=`args`) |
| `256` | `src/models/HSTRNet.py:65` (holder=`args`) |
| `256` | `src/models/usg.py:1140` (holder=`args`) |
| `512` | `src/methods/cvc_method.py:46` (holder=`self.hparams`) |
| `512` | `src/models/cvc.py:548` (holder=`args`) |
| `512` | `src/models/flowsg.py:952` (holder=`args`) |
| `512` | `src/models/gpsnet.py:387` (holder=`args`) |
| `512` | `src/models/imp.py:226` (holder=`args`) |
| `512` | `src/models/motifs.py:1483` (holder=`args`) |
| `512` | `src/models/motifs.py:1519` (holder=`args`) |
| `512` | `src/models/ra_sgg.py:2091` (holder=`args`) |
| `512` | `src/models/react_sgg.py:354` (holder=`args`) |
| `512` | `src/models/shagcl.py:370` (holder=`args`) |
| `512` | `src/models/squat.py:359` (holder=`args`) |
| `512` | `src/models/transformer_sgg.py:285` (holder=`args`) |
| `512` | `src/models/vctree.py:437` (holder=`args`) |

### `num_queries`

| 回退值 | 位置 |
|---|---|
| `100` | `src/models/flowsg.py:957` (holder=`args`) |
| `100` | `src/models/usg.py:1141` (holder=`args`) |
| `200` | `src/methods/egtr_method.py:140` (holder=`args`) |
| `200` | `src/methods/egtr_method.py:642` (holder=`model_config`) |

### `rel_loss_coef`

| 回退值 | 位置 |
|---|---|
| `0.8` | `src/models/usg.py:1224` (holder=`args`) |
| `15.0` | `src/methods/egtr_method.py:146` (holder=`args`) |

### `use_freq_bias`

| 回退值 | 位置 |
|---|---|
| `False` | `src/models/penet.py:1127` (holder=`args`) |
| `True` | `src/methods/egtr_method.py:141` (holder=`args`) |
| `True` | `src/models/gpsnet.py:389` (holder=`args`) |
| `True` | `src/models/imp.py:228` (holder=`args`) |
| `True` | `src/models/motifs.py:1489` (holder=`args`) |
| `True` | `src/models/motifs.py:1525` (holder=`args`) |
| `True` | `src/models/react_sgg.py:358` (holder=`args`) |
| `True` | `src/models/shagcl.py:375` (holder=`args`) |
| `True` | `src/models/squat.py:363` (holder=`args`) |
| `True` | `src/models/transformer_sgg.py:289` (holder=`args`) |
| `True` | `src/models/vctree.py:438` (holder=`args`) |

### `visual_dim`

| 回退值 | 位置 |
|---|---|
| `2048` | `src/models/gpsnet.py:386` (holder=`args`) |
| `2048` | `src/models/imp.py:225` (holder=`args`) |
| `2048` | `src/models/motifs.py:1482` (holder=`args`) |
| `2048` | `src/models/motifs.py:1518` (holder=`args`) |
| `2048` | `src/models/ra_sgg.py:2088` (holder=`args`) |
| `2048` | `src/models/react_sgg.py:353` (holder=`args`) |
| `2048` | `src/models/shagcl.py:369` (holder=`args`) |
| `2048` | `src/models/squat.py:358` (holder=`args`) |
| `2048` | `src/models/transformer_sgg.py:284` (holder=`args`) |
| `2048` | `src/models/vctree.py:436` (holder=`args`) |
| `4096` | `src/models/penet.py:1119` (holder=`args`) |

## 1.5 全部键与回退值

| 键 | 回退值 | 读取点数 |
|---|---|---|
| `architecture` | `'SenseTime/deformable-detr'` | 1 |
| `aux_loss` | `True` | 1 |
| `backbone_arch` | `'resnet50'` | 1 |
| `backbone_frozen` | `True` | 3 |
| `backbone_pretrained` | `True` | 2 |
| `bbox_loss_coef` | `5.0` | 3 |
| `ce_loss_coef` | `2.0` | 1 |
| `ckpt_path` | `None` | 1 |
| `class_embed_dim` | `256` | 1 |
| `class_loss_coef` | `1.0` | 1 |
| `clip_model` | `None` | 1 |
| `clip_pretrained` | `DEFAULT_PRETRAINED` | 1 |
| `codebook_size` | `64` | 1 |
| `config` | `None` | 1 |
| `connectivity_loss_coef` | `30.0` | 1 |
| `cvc_bias_lambda` | `0.5` | 1 |
| `cvc_num_compositions` | `10068` | 1 |
| `cvc_visual_dim` | `hdim` | 1 |
| `data_root` | `'./data'`, `None` | 4 |
| `dataname` | `'VisualGenome'` | 1 |
| `dec_layers` | `6` | 1 |
| `decoder_lin` | `None` | 1 |
| `device` | `'cuda'` | 1 |
| `discrete_flow_coef` | `1.0` | 1 |
| `dropout` | `0.0`, `0.1`, `0.2` | 14 |
| `edge_lstm_layers` | `1` | 2 |
| `edge_only_prob` | `0.2` | 1 |
| `egtr_entity_nums_include_bg` | `entity_nums > 150` | 1 |
| `egtr_num_labels` | （1 种，见 1.4） | 1 |
| `egtr_num_rel_labels` | （1 种，见 1.4） | 1 |
| `egtr_pretrained_path` | `None` | 1 |
| `egtr_rel_nums_include_bg` | `rel_nums > 50` | 1 |
| `egtr_sgdet_postprocess` | `'query'` | 1 |
| `embed_dim` | `200` | 2 |
| `entity_nums` | `151` | 15 |
| `eos_coef` | `0.1` | 3 |
| `eval_mode` | `'predcls'` | 4 |
| `ffn_dim` | `2048` | 1 |
| `fg_matrix` | `None` | 1 |
| `flow_loss_coef` | `1.0` | 1 |
| `focal_alpha` | `0.25` | 1 |
| `freq_bias_eps` | `0.001`, `1e-12` | 12 |
| `giou_loss_coef` | `2.0` | 3 |
| `glove_dir` | `None` | 1 |
| `gpsnet_iterations` | `2` | 1 |
| `hidden_dim` | `2048`, `256`, `512` | 16 |
| `imp_iterations` | `2` | 1 |
| `in_channels` | `3` | 1 |
| `logit_adj_tau` | `0.3` | 1 |
| `logit_adjustment` | `False` | 1 |
| `loss` | `'hstrnet_loss'` | 1 |
| `mask_decoder_layers` | `9` | 1 |
| `max_relation_pairs` | `32` | 1 |
| `max_size` | `1333` | 1 |
| `min_size` | `800` | 1 |
| `motifs_effect_analysis` | `True` | 2 |
| `motifs_include_bg_predicate` | `rel_nums > 50` | 2 |
| `motifs_num_predicates` | `default_num_predicates` | 2 |
| `motifs_obj_feat_dim` | `getattr(args, 'pooling_dim', 4096)` | 2 |
| `motifs_obj_feat_to_edge` | `True` | 2 |
| `motifs_order` | `'leftright'` | 2 |
| `motifs_pos_batchnorm` | `True` | 2 |
| `motifs_pos_embed_dim` | `128` | 2 |
| `motifs_predicate_bg_index` | `'first'` | 2 |
| `nheads` | `8` | 2 |
| `num_blocks` | `5` | 1 |
| `num_flow_steps` | `10` | 1 |
| `num_object_classes` | `23` | 1 |
| `num_object_queries` | `16` | 1 |
| `num_predicates` | `29` | 1 |
| `num_queries` | `100`, `200` | 4 |
| `num_rel_labels` | `51` | 1 |
| `num_slots` | `4` | 1 |
| `obj_class_names` | `None` | 1 |
| `obj_ctx_trans` | `None` | 1 |
| `obj_lstm_layers` | `1` | 2 |
| `pair_loss_coef` | `1.0` | 1 |
| `penet_context_hidden_dim` | `512` | 1 |
| `penet_embed_dim` | `300` | 1 |
| `penet_nms_thresh` | `0.5` | 1 |
| `penet_pooling_dim` | `4096` | 1 |
| `penet_pos_frac` | `0.25` | 1 |
| `penet_train_pairs` | `512` | 1 |
| `pixel_ffn_dim` | `1024` | 1 |
| `pooling_dim` | `4096` | 4 |
| `pred_class_names` | `None` | 1 |
| `prototype_dim` | `self.hidden_dim` | 1 |
| `prototype_levels` | `[8, 16]` | 1 |
| `ra_sgg_embed_dim` | `300` | 1 |
| `ra_sgg_glove_dir` | `'data/glove'` | 1 |
| `ra_sgg_memory_bank_path` | `None` | 1 |
| `ra_sgg_mixup` | `True` | 1 |
| `ra_sgg_mixup_alpha` | `20.0` | 1 |
| `ra_sgg_mixup_beta` | `5.0` | 1 |
| `ra_sgg_mlp_dim` | `2048` | 1 |
| `ra_sgg_num_correct_bg` | `1` | 1 |
| `ra_sgg_num_retrievals` | `10` | 1 |
| `ra_sgg_relation_nms` | `True` | 1 |
| `ra_sgg_relation_nms_iou_threshold` | `0.6` | 1 |
| `ra_sgg_relation_nms_l21_threshold` | `0.7` | 1 |
| `ra_sgg_threshold` | `0.3` | 1 |
| `ra_sgg_use_bias` | `False` | 1 |
| `ra_sgg_use_union` | `True` | 1 |
| `react_embed_dim` | `200` | 1 |
| `react_text_only` | `False` | 1 |
| `react_use_union` | `True` | 1 |
| `rel_loss_coef` | `0.8`, `15.0` | 2 |
| `rel_loss_type` | `'ce'` | 1 |
| `rel_nums` | `51` | 16 |
| `rel_sample_negatives` | `80` | 1 |
| `rel_sample_negatives_largest` | `True` | 1 |
| `rel_sample_nonmatching` | `80` | 1 |
| `rel_sample_nonmatching_largest` | `True` | 1 |
| `relation_layers` | `6` | 1 |
| `return_relation_features` | `False` | 1 |
| `reweight_beta` | `0.99999` | 1 |
| `roi_output_size` | `7` | 3 |
| `rpc_layers` | `4` | 1 |
| `shagcl_gcl_weight` | `0.1` | 1 |
| `shagcl_nhead` | `4` | 1 |
| `shagcl_num_groups` | `3` | 2 |
| `shagcl_num_layers` | `2` | 1 |
| `smoothing` | `1e-14` | 1 |
| `squat_nhead` | `4` | 1 |
| `squat_num_layers` | `2` | 1 |
| `squat_topk_ratio` | `0.5` | 1 |
| `steps_per_epoch` | `None` | 1 |
| `tde_average_ratio` | `0.0005` | 1 |
| `tde_effect_type` | `'TDE'` | 1 |
| `tde_fusion` | `'sum'` | 1 |
| `tde_fusion_type` | `getattr(args, 'tde_fusion', 'sum')` | 1 |
| `tde_separate_spatial` | `False` | 1 |
| `tde_spatial_for_vision` | `True` | 1 |
| `temporal_dropout` | `0.1` | 1 |
| `temporal_num_heads` | `8` | 1 |
| `temporal_num_layers` | `2` | 1 |
| `top_k` | `100` | 1 |
| `transformer_dim_feedforward` | `2048` | 1 |
| `transformer_encoder_layers` | `2` | 1 |
| `transformer_nhead` | `8` | 1 |
| `use_augment` | `False` | 1 |
| `use_freq_bias` | `False`, `True` | 11 |
| `use_log_softmax` | `False` | 1 |
| `use_tanh` | `False` | 2 |
| `use_vision` | `True` | 2 |
| `usg_class_names_path` | `None` | 1 |
| `usg_num_predicates` | （1 种，见 1.4） | 1 |
| `visual_dim` | `2048`, `4096` | 11 |
| `vq_loss_coef` | `1.0` | 1 |
| `warmup_steps` | `None` | 1 |
| `weight_dict` | `{}` | 1 |

## 1.6 非字面量回退

17 处回退值不是字面量，而是**表达式**，涉及 14 个键。这一类比"同键不同字面量"更难察觉：
回退值本身依赖另一个值，于是默认行为变成两层推导。

| 键 | 回退表达式 | 位置 |
|---|---|---|
| `egtr_num_labels` | `entity_nums - 1 if getattr(args, 'egtr_entity_nums_include_bg', entity_nums > 150) else entity_nums` | `src/methods/egtr_method.py:130` |
| `egtr_entity_nums_include_bg` | `entity_nums > 150` | `src/methods/egtr_method.py:132` |
| `egtr_rel_nums_include_bg` | `rel_nums > 50` | `src/methods/egtr_method.py:137` |
| `egtr_num_rel_labels` | `rel_nums - 1 if getattr(args, 'egtr_rel_nums_include_bg', rel_nums > 50) else rel_nums` | `src/methods/egtr_method.py:135` |
| `motifs_include_bg_predicate` | `rel_nums > 50` | `src/models/motifs.py:1477`, `:1507` |
| `motifs_num_predicates` | `default_num_predicates` | `src/models/motifs.py:1481`, `:1517` |
| `motifs_obj_feat_dim` | `getattr(args, 'pooling_dim', 4096)` | `src/models/motifs.py:1485`, `:1521` |
| `tde_fusion_type` | `getattr(args, 'tde_fusion', 'sum')` | `src/models/motifs.py:1510` |
| `usg_num_predicates` | `rel_nums - 1 if rel_nums is not None and rel_nums > 50 else rel_nums` | `src/models/usg.py:1135` |
| `cvc_visual_dim` | `hdim` | `src/models/cvc.py:550` |
| `clip_pretrained` | `DEFAULT_PRETRAINED` | `src/models/usg.py:1152` |
| `prototype_dim` | `self.hidden_dim` | `src/models/HSTRNet.py:75` |
| `prototype_levels` | `[8, 16]` | `src/models/HSTRNet.py:70` |
| `weight_dict` | `{}` | `src/methods/reltr_method.py:20` |

其中三处值得单独指出，因为它们的默认值**嵌套了另一次回退**：

- `egtr_num_labels` 的默认值里含 `getattr(args, 'egtr_entity_nums_include_bg', entity_nums > 150)`
  —— 回退套回退，且内层默认由 `entity_nums` 的**大小比较**推得；
- `motifs_obj_feat_dim` 与 `tde_fusion_type` 的默认值是 `getattr(args, '<另一个键>', …)`。

这类推导在把默认值集中声明时**无法原样搬运**：它们依赖运行时才可得的量（`entity_nums`、
`rel_nums`），因此契约必须要么把这些键显式化，要么把推导本身写成具名函数并单独测试。
这是 02 号文件列为前置判断的那一项。

## 1.7 未测量项

- 各回退值**本身**哪个才是正确的：那是建模问题，需 ADR，非结构审计可答。
- `getattr` 之外的配置读取形态（如 `args.xxx` 直接属性访问、`config['k']`、
  `self.hparams` 直读）。本次清点只覆盖带回退的 `getattr` 形态。
- 运行期实际取值：未执行任何训练或评测。

