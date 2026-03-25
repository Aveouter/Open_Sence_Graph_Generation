import re
import numpy as np
import torch

from utils.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list


def metric(pred,
           true,
           metrics=('sgdet_mR@20', 'sgdet_mR@50'),
           rel_nums=None,
           entity_nums=None,
           multiple_preds=False,
           dataset='vg'):
    """
    基于官方 BasicSceneGraphEvaluator 的统一 metric 接口

    参数
    ----
    pred : dict
        模型输出，至少包含：
            sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits

    true : list[dict]
        GT，长度为 batch size，每个元素至少包含：
            boxes, labels, rel_annotations, orig_size

    metrics : tuple/list[str]
        例如：
            predcls_R@20
            predcls_mR@50
            sgcls_R@50
            sgdet_R@100

    rel_nums : int
        关系类别数（不含背景）

    entity_nums : int
        实体类别数（不含背景）
        这里主要用于一致性检查，官方 evaluator 实际不直接依赖它

    multiple_preds : bool
        传给官方 evaluator 的模式，默认 False，与官方 RelTR VG 评估一致

    dataset : str
        当前默认按 'vg' 逻辑处理。
        若以后要兼容 openimages，可再扩展。

    返回
    ----
    eval_res : dict
    eval_log : str
    """
    if rel_nums is None:
        raise ValueError('rel_nums 不能为空')
    if entity_nums is None:
        raise ValueError('entity_nums 不能为空')

    parsed_metrics = [_parse_metric_name(m) for m in metrics]
    requested_tasks = sorted(set([m['task'] for m in parsed_metrics]))

    pred = _to_cpu_detach(pred)
    true = _to_cpu_detach(true)

    if dataset.lower() != 'vg':
        raise NotImplementedError(
            f"当前这版统一 metric 先按官方 VG evaluator 重写。dataset={dataset} 暂未实现。"
        )

    # 1) 建立官方 evaluator
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=multiple_preds)

    # 2) 若请求 mR，则建立 per-class evaluator_list
    need_mr = any(m['metric'] == 'mR' for m in parsed_metrics)
    evaluator_list = None
    if need_mr:
        evaluator_list = []
        # 官方代码里 index=0 跳过，默认 0 是背景关系类
        for rel_id in range(1, rel_nums + 1):
            evaluator_list.append((rel_id, str(rel_id), BasicSceneGraphEvaluator.all_modes(multiple_preds=multiple_preds)))

    # 3) 跑 batch 评估
    _evaluate_rel_batch_vg(
        outputs=pred,
        targets=true,
        evaluator=evaluator,
        evaluator_list=evaluator_list,
        rel_nums=rel_nums,
        entity_nums=entity_nums
    )

    # 4) 汇总官方 evaluator 结果
    all_results = _collect_results_from_official_evaluator(
        evaluator=evaluator,
        evaluator_list=evaluator_list,
        requested_tasks=requested_tasks,
        rel_nums=rel_nums
    )

    # 5) 只取用户请求的 metric
    eval_res = {}
    for m in parsed_metrics:
        key = f"{m['task']}_{m['metric']}@{m['k']}"
        if key not in all_results:
            raise KeyError(f'官方 evaluator 结果中不存在指标: {key}')
        eval_res[m['raw']] = float(all_results[key])

    eval_log = ' | '.join([f'{k}: {v:.4f}' for k, v in eval_res.items()])
    return eval_res, eval_log


def _parse_metric_name(metric_name):
    """
    支持：
        predcls_R@20
        predcls_mR@50
        sgcls_R@50
        sgdet_mR@100
    """
    pattern = r'^(predcls|sgcls|sgdet)_(R|mR)@(\d+)$'
    match = re.match(pattern, metric_name)
    if match is None:
        raise ValueError(
            f"非法 metric: {metric_name}. "
            f"必须形如 predcls_mR@20 / sgcls_R@50 / sgdet_mR@100"
        )

    task, metric_type, k = match.groups()
    return {
        'raw': metric_name,
        'task': task,
        'metric': metric_type,
        'k': int(k)
    }


def _evaluate_rel_batch_vg(outputs, targets, evaluator, evaluator_list, rel_nums, entity_nums):
    """
    基本复刻官方 VG evaluate_rel_batch 逻辑，但支持统一接口。

    官方关键逻辑参考：
    - GT box / pred box 都恢复到 orig_size
    - pred_entry 使用：
        sub_boxes / sub_classes / sub_scores
        obj_boxes / obj_classes / obj_scores
        rel_scores
    - evaluator['sgdet'].evaluate_scene_graph_entry(...)
    - evaluator_list 做 mR
    """
    required_output_keys = ['sub_boxes', 'obj_boxes', 'sub_logits', 'obj_logits', 'rel_logits']
    for k in required_output_keys:
        if k not in outputs:
            raise KeyError(f'pred 缺少字段: {k}')

    for batch_idx, target in enumerate(targets):
        _check_target_fields(target)

        # GT boxes: 还原到原图尺寸
        orig_wh = torch.flip(target['orig_size'], dims=[0]).cpu()
        gt_boxes_scaled = rescale_bboxes(target['boxes'].cpu(), orig_wh).clone().numpy()

        gt_entry = {
            'gt_classes': target['labels'].cpu().clone().numpy(),
            'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
            'gt_boxes': gt_boxes_scaled
        }

        # pred boxes: 还原到原图尺寸
        sub_boxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch_idx].cpu(), orig_wh).clone().numpy()
        obj_boxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch_idx].cpu(), orig_wh).clone().numpy()

        # object scores / classes
        sub_prob = outputs['sub_logits'][batch_idx].softmax(-1)
        obj_prob = outputs['obj_logits'][batch_idx].softmax(-1)

        # 默认 object 最后一类是背景，按官方写法取 [:, :-1]
        pred_sub_scores, pred_sub_classes = torch.max(sub_prob[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(obj_prob[:, :-1], dim=1)

        # relation scores
        rel_logits = outputs['rel_logits'][batch_idx]

        # 官方 VG 代码：
        # rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)
        #
        # 这意味着 relation logits 两端有特殊类位（第0类与最后1类都不参与评估）
        # 为了兼容你当前自定义场景，做一个更稳妥的分支：
        #
        # 情况A: shape[-1] == rel_nums + 2  -> 使用 [1:-1]
        # 情况B: shape[-1] == rel_nums + 1  -> 使用 [:-1] 或 [1:] 需要看你的定义
        # 这里为了尽量贴官方 VG，优先支持 rel_nums + 2。
        #
        if rel_logits.shape[-1] == rel_nums + 2:
            rel_scores = rel_logits[:, 1:-1].softmax(-1)
        elif rel_logits.shape[-1] == rel_nums + 1:
            # 常见自定义情况：最后一类背景
            rel_scores = rel_logits[:, :-1].softmax(-1)
        elif rel_logits.shape[-1] == rel_nums:
            rel_scores = rel_logits.softmax(-1)
        else:
            raise ValueError(
                f"rel_logits.shape[-1]={rel_logits.shape[-1]} 与 rel_nums={rel_nums} 不匹配，"
                f"无法确定背景类切片方式。"
            )

        pred_entry = {
            'sub_boxes': sub_boxes_scaled,
            'sub_classes': pred_sub_classes.cpu().clone().numpy(),
            'sub_scores': pred_sub_scores.cpu().clone().numpy(),
            'obj_boxes': obj_boxes_scaled,
            'obj_classes': pred_obj_classes.cpu().clone().numpy(),
            'obj_scores': pred_obj_scores.cpu().clone().numpy(),
            'rel_scores': rel_scores.cpu().clone().numpy()
        }

        # 官方 evaluator 会同时维护多个 mode
        # 这里只评每个 mode 中实际关心的 scene graph entry
        for mode_name, mode_eval in evaluator.items():
            mode_eval.evaluate_scene_graph_entry(gt_entry, pred_entry)

        # mR: per-class evaluator
        if evaluator_list is not None:
            gt_rel = gt_entry['gt_relations']
            for pred_id, _, eval_per_rel in evaluator_list:
                gt_entry_rel = {
                    'gt_classes': gt_entry['gt_classes'].copy(),
                    'gt_boxes': gt_entry['gt_boxes'].copy(),
                    'gt_relations': gt_rel[np.in1d(gt_rel[:, -1], pred_id)]
                }
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue

                for mode_name, mode_eval in eval_per_rel.items():
                    mode_eval.evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def _collect_results_from_official_evaluator(evaluator, evaluator_list, requested_tasks, rel_nums):
    """
    从官方 evaluator 中尽量稳妥地读取:
        task_R@20 / task_R@50 / task_R@100
        task_mR@20 / task_mR@50 / task_mR@100

    不同实现版本里 BasicSceneGraphEvaluator 内部字段名可能略有差异，
    所以这里做了兼容读取。
    """
    results = {}

    # 先读取普通 R@K
    for task in requested_tasks:
        if task not in evaluator:
            continue
        mode_eval = evaluator[task]
        recall_dict = _extract_recall_dict(mode_eval)

        for k, v in recall_dict.items():
            results[f'{task}_R@{k}'] = float(v)

    # 再读取 mR@K
    if evaluator_list is not None:
        mr_dict = _extract_mean_recall_from_evaluator_list(evaluator_list, requested_tasks, rel_nums)
        results.update(mr_dict)

    return results


def _extract_recall_dict(mode_eval):
    """
    从官方 evaluator 单个 mode 中抽取 recall。
    尽量兼容不同实现版本。
    返回:
        {20: val, 50: val, 100: val, ...}
    """
    # 常见实现里可能有 result_dict
    candidate_attrs = ['result_dict', 'results', 'res']
    for attr in candidate_attrs:
        if hasattr(mode_eval, attr):
            obj = getattr(mode_eval, attr)
            recall_dict = _parse_recall_from_result_obj(obj)
            if len(recall_dict) > 0:
                return recall_dict

    # 也可能 evaluator 自己就有 recall dict
    recall_dict = _parse_recall_from_result_obj(mode_eval)
    return recall_dict


def _parse_recall_from_result_obj(obj):
    """
    兼容从 dict / 类对象里提取 R@K
    """
    recall_dict = {}

    # 情况1：dict 中直接有键名类似 'R@20' / 'recall'
    if isinstance(obj, dict):
        # 可能是 {'R@20': x, 'R@50': y}
        for k, v in obj.items():
            if isinstance(k, str):
                m = re.match(r'^R@(\d+)$', k)
                if m:
                    recall_dict[int(m.group(1))] = float(v)

        # 可能是 {'recall': {20: x, 50: y}}
        if 'recall' in obj and isinstance(obj['recall'], dict):
            for k, v in obj['recall'].items():
                try:
                    recall_dict[int(k)] = float(v)
                except Exception:
                    pass

        # 可能是 {'sgdet_recall': {20: x, 50: y}}
        for k, v in obj.items():
            if 'recall' in str(k).lower() and isinstance(v, dict):
                tmp = {}
                ok = True
                for kk, vv in v.items():
                    try:
                        tmp[int(kk)] = float(vv)
                    except Exception:
                        ok = False
                        break
                if ok and len(tmp) > 0:
                    recall_dict.update(tmp)

    # 情况2：对象有这些字段
    for attr in ['recall', 'Recall', 'sgdet_recall', 'result_dict']:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, dict):
                tmp = {}
                ok = True
                for k, v in val.items():
                    try:
                        tmp[int(k)] = float(v)
                    except Exception:
                        ok = False
                        break
                if ok and len(tmp) > 0:
                    recall_dict.update(tmp)

    return recall_dict


def _extract_mean_recall_from_evaluator_list(evaluator_list, requested_tasks, rel_nums):
    """
    参考官方 calculate_mR_from_evaluator_list(...) 的目标，
    但为了统一接口，尽量直接从 evaluator_list 中读取 per-class recall 后做平均。

    返回:
        {
            'sgdet_mR@20': ...,
            'sgdet_mR@50': ...,
            ...
        }
    """
    results = {}

    # 先尝试调用官方函数，让它内部完成统计
    # 不同版本的 calculate_mR_from_evaluator_list 可能是打印型而不是返回型，
    # 所以这里做兼容：如果没返回可用结果，再手工聚合。
    for task in requested_tasks:
        try:
            official_ret = calculate_mR_from_evaluator_list(evaluator_list, task)
            parsed = _parse_mr_return(official_ret, task)
            results.update(parsed)
        except Exception:
            pass

    # 若官方函数没返回结构化结果，则手工从 per-class evaluator 里聚合
    for task in requested_tasks:
        needed_keys_exist = any(k.startswith(f'{task}_mR@') for k in results.keys())
        if needed_keys_exist:
            continue

        # 收集每个 relation 类别的 recall@k
        per_class_recalls = {}
        for rel_id, _, eval_per_rel in evaluator_list:
            if task not in eval_per_rel:
                continue
            recall_dict = _extract_recall_dict(eval_per_rel[task])
            for k, v in recall_dict.items():
                per_class_recalls.setdefault(k, []).append(float(v))

        for k, vals in per_class_recalls.items():
            if len(vals) > 0:
                results[f'{task}_mR@{k}'] = float(np.mean(vals))
            else:
                results[f'{task}_mR@{k}'] = 0.0

    return results


def _parse_mr_return(official_ret, task):
    """
    兼容官方 calculate_mR_from_evaluator_list 的可能返回格式
    """
    out = {}
    if official_ret is None:
        return out

    if isinstance(official_ret, dict):
        # 可能直接 {'mR@20': x, 'mR@50': y}
        for k, v in official_ret.items():
            if isinstance(k, str):
                m = re.match(r'^mR@(\d+)$', k)
                if m:
                    out[f'{task}_mR@{int(m.group(1))}'] = float(v)

        # 可能 {'sgdet': {'mR@20': x, ...}}
        if task in official_ret and isinstance(official_ret[task], dict):
            for k, v in official_ret[task].items():
                m = re.match(r'^mR@(\d+)$', str(k))
                if m:
                    out[f'{task}_mR@{int(m.group(1))}'] = float(v)

    return out


def _check_target_fields(target):
    required = ['boxes', 'labels', 'rel_annotations', 'orig_size']
    for k in required:
        if k not in target:
            raise KeyError(f'target 缺少字段: {k}')


def _to_cpu_detach(x):
    if isinstance(x, dict):
        return {k: _to_cpu_detach(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_detach(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_detach(v) for v in x)
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x
