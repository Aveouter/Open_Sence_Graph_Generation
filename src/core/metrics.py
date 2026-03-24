import re
import numpy as np
import torch


def metric(pred, true, metrics=('sgdet_mR@20', 'sgdet_mR@50'), rel_nums=None, entity_nums=None,
           iou_thresh=0.5, multiple_preds=False):
    """
    统一 Scene Graph / RelTR 评测函数

    参数
    ----
    pred : dict
        预测输出，至少包含：
            sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits

    true : list[dict]
        GT，长度为 batch size，每个元素至少包含：
            boxes, labels, rel_annotations

    metrics : list[str]
        形如：
            predcls_R@20
            predcls_mR@50
            sgcls_R@100
            sgdet_mR@20

    rel_nums : int
        关系类别数（不含背景）

    entity_nums : int
        实体类别数（不含背景）

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

    tasks = sorted(set([m['task'] for m in parsed_metrics]))
    ks = sorted(set([m['k'] for m in parsed_metrics]))

    pred = _to_cpu_detach(pred)
    true = _to_cpu_detach(true)

    all_results = {}
    for task in tasks:
        task_res = _eval_scene_graph_task(
            pred=pred,
            true=true,
            task=task,
            rel_nums=rel_nums,
            entity_nums=entity_nums,
            ks=ks,
            iou_thresh=iou_thresh,
            multiple_preds=multiple_preds
        )
        all_results.update(task_res)

    eval_res = {}
    for m in parsed_metrics:
        name = m['raw']
        key = f"{m['task']}_{m['metric']}@{m['k']}"
        eval_res[name] = all_results[key]

    eval_log = ' | '.join([f'{k}: {v:.4f}' for k, v in eval_res.items()])
    return eval_res, eval_log


def _parse_metric_name(metric_name):
    """
    解析:
        predcls_mR@20
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


def _eval_scene_graph_task(pred, true, task, rel_nums, entity_nums, ks, iou_thresh=0.5, multiple_preds=False):
    total_gt = 0
    matched_global = {k: 0 for k in ks}

    per_rel_gt = np.zeros(rel_nums, dtype=np.int64)
    per_rel_matched = {k: np.zeros(rel_nums, dtype=np.int64) for k in ks}

    B = len(true)
    for b in range(B):
        gt_triplets = _build_gt_triplets(true[b], rel_nums)

        if task == 'predcls':
            pred_triplets = _build_pred_triplets_predcls(
                pred=pred, target=true[b], batch_idx=b,
                rel_nums=rel_nums, entity_nums=entity_nums,
                multiple_preds=multiple_preds
            )
        elif task == 'sgcls':
            pred_triplets = _build_pred_triplets_sgcls(
                pred=pred, target=true[b], batch_idx=b,
                rel_nums=rel_nums, entity_nums=entity_nums,
                multiple_preds=multiple_preds
            )
        elif task == 'sgdet':
            pred_triplets = _build_pred_triplets_sgdet(
                pred=pred, target=true[b], batch_idx=b,
                rel_nums=rel_nums, entity_nums=entity_nums,
                multiple_preds=multiple_preds
            )
        else:
            raise ValueError(f'Unsupported task: {task}')

        total_gt += len(gt_triplets)

        for gt in gt_triplets:
            rel_label = gt['rel_label']
            if 0 <= rel_label < rel_nums:
                per_rel_gt[rel_label] += 1

        for k in ks:
            matched_idx = _match_triplets(pred_triplets[:k], gt_triplets, iou_thresh)
            matched_global[k] += len(matched_idx)

            for idx in matched_idx:
                rel_label = gt_triplets[idx]['rel_label']
                if 0 <= rel_label < rel_nums:
                    per_rel_matched[k][rel_label] += 1

    res = {}
    valid_rel = per_rel_gt > 0
    for k in ks:
        res[f'{task}_R@{k}'] = float(matched_global[k] / max(total_gt, 1))

        class_recall = np.zeros(rel_nums, dtype=np.float64)
        class_recall[valid_rel] = (
            per_rel_matched[k][valid_rel] / np.maximum(per_rel_gt[valid_rel], 1)
        )
        res[f'{task}_mR@{k}'] = float(class_recall[valid_rel].mean()) if valid_rel.any() else 0.0

    return res


def _build_gt_triplets(target, rel_nums):
    boxes = _as_numpy(target['boxes']).astype(np.float32)
    labels = _as_numpy(target['labels']).astype(np.int64)
    rels = _as_numpy(target['rel_annotations']).astype(np.int64)

    triplets = []
    for rel in rels:
        s_idx, o_idx, r_label = rel.tolist()

        if not (0 <= r_label < rel_nums):
            continue
        if not (0 <= s_idx < len(labels) and 0 <= o_idx < len(labels)):
            continue

        triplets.append({
            'sub_label': int(labels[s_idx]),
            'obj_label': int(labels[o_idx]),
            'rel_label': int(r_label),
            'sub_box': boxes[s_idx],
            'obj_box': boxes[o_idx],
        })
    return triplets


def _build_pred_triplets_predcls(pred, target, batch_idx, rel_nums, entity_nums, multiple_preds=False):
    """
    predcls:
    - object label 和 object box 使用 GT
    - relation 使用预测
    注意：这里假设 query 和 GT 对齐。若你的实现不是这样，需要再改。
    """
    gt_boxes = _as_numpy(target['boxes']).astype(np.float32)
    gt_labels = _as_numpy(target['labels']).astype(np.int64)
    rel_prob = torch.softmax(pred['rel_logits'][batch_idx], dim=-1).cpu().numpy()

    # 默认第0类为背景关系
    if rel_prob.shape[1] == rel_nums + 1:
        rel_prob = rel_prob[:, 1:]

    nq = min(len(gt_boxes), rel_prob.shape[0])

    triplets = []
    for i in range(nq):
        if multiple_preds:
            rel_order = np.argsort(-rel_prob[i])
            for r in rel_order:
                score = float(rel_prob[i, r])
                triplets.append({
                    'sub_label': int(gt_labels[i]),
                    'obj_label': int(gt_labels[i]),
                    'rel_label': int(r),
                    'sub_box': gt_boxes[i],
                    'obj_box': gt_boxes[i],
                    'score': score,
                })
        else:
            r = int(np.argmax(rel_prob[i]))
            score = float(np.max(rel_prob[i]))
            triplets.append({
                'sub_label': int(gt_labels[i]),
                'obj_label': int(gt_labels[i]),
                'rel_label': int(r),
                'sub_box': gt_boxes[i],
                'obj_box': gt_boxes[i],
                'score': score,
            })

    triplets.sort(key=lambda x: x['score'], reverse=True)
    return triplets


def _build_pred_triplets_sgcls(pred, target, batch_idx, rel_nums, entity_nums, multiple_preds=False):
    """
    sgcls:
    - object box 用 GT
    - object label / relation 用预测
    注意：这里同样假设 query 和 GT 对齐。
    """
    gt_boxes = _as_numpy(target['boxes']).astype(np.float32)

    sub_prob = torch.softmax(pred['sub_logits'][batch_idx], dim=-1).cpu().numpy()
    obj_prob = torch.softmax(pred['obj_logits'][batch_idx], dim=-1).cpu().numpy()
    rel_prob = torch.softmax(pred['rel_logits'][batch_idx], dim=-1).cpu().numpy()

    # 默认 object 最后一类为背景
    if sub_prob.shape[1] == entity_nums + 1:
        sub_prob = sub_prob[:, :-1]
    if obj_prob.shape[1] == entity_nums + 1:
        obj_prob = obj_prob[:, :-1]
    # 默认 relation 第0类为背景
    if rel_prob.shape[1] == rel_nums + 1:
        rel_prob = rel_prob[:, 1:]

    pred_sub_scores = sub_prob.max(axis=1)
    pred_sub_labels = sub_prob.argmax(axis=1)
    pred_obj_scores = obj_prob.max(axis=1)
    pred_obj_labels = obj_prob.argmax(axis=1)

    nq = min(len(gt_boxes), len(pred_sub_labels), len(pred_obj_labels), len(rel_prob))

    triplets = []
    for i in range(nq):
        if multiple_preds:
            rel_order = np.argsort(-rel_prob[i])
            for r in rel_order:
                score = float(pred_sub_scores[i] * pred_obj_scores[i] * rel_prob[i, r])
                triplets.append({
                    'sub_label': int(pred_sub_labels[i]),
                    'obj_label': int(pred_obj_labels[i]),
                    'rel_label': int(r),
                    'sub_box': gt_boxes[i],
                    'obj_box': gt_boxes[i],
                    'score': score,
                })
        else:
            r = int(np.argmax(rel_prob[i]))
            r_score = float(np.max(rel_prob[i]))
            score = float(pred_sub_scores[i] * pred_obj_scores[i] * r_score)
            triplets.append({
                'sub_label': int(pred_sub_labels[i]),
                'obj_label': int(pred_obj_labels[i]),
                'rel_label': int(r),
                'sub_box': gt_boxes[i],
                'obj_box': gt_boxes[i],
                'score': score,
            })

    triplets.sort(key=lambda x: x['score'], reverse=True)
    return triplets


def _build_pred_triplets_sgdet(pred, target, batch_idx, rel_nums, entity_nums, multiple_preds=False):
    """
    sgdet:
    - object box / object label / relation 全预测
    """
    sub_boxes = _as_numpy(pred['sub_boxes'][batch_idx]).astype(np.float32)
    obj_boxes = _as_numpy(pred['obj_boxes'][batch_idx]).astype(np.float32)

    sub_prob = torch.softmax(pred['sub_logits'][batch_idx], dim=-1).cpu().numpy()
    obj_prob = torch.softmax(pred['obj_logits'][batch_idx], dim=-1).cpu().numpy()
    rel_prob = torch.softmax(pred['rel_logits'][batch_idx], dim=-1).cpu().numpy()

    if sub_prob.shape[1] == entity_nums + 1:
        sub_prob = sub_prob[:, :-1]
    if obj_prob.shape[1] == entity_nums + 1:
        obj_prob = obj_prob[:, :-1]
    if rel_prob.shape[1] == rel_nums + 1:
        rel_prob = rel_prob[:, 1:]

    pred_sub_scores = sub_prob.max(axis=1)
    pred_sub_labels = sub_prob.argmax(axis=1)
    pred_obj_scores = obj_prob.max(axis=1)
    pred_obj_labels = obj_prob.argmax(axis=1)

    nq = min(len(sub_boxes), len(obj_boxes), len(pred_sub_labels), len(pred_obj_labels), len(rel_prob))

    triplets = []
    for i in range(nq):
        if multiple_preds:
            rel_order = np.argsort(-rel_prob[i])
            for r in rel_order:
                score = float(pred_sub_scores[i] * pred_obj_scores[i] * rel_prob[i, r])
                triplets.append({
                    'sub_label': int(pred_sub_labels[i]),
                    'obj_label': int(pred_obj_labels[i]),
                    'rel_label': int(r),
                    'sub_box': sub_boxes[i],
                    'obj_box': obj_boxes[i],
                    'score': score,
                })
        else:
            r = int(np.argmax(rel_prob[i]))
            r_score = float(np.max(rel_prob[i]))
            score = float(pred_sub_scores[i] * pred_obj_scores[i] * r_score)
            triplets.append({
                'sub_label': int(pred_sub_labels[i]),
                'obj_label': int(pred_obj_labels[i]),
                'rel_label': int(r),
                'sub_box': sub_boxes[i],
                'obj_box': obj_boxes[i],
                'score': score,
            })

    triplets.sort(key=lambda x: x['score'], reverse=True)
    return triplets


def _match_triplets(pred_triplets, gt_triplets, iou_thresh=0.5):
    matched_gt = set()
    for pred in pred_triplets:
        for g_idx, gt in enumerate(gt_triplets):
            if g_idx in matched_gt:
                continue
            if _is_triplet_match(pred, gt, iou_thresh):
                matched_gt.add(g_idx)
                break
    return matched_gt


def _is_triplet_match(pred, gt, iou_thresh=0.5):
    if pred['sub_label'] != gt['sub_label']:
        return False
    if pred['obj_label'] != gt['obj_label']:
        return False
    if pred['rel_label'] != gt['rel_label']:
        return False

    if _bbox_iou(pred['sub_box'], gt['sub_box']) < iou_thresh:
        return False
    if _bbox_iou(pred['obj_box'], gt['obj_box']) < iou_thresh:
        return False
    return True


def _bbox_iou(box1, box2, eps=1e-12):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter

    return inter / max(union, eps)


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


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)